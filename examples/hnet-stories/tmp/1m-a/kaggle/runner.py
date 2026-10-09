"""The Kaggle runner of the hnet-stories search (`kpush.sh` pushes it as a
private GPU kernel, `kpull.sh` fetches its outputs).

The kernel reads two private datasets: <owner>/hnet-corpus (the cased text
caches) and <owner>/hnet-payload (`payload.tar.gz`: the binary, the scripts,
`cfg/`, the source runs of the batch, `kqueue.txt` and `kaggle-budget.txt`;
Kaggle can serve it unpacked).
It runs the lines of `kqueue.txt` in the workspace, as `queue.sh` does, on
one lane per GPU (Kaggle's "T4 × 2" has two): each lane takes the next
line and runs it with `CUDA_VISIBLE_DEVICES` set to its GPU. So the lines
run in parallel, in no fixed order: steps that depend on each other go on
one line, joined with `&&`. It logs `start` / `done` / `fail` / `skip`
(with the GPU) into `kqueue.log`. A line that would start after the budget
(seconds from the start of the kernel) is skipped. After each line, the new
runs, the context copies and the text outputs go to /kaggle/working (the
outputs of the kernel), each run with a `host.txt`. The binary stays out of
the outputs.
"""
import glob
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time

T0 = time.time()
LOCK = threading.Lock()
INPUT = "/kaggle/input"
OUT = "/kaggle/working"
WS = "/tmp/ws"
CORPUS = ("train-262144-cased.txt", "valid-256-cased.txt", "valid-2048-cased.txt")
TEXT_OUTPUTS = ("probes.txt", "knobs.txt", "ctxg.txt", "speeds.txt")
# The batch that this kernel must run: kpush.sh writes it into the pushed
# copy. The payload must carry its marker `batch-<id>.txt` (Kaggle can attach
# an older version of the dataset).
EXPECTED_BATCH = None


def log(msg):
    line = f"{msg}\n"
    with LOCK:
        print(line, end="", flush=True)
        with open(os.path.join(OUT, "kqueue.log"), "a") as f:
            f.write(line)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def find(name, required=True):
    """The path of a file of an attached dataset (`None` when it is not
    there and not `required`)."""
    hits = glob.glob(f"{INPUT}/**/{name}", recursive=True)
    if not hits and required:
        tree = "\n".join(sorted(glob.glob(f"{INPUT}/*/*")) + sorted(glob.glob(f"{INPUT}/*/*/*"))[:50])
        sys.exit(f"missing {name} under {INPUT}. The inputs:\n{tree}")
    return hits[0] if hits else None


def unpack_payload():
    """The payload into WS. Kaggle can keep the archive of the dataset, or
    unpack it (then the tree holds `kqueue.txt`). An unpacked tree can lose
    the exec bits, so they are set again."""
    archive = find("payload.tar.gz", required=False)
    if archive:
        with tarfile.open(archive) as tar:
            tar.extractall(WS)
    else:
        shutil.copytree(os.path.dirname(find("kqueue.txt")), WS, dirs_exist_ok=True)
    for path in glob.glob(f"{WS}/*.sh") + glob.glob(f"{WS}/*.py") + [f"{WS}/bin/hn-cuda"]:
        os.chmod(path, 0o755)


def host():
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    return f"host: kaggle ({os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '?')})\ngpu: {gpu}\nstarted: {now()}\n"


def sync(before, host_txt):
    """Copy the new runs, the context copies and the text outputs to OUT."""
    for name in sorted(set(os.listdir(os.path.join(WS, "runs"))) - before):
        dst = os.path.join(OUT, "runs", name)
        shutil.copytree(os.path.join(WS, "runs", name), dst, dirs_exist_ok=True)
        with open(os.path.join(dst, "host.txt"), "w") as f:
            f.write(host_txt)
    for root in ("ctxg", "ctx"):
        if os.path.isdir(os.path.join(WS, root)):
            shutil.copytree(os.path.join(WS, root), os.path.join(OUT, root), dirs_exist_ok=True)
    for name in TEXT_OUTPUTS:
        if os.path.exists(os.path.join(WS, name)):
            shutil.copy(os.path.join(WS, name), OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    if EXPECTED_BATCH is not None and find(f"batch-{EXPECTED_BATCH}.txt", required=False) is None:
        markers = sorted(os.path.basename(p) for p in glob.glob(f"{INPUT}/**/batch-*.txt", recursive=True))
        log(f"stale payload: expected batch-{EXPECTED_BATCH}.txt, found {markers or 'no marker'}")
        sys.exit(2)
    cache = os.path.expanduser("~/.cache/burn-dataset/tinystories-gpt4-clean")
    os.makedirs(cache, exist_ok=True)
    for name in CORPUS:
        shutil.copy(find(name), cache)
    unpack_payload()
    os.makedirs(os.path.join(WS, "runs"), exist_ok=True)
    budget = float(open(os.path.join(WS, "kaggle-budget.txt")).read())
    host_txt = host()
    with open(os.path.join(OUT, "host.txt"), "w") as f:
        f.write(host_txt)
    log(f"host {host_txt.splitlines()[1]}")
    before = set(os.listdir(os.path.join(WS, "runs")))
    jobs = queue.Queue()
    for line in open(os.path.join(WS, "kqueue.txt")).read().splitlines():
        if line.strip() and not line.startswith("#"):
            jobs.put(line)

    def lane(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                line = jobs.get_nowait()
            except queue.Empty:
                return
            if time.time() - T0 > budget:
                log(f"skip {now()} gpu{gpu} (budget {budget:.0f} s) {line}")
                continue
            log(f"start {now()} gpu{gpu} {line}")
            rc = subprocess.run(["bash", "-c", line], cwd=WS, env=env).returncode
            log(f"{'done' if rc == 0 else f'fail rc={rc}'} {now()} gpu{gpu} {line}")
            with LOCK:
                sync(before, host_txt)

    n_gpus = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("\n")) - 1)
    lanes = [threading.Thread(target=lane, args=(gpu,)) for gpu in range(n_gpus)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    log(f"empty {now()}")


if __name__ == "__main__":
    main()
