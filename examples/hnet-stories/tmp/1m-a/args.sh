# Sourced by the scripts that run bin/hn-cuda.
# split_args "$@": sets app=(the AppArgs flags) and ex=(the example flags after
# `--`), and:
# - adds `--max-vram $VRAM_MAX_MIB` (default 3750) to ex unless the caller
#   gave one. 3750 MiB of pools + ~140 MiB of CUDA context keeps the process
#   under 3.8 GiB (the search limit).
# - adds `--grad-checkpointing` (the default of the search: less VRAM, a
#   little more time, the same losses) unless the caller gave it, or gave
#   `--no-gc`. `--no-gc` is not a flag of the example: it is removed here.
# - adds `--vram-cleanup-every 1` unless the caller gave a period: the pages
#   that the pools drop never wait past a window, so the `peak` of the
#   `prof` lines plus the CUDA context (~140 MiB) is the VRAM of the
#   process. R-N15-B5: the same peak as the old schedule, ~10% faster.
# Call as: bin/hn-cuda … "${app[@]}" -- "${ex[@]}".
# stamp <dir>: writes <dir>/build.txt, the build info of bin/hn-cuda (see
# build.sh). bin_tag: the tag of bin/hn-cuda, for the `==` lines of the logs.
stamp() { cp bin/hn-cuda.txt "$1/build.txt"; }
bin_tag() { sed -n 's/^tag: //p' bin/hn-cuda.txt; }
split_args() {
  app=() ex=()
  local seen=0 gc=1 a
  for a in "$@"; do
    if [ "$seen" = 0 ] && [ "$a" = "--" ]; then seen=1
    elif [ "$seen" = 0 ]; then app+=("$a")
    elif [ "$a" = "--no-gc" ]; then gc=0
    elif [ "$a" = "--grad-checkpointing" ]; then :
    else ex+=("$a")
    fi
  done
  [ "$gc" = 1 ] && ex+=(--grad-checkpointing)
  case " ${ex[*]} " in
    *" --max-vram "*) ;;
    *) ex+=(--max-vram "${VRAM_MAX_MIB:-3750}") ;;
  esac
  case " ${ex[*]} " in
    *" --vram-cleanup-every "*) ;;
    *) ex+=(--vram-cleanup-every 1) ;;
  esac
}
