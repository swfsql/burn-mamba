#!/usr/bin/env bash
# usage: build.sh
# Builds the example from the burn-mamba checkout (`default,backend-cuda`) and
# installs it as bin/ts-cuda. A host job: run nothing else on the host.
# - The tag is v<version>-<commit>, + -dirty-<hash> when the checkout has
#   changes (the hash of the diff and of the untracked files).
# - bin/ts-cuda-<tag> keeps the binary, and bin/ts-cuda-<tag>.txt its build
#   info: the tag, the commit, the burn and burn-stack revs, `git status`
#   and the full diff. bin/ts-cuda.txt is the info of the installed build.
# - A tag that exists already is not built again (the same source).
# Each script copies bin/ts-cuda.txt into the directories that it creates
# (`stamp` in args.sh). stamp.sh does it for a directory made by hand.
set -euo pipefail
cd "$(dirname "$0")"
repo=$(git rev-parse --show-toplevel)
ver=$(sed -n 's/^version = "\(.*\)"/\1/p' "$repo/Cargo.toml" | head -1)
commit=$(git -C "$repo" rev-parse --short HEAD)
changes=$(git -C "$repo" diff HEAD
  git -C "$repo" ls-files --others --exclude-standard -z |
    xargs -0 -r -I{} git -C "$repo" diff --no-index /dev/null {} || true)
tag=v$ver-$commit
[ -n "$changes" ] && tag=$tag-dirty-$(printf '%s' "$changes" | sha256sum | cut -c1-8)
if [ -e "bin/ts-cuda-$tag" ]; then
  echo "exists: bin/ts-cuda-$tag (not built again)"
else
  (cd "$repo" && cargo build --release --example tiny-stories --features default,backend-cuda)
  cp "$repo/target/release/examples/tiny-stories" "bin/ts-cuda-$tag"
fi
if [ ! -e "bin/ts-cuda-$tag.txt" ]; then
  {
    echo "tag: $tag"
    echo "built: $(date -Is -r "bin/ts-cuda-$tag")"
    echo "burn-mamba: $(git -C "$repo" log -1 --format='%H %s') (version $ver)"
    echo "burn: $(sed -n 's/^burn = .*rev = "\([0-9a-f]*\)".*/\1/p' "$repo/Cargo.toml" | head -1)"
    echo "burn-stack: $(sed -n 's/^burn-stack = .*rev = "\([0-9a-f]*\)".*/\1/p' "$repo/Cargo.toml" | head -1)"
    echo "features: default,backend-cuda"
    echo "== git status --short"
    git -C "$repo" status --short
    echo "== changes (git diff HEAD, then the untracked files)"
    printf '%s\n' "$changes"
  } > "bin/ts-cuda-$tag.txt"
fi
cp "bin/ts-cuda-$tag" bin/ts-cuda.new && mv bin/ts-cuda.new bin/ts-cuda
cp "bin/ts-cuda-$tag.txt" bin/ts-cuda.txt
echo "installed bin/ts-cuda = $tag"
