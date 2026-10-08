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
# - When Cargo.toml patches burn-stack to a local path (`[patch."…burn-stack
#   .git"]`), the build uses that checkout. Then the hash of the tag also
#   covers its commit and its changes, and the build info holds them.
# Each script copies bin/ts-cuda.txt into the directories that it creates
# (`stamp` in args.sh). stamp.sh does it for a directory made by hand.
set -euo pipefail
cd "$(dirname "$0")"
repo=$(git rev-parse --show-toplevel)
ver=$(sed -n 's/^version = "\(.*\)"/\1/p' "$repo/Cargo.toml" | head -1)
commit=$(git -C "$repo" rev-parse --short HEAD)
# The changes of a checkout: `git diff HEAD`, then the untracked files.
changes_of() {
  git -C "$1" diff HEAD
  git -C "$1" ls-files --others --exclude-standard -z |
    xargs -0 -r -I{} git -C "$1" diff --no-index /dev/null {} || true
}
changes=$(changes_of "$repo")
# The local burn-stack of an active patch (empty when there is none).
stack=$(sed -n '/^\[patch\."https:\/\/github.com\/swfsql\/burn-stack.git"\]/,/^$/ s/^burn-stack = { path = "\(.*\)" }/\1/p' "$repo/Cargo.toml")
[ -n "$stack" ] && stack=$(cd "$repo" && cd "$stack" && pwd)
stack_changes=
if [ -n "$stack" ]; then
  stack_changes="burn-stack $(git -C "$stack" rev-parse HEAD)
$(changes_of "$stack")"
fi
tag=v$ver-$commit
all="$changes$stack_changes"
[ -n "$changes" ] || [ -n "$stack" ] && tag=$tag-dirty-$(printf '%s' "$all" | sha256sum | cut -c1-8)
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
    if [ -n "$stack" ]; then
      echo "burn-stack: local patch $stack at $(git -C "$stack" log -1 --format='%H %s')"
    else
      echo "burn-stack: $(sed -n 's/^burn-stack = .*rev = "\([0-9a-f]*\)".*/\1/p' "$repo/Cargo.toml" | head -1)"
    fi
    echo "features: default,backend-cuda"
    echo "== git status --short"
    git -C "$repo" status --short
    echo "== changes (git diff HEAD, then the untracked files)"
    printf '%s\n' "$changes"
    if [ -n "$stack" ]; then
      echo "== burn-stack git status --short"
      git -C "$stack" status --short
      echo "== burn-stack changes (git diff HEAD, then the untracked files)"
      changes_of "$stack"
    fi
  } > "bin/ts-cuda-$tag.txt"
fi
cp "bin/ts-cuda-$tag" bin/ts-cuda.new && mv bin/ts-cuda.new bin/ts-cuda
cp "bin/ts-cuda-$tag.txt" bin/ts-cuda.txt
echo "installed bin/ts-cuda = $tag"
