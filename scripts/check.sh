#!/usr/bin/env bash
# The check of this repo. CI runs it and skips the host-path step.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${CI:-}" ]; then
    # This box gives the two roots, so this file names no path of a box.
    repo_root="/$(pwd | cut -d/ -f2)"
    home_root="/$(printf '%s' "$HOME" | cut -d/ -f2)"
    if git grep --untracked -n -I -E "(^|[^[:alnum:]_./-])($repo_root|$home_root)(/|\$)"; then
        echo "error: the lines above hold a host path. Write a path relative to the repo." >&2
        exit 1
    fi
fi

cargo check
cargo test --lib -- --test-threads=1
cargo doc --all --no-deps
