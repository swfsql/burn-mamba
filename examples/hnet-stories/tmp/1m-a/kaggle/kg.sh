#!/usr/bin/env bash
# usage: kaggle/kg.sh <kaggle CLI args>
# Runs the kaggle CLI (1.7.4.5) with the access token of the repo
# (`.kaggle/access_token`, gitignored, a `KGAT_` token). That CLI reads only
# a username and a key, but its kagglesdk client sends `KAGGLE_API_TOKEN`
# as a Bearer token first. So the token goes in KAGGLE_API_TOKEN, and the
# username and key only pass the check of the CLI (the username also names
# the owner of the kernels and datasets: KAGGLE_OWNER, see kaggle/owner.txt).
# The token never appears in a command line or a log.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
KAGGLE_API_TOKEN=$(tr -d '\n' < "$repo/.kaggle/access_token")
owner=unknown
[ -f "$here/owner.txt" ] && owner=$(tr -d '\n' < "$here/owner.txt")
export KAGGLE_API_TOKEN KAGGLE_USERNAME=$owner KAGGLE_KEY=unused
exec kaggle "$@"
