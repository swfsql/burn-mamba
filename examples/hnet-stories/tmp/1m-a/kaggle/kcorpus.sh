#!/usr/bin/env bash
# usage: kaggle/kcorpus.sh   (once; a new version only if the caches change)
# Uploads the cased text caches of the corpus as the private Kaggle dataset
# <owner>/hnet-corpus: the train split of the search (262144 stories), the
# valid split of the metric (256) and of the context probe (2048). With them
# in the cache directory, the binary reads no parquet and needs no internet.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
owner=$(tr -d '\n' < "$here/owner.txt")
src=$HOME/.cache/burn-dataset/tinystories-gpt4-clean
dir=/tmp/kaggle-corpus
mkdir -p "$dir"
for f in train-262144-cased.txt valid-256-cased.txt valid-2048-cased.txt; do
  cp "$src/$f" "$dir/"
done
cat > "$dir/dataset-metadata.json" <<EOF
{"title": "hnet-corpus", "id": "$owner/hnet-corpus", "licenses": [{"name": "CDLA-Sharing-1.0"}]}
EOF
if "$here/kg.sh" datasets status "$owner/hnet-corpus" > /dev/null 2>&1; then
  "$here/kg.sh" datasets version -p "$dir" -m "the cased caches"
else
  "$here/kg.sh" datasets create -p "$dir"
fi
