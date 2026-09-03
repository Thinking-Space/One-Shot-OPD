#!/bin/bash
# Fetch a ModelScope model or dataset repo over plain HTTP.
#
#   bash data/prep/ms_fetch.sh models   agentica-org/DeepCoder-1.5B-Preview "$MODEL_BASE/DeepCoder-1.5B-Preview"
#   bash data/prep/ms_fetch.sh datasets BAAI/TACO                          "$TACO_CACHE"  train
#
# Why not `pip install modelscope`: its dependency tree pulls a different
# transformers, and this cluster's combination (transformers 5.10.4) is what the
# running jobs are validated against -- opencompass hit the same trap and was
# solved the same way (take the bytes, not the package).
#
# Why not hf-mirror: as of 2026-08-29 hf-mirror.com resets the connection
# (curl exit 35, http 000). It used to answer 200; that is stale.
# ModelScope answers 200 and carries both assets we are missing.
#
# Args: <models|datasets> <owner/name> <dest-dir> [subdir, default: repo root]
set -uo pipefail

KIND=${1:?models or datasets}
REPO=${2:?owner/name}
DEST=${3:?destination directory}
ROOT=${4:-}

API="https://www.modelscope.cn/api/v1/${KIND}/${REPO}"
CURL="/usr/bin/curl"   # conda's curl has a broken CA bundle

mkdir -p "$DEST"

# The two repo kinds want different listing endpoints, and neither recurses:
# models answer /repo/files flat, datasets only answer /repo/tree and need an
# explicit Root= per subdirectory (Recursive=True is rejected).
if [ "$KIND" = models ]; then
    LIST_URL="${API}/repo/files?Revision=master"
else
    LIST_URL="${API}/repo/tree?Revision=master&Root=${ROOT}"
fi

files=$("$CURL" -sS --max-time 60 "$LIST_URL" | python3 -c "
import json,sys
d = json.load(sys.stdin).get('Data') or {}
for f in (d.get('Files') or d.get('files') or []):
    if (f.get('Type') or f.get('type')) == 'tree':
        continue
    p = f.get('Path') or f.get('Name')
    if p:
        print(p)
")

[ -n "$files" ] || { echo "no files listed for ${KIND}/${REPO}" >&2; exit 1; }

rc=0
for f in $files; do
    case "$f" in .gitattributes|README.md|configuration.json|*.py) continue ;; esac
    out="${DEST}/${f}"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then echo "have  $f"; continue; fi
    echo "get   $f"
    # -C - resumes a partial file; a pod restart mid-download is expected here.
    "$CURL" -sS -L -C - --retry 5 --retry-delay 5 --max-time 3600 \
        -o "$out" "${API}/repo?Revision=master&FilePath=${f}" || { rc=1; echo "FAILED $f" >&2; }
done
exit $rc
