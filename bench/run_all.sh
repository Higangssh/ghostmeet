#!/bin/sh
# Full benchmark, one engine per process so memory numbers do not leak between runs.
# Run from the repo root: sh bench/run_all.sh
set -e
PY="bench/.venv/bin/python -I"
ENGINES="sensevoice wcpp-small fw-small wcpp-large-v3-turbo fw-large-v3-turbo wcpp-small-cpu"
for e in $ENGINES; do $PY bench/run_utterances.py "$e"; done
for e in sensevoice wcpp-small fw-small wcpp-large-v3-turbo fw-large-v3-turbo; do
  $PY bench/run_stream.py "$e" --mode backlog
  $PY bench/run_stream.py "$e" --mode live
done
$PY bench/report.py
