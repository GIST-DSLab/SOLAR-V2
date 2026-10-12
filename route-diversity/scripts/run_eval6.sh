#!/bin/bash
# eval_cg6를 청크로 끝까지. 청크마다 프로세스가 새로 뜨므로 누수가 쌓이지 않는다.
cd /hdd_data/yunho/cctmp
export TMPDIR=/hdd_data/yunho/cctmp SOLAR_CHUNK=25
LOG=/hdd_data/yunho/cctmp/eval_cg6.log
for i in $(seq 1 30); do
  out=$(python3 -I eval_cg6.py 2>&1 | tail -3)
  echo "[$(date +%m-%d\ %H:%M)] $out" >> "$LOG"
  case "$out" in *done*) echo "[$(date +%m-%d\ %H:%M)] all done" >> "$LOG"; break;; esac
done
