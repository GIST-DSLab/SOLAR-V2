#!/bin/bash
# 생성 큐가 끝나면 400개 경로 가짓수를 다시 센다. LLM을 안 쓴다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/recount.log
while pgrep -f "cctmp/run_queue.sh" > /dev/null || pgrep -f "[g]en_rearc_makers_llm.py" > /dev/null; do
  sleep 180
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue finished, recounting" >> "$LOG"
rm -f /hdd_data/yunho/cctmp/diversity_after.json
sed 's#diversity_all.json#diversity_after.json#' /hdd_data/yunho/cctmp/diversity_all.py \
  > /hdd_data/yunho/cctmp/diversity_after.py
for a in 1 2 3 4; do
  python3 /hdd_data/yunho/cctmp/diversity_after.py >> "$LOG" 2>&1
  N=$(python3 -c '
import json
from pathlib import Path
p = Path("/hdd_data/yunho/cctmp/diversity_after.json")
print(len(json.loads(p.read_text())) if p.is_file() else 0)
')
  [ "$N" -ge 400 ] && break
  sleep 5
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] recount done" >> "$LOG"
