#!/bin/bash
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/recount.log
while pgrep -f "cctmp/run_queue2.sh" > /dev/null || pgrep -f "[g]en_rearc_makers_llm.py" > /dev/null; do
  sleep 180
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue finished, final recount" >> "$LOG"
rm -f /hdd_data/yunho/cctmp/diversity_final.json
sed 's#diversity_all.json#diversity_final.json#' /hdd_data/yunho/cctmp/diversity_all.py \
  > /hdd_data/yunho/cctmp/diversity_final.py
for a in 1 2 3 4; do
  python3 /hdd_data/yunho/cctmp/diversity_final.py >> "$LOG" 2>&1
  N=$(python3 -c '
import json
from pathlib import Path
p = Path("/hdd_data/yunho/cctmp/diversity_final.json")
print(len(json.loads(p.read_text())) if p.is_file() else 0)
')
  [ "$N" -ge 400 ] && break
  sleep 5
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] final recount done" >> "$LOG"
