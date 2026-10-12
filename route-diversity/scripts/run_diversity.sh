#!/bin/bash
# 끊겨도 이어서 돈다: 과제마다 저장하고, 죽으면 남은 것부터 다시 시작한다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/diversity_all.log
for attempt in 1 2 3 4 5 6; do
  echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] attempt $attempt" >> "$LOG"
  python3 /hdd_data/yunho/cctmp/diversity_all.py >> "$LOG" 2>&1
  LEFT=$(python3 -c '
import json
from pathlib import Path
p = Path("/hdd_data/yunho/cctmp/diversity_all.json")
print(400 - len(json.loads(p.read_text())) if p.is_file() else 400)
')
  [ "$LEFT" -le 0 ] && { echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] finished" >> "$LOG"; break; }
  sleep 5
done
