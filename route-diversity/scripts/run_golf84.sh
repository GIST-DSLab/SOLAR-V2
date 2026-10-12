#!/bin/bash
# codegolf 생성 84개. 끊겨도 이어서 돈다 -- --write_only_valid는 통과한 것만
# 쓰므로, 이미 쓰인 과제를 빼고 남은 것만 다시 돌리면 된다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/golf84.log
OUT=maker/golf84

left(){ python3 - <<'PY'
import json
from pathlib import Path
want = set(json.load(open("/hdd_data/yunho/cctmp/solo84.json")))
have = {p.name for p in Path("/hdd_data/yunho/solar-traj/maker/golf84").iterdir()
        if p.is_dir() and (p / "grid_maker.py").exists()} \
       if Path("/hdd_data/yunho/solar-traj/maker/golf84").is_dir() else set()
print(" ".join(sorted(want - have)))
PY
}

week(){ timeout 120 claude -p /usage 2>/dev/null \
        | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

for attempt in 1 2 3 4 5 6 7 8; do
  TODO=$(left)
  [ -z "$TODO" ] && { echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] all done" >> "$LOG"; break; }
  W=$(week)
  echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] attempt $attempt, weekly ${W:-?}%, $(echo $TODO | wc -w) left" >> "$LOG"
  if [ -n "$W" ] && [ "$W" -ge 90 ]; then
    echo "  weekly ${W}% >= 90, stopping" >> "$LOG"; break
  fi
  python3 pipeline/gen_rearc_makers_llm.py \
    --tasks $TODO \
    --output_subdir golf84 \
    --task_feedback_file /hdd_data/yunho/cctmp/golf84_feedback.json \
    --write_only_valid --overwrite \
    --parallel 4 --attempts 2 --llm_timeout 1800 \
    >> "$LOG" 2>&1
  sleep 10
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] loop end" >> "$LOG"
