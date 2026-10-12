#!/bin/bash
# BARC 참조 생성. golf84가 끝난 뒤에 시작하고, 끊겨도 이어서 돈다.
# 파일럿(경로 1가지뿐인 29개) 먼저, 예산이 남으면 나머지 131개.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/barc.log

week(){ timeout 120 claude -p /usage 2>/dev/null \
        | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

left(){ python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
want = set(json.load(open(sys.argv[1])))
d = Path("/hdd_data/yunho/solar-traj/maker/barc1")
have = {p.name for p in d.iterdir() if p.is_dir() and (p/"grid_maker.py").exists()} \
       if d.is_dir() else set()
print(" ".join(sorted(want - have)))
PY
}

# golf84가 끝날 때까지 기다린다 -- 동시에 돌면 예산을 같이 먹는다.
while pgrep -f "[g]en_rearc_makers_llm.py.*golf84" > /dev/null; do sleep 120; done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] golf84 done, starting barc" >> "$LOG"

for LIST in /hdd_data/yunho/cctmp/barc_pilot.json /hdd_data/yunho/cctmp/barc_rest.json; do
  for attempt in 1 2 3 4 5 6; do
    TODO=$(left "$LIST")
    [ -z "$TODO" ] && break
    W=$(week)
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] $(basename $LIST) attempt $attempt, weekly ${W:-?}%, $(echo $TODO | wc -w) left" >> "$LOG"
    if [ -n "$W" ] && [ "$W" -ge 90 ]; then
      echo "  weekly ${W}% >= 90, stopping" >> "$LOG"; exit 0
    fi
    python3 pipeline/gen_rearc_makers_llm.py \
      --tasks $TODO \
      --output_subdir barc1 \
      --task_feedback_file /hdd_data/yunho/cctmp/barc_feedback.json \
      --write_only_valid --overwrite \
      --parallel 4 --attempts 2 --llm_timeout 1800 \
      >> "$LOG" 2>&1
    sleep 10
  done
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] barc loop end" >> "$LOG"
