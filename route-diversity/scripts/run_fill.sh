#!/bin/bash
# 프롬프트는 써놓고 생성을 안 한 두 갈래를 채운다.
#   group3  254 중 219 생성 -> 35 남음   (다르게 묶기, 수율 0.78~0.92)
#   golf292 292 중 183 생성 -> 109 남음  (짧게, 수율 0.52)
#
# 수율 높은 쪽을 먼저 끝낸다. 예산이 끊겨도 이득이 큰 쪽은 끝나 있다.
#
# 게이트·재시도·PAUSE 처리는 run_cg_all.sh와 같다. 다른 것은 arm 목록뿐이다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/fill.log
SESSION_MAX=80
WEEK_MAX=90
CHUNK=10
ARMS="group3 golf292"

usage(){ timeout 120 claude -p /usage 2>/dev/null; }
sess(){ echo "$1" | sed -n 's/.*session:[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }
week(){ echo "$1" | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

left(){ python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
a = sys.argv[1]
want = set(json.load(open(f"/hdd_data/yunho/cctmp/{a}_feedback.json")))
d = Path(f"/hdd_data/yunho/solar-traj/maker/{a}")
have = {p.name for p in d.iterdir() if p.is_dir() and (p/"grid_maker.py").exists()} \
       if d.is_dir() else set()
print(" ".join(sorted(want - have)))
PY
}

wait_budget(){
  while :; do
    if [ -f /hdd_data/yunho/cctmp/PAUSE ]; then
      echo "  [$(TZ=Asia/Seoul date '+%m-%d %H:%M')] PAUSE 걸림, 5분 대기" >> "$LOG"
      sleep 300; continue
    fi
    U=$(usage); S=$(sess "$U"); W=$(week "$U")
    if [ -n "$W" ] && [ "$W" -ge "$WEEK_MAX" ]; then
      echo "  [$(TZ=Asia/Seoul date '+%m-%d %H:%M')] weekly ${W}% >= ${WEEK_MAX}, waiting 30m" >> "$LOG"
      sleep 1800; continue
    fi
    if [ -z "$S" ] || [ "$S" -lt "$SESSION_MAX" ]; then
      echo "$S/$W"; return 0
    fi
    echo "  [$(TZ=Asia/Seoul date '+%m-%d %H:%M')] session ${S}% >= ${SESSION_MAX}, waiting 20m" >> "$LOG"
    sleep 1200
  done
}

echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] fill start" >> "$LOG"
for arm in $ARMS; do
  tries=0
  while [ "$tries" -lt 4 ]; do
    ALL=$(left "$arm")
    [ -z "$ALL" ] && { echo "  $arm: done" >> "$LOG"; break; }
    BUD=$(wait_budget)
    TODO=$(echo $ALL | tr ' ' '\n' | head -n "$CHUNK" | tr '\n' ' ')
    BEFORE=$(echo $ALL | wc -w)
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] $arm  ${BUD}  ${BEFORE} left, chunk $(echo $TODO | wc -w)" >> "$LOG"
    T0=$(date +%s)
    /hdd_data/yunho/cctmp/limited.sh 12G 300 \
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
      --output_subdir "$arm" \
      --task_feedback_file "/hdd_data/yunho/cctmp/${arm}_feedback.json" \
      --write_only_valid --overwrite \
      --parallel 3 --attempts 2 --llm_timeout 1800 \
      >> "/hdd_data/yunho/cctmp/fill_$arm.log" 2>&1
    AFTER=$(left "$arm"); AFTER=$(echo $AFTER | wc -w)
    tries=0
    DT=$(( $(date +%s) - T0 ))
    if [ "$AFTER" -ge "$BEFORE" ] && [ "$DT" -lt 300 ]; then
      echo "  no progress in ${DT}s -- throttled, waiting 20m" >> "$LOG"
      sleep 1200
    else
      [ "$AFTER" -ge "$BEFORE" ] && tries=$((tries + 1))
      sleep 5
    fi
  done
done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] fill end" >> "$LOG"
