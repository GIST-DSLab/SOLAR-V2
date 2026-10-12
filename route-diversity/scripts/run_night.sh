#!/bin/bash
# 남은 갈래를 순서대로 끝까지. 아침에 사람이 쓸 수 있게 04:00에 선다.
#
# 세션은 5시간 롤링 창이다. 04:00에 멈추면 09:00에는 창이 다 굴러가 0%가
# 되어 있다. 08:00에 멈추면 09:00에도 직전 네 시간치가 남아 있다 -- 그래서
# 기준은 08시가 아니라 04시다.
#
# 순서는 수율순이다. 예산이 중간에 끊겨도 이득 큰 쪽은 끝나 있다.
#   group3      35   다르게 묶기, 수율 0.78~0.92 (프롬프트 기존 것)
#   golf292    109   짧게, 수율 0.52            (프롬프트 기존 것)
#   cristianoc 400   7번째 참조, 400/400 커버
#   sakana     380   8번째 참조
#   cctofu      46   9번째 참조, 커버 얇음
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/night.log
SESSION_MAX=80
WEEK_MAX=90
CHUNK=10
STOP_HOUR=4          # 이 시각부터 09시까지는 아무것도 하지 않는다
RESUME_HOUR=9
# arm:프롬프트파일 (프롬프트 이름이 arm 이름과 다른 경우가 있다)
ARMS="group3:group3 golf292:golf292 cg_cristianoc:cg_cristianoc cg_sakana:cg_sakana cg_cctofu:cg_cctofu"

usage(){ timeout 120 claude -p /usage 2>/dev/null; }
sess(){ echo "$1" | sed -n 's/.*session:[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }
week(){ echo "$1" | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

quiet_hours(){
  H=$(TZ=Asia/Seoul date +%-H)
  [ "$H" -ge "$STOP_HOUR" ] && [ "$H" -lt "$RESUME_HOUR" ]
}

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
    if quiet_hours; then
      echo "  [$(TZ=Asia/Seoul date '+%m-%d %H:%M')] 04~09시 정지 시간, 종료" >> "$LOG"
      return 1
    fi
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

echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] night start" >> "$LOG"
for pair in $ARMS; do
  arm=${pair%%:*}; fb=${pair##*:}
  tries=0
  while [ "$tries" -lt 4 ]; do
    quiet_hours && { echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] night stop (quiet hours)" >> "$LOG"; exit 0; }
    ALL=$(left "$fb")
    [ -z "$ALL" ] && { echo "  $arm: done" >> "$LOG"; break; }
    BUD=$(wait_budget) || exit 0
    TODO=$(echo $ALL | tr ' ' '\n' | head -n "$CHUNK" | tr '\n' ' ')
    BEFORE=$(echo $ALL | wc -w)
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] $arm  ${BUD}  ${BEFORE} left, chunk $(echo $TODO | wc -w)" >> "$LOG"
    T0=$(date +%s)
    /hdd_data/yunho/cctmp/limited.sh 12G 300 \
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
      --output_subdir "$arm" \
      --task_feedback_file "/hdd_data/yunho/cctmp/${fb}_feedback.json" \
      --write_only_valid --overwrite \
      --parallel 3 --attempts 2 --llm_timeout 1800 \
      >> "/hdd_data/yunho/cctmp/night_$arm.log" 2>&1
    AFTER=$(left "$fb"); AFTER=$(echo $AFTER | wc -w)
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
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] night end" >> "$LOG"
