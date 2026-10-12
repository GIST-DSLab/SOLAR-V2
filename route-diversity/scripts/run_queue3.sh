#!/bin/bash
# 생성 큐 v2. 앞 판에서 밟은 두 가지를 고친다.
#   1) 세션 사용량도 본다. 전에는 주간만 보고 세션은 안 봐서, 세션이 꽉 찬
#      채로 호출을 계속 던졌다.
#   2) 즉시 실패한 라운드는 재시도 횟수로 세지 않는다. 한도에 걸리면 호출이
#      바로 떨어지므로, 전에는 재시도 6번이 2분 만에 타고 큐가 죽었다.
#      일을 실제로 한 라운드만 센다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/queue.log
SESSION_MAX=85
WEEK_MAX=88

usage(){ timeout 120 claude -p /usage 2>/dev/null; }
sess(){ echo "$1" | sed -n 's/.*session:[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }
week(){ echo "$1" | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

left(){ python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
want = set(json.load(open(sys.argv[1])))
d = Path("/hdd_data/yunho/solar-traj/maker") / sys.argv[2]
have = {p.name for p in d.iterdir() if p.is_dir() and (p/"grid_maker.py").exists()} \
       if d.is_dir() else set()
print(" ".join(sorted(want - have)))
PY
}

# 여유가 생길 때까지 기다린다. 기다리는 동안은 아무것도 쓰지 않는다.
wait_budget(){
  while :; do
    U=$(usage); S=$(sess "$U"); W=$(week "$U")
    if [ -n "$W" ] && [ "$W" -ge "$WEEK_MAX" ]; then
      echo "  weekly ${W}% >= ${WEEK_MAX}, queue stops" >> "$LOG"; return 1
    fi
    if [ -z "$S" ] || [ "$S" -lt "$SESSION_MAX" ]; then
      echo "$S $W"; return 0
    fi
    echo "  [$(TZ=Asia/Seoul date '+%H:%M')] session ${S}% >= ${SESSION_MAX}, waiting 20m" >> "$LOG"
    sleep 1200
  done
}

stage(){  # $1=목록 $2=출력폴더 $3=피드백파일(없으면 "-")
  tries=0
  while [ "$tries" -lt 6 ]; do
    TODO=$(left "$1" "$2")
    [ -z "$TODO" ] && { echo "  $(basename $1) -> $2 : done" >> "$LOG"; return 0; }
    BUD=$(wait_budget) || return 1
    BEFORE=$(echo $TODO | wc -w)
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] $(basename $1) -> $2, session/weekly ${BUD}, ${BEFORE} left" >> "$LOG"
    T0=$(date +%s)
    if [ "$3" = "-" ]; then
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
        --output_subdir "$2" --write_only_valid --overwrite \
        --parallel 4 --attempts 2 --llm_timeout 1800 >> "$LOG" 2>&1
    else
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
        --output_subdir "$2" --task_feedback_file "$3" --write_only_valid --overwrite \
        --parallel 4 --attempts 2 --llm_timeout 1800 >> "$LOG" 2>&1
    fi
    AFTER=$(left "$1" "$2"); AFTER=$(echo $AFTER | wc -w)
    DT=$(( $(date +%s) - T0 ))
    if [ "$AFTER" -ge "$BEFORE" ] && [ "$DT" -lt 300 ]; then
      # 아무것도 못 만들고 금방 끝났다 = 한도에 막힌 것. 재시도로 세지 않는다.
      echo "  no progress in ${DT}s -- throttled, waiting 20m (tries ${tries})" >> "$LOG"
      sleep 1200
    else
      tries=$((tries + 1))
      sleep 10
    fi
  done
  echo "  $(basename $1) -> $2 : gave up after $tries working rounds" >> "$LOG"
}

echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue v3 start (group41)" >> "$LOG"

stage /hdd_data/yunho/cctmp/solo41.json group1 /hdd_data/yunho/cctmp/group41_feedback.json || exit 0
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue v3 end" >> "$LOG"
