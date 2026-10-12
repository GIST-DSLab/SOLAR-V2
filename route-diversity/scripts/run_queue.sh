#!/bin/bash
# 생성 큐. 한 번에 하나씩, 끊겨도 이어서, 주간 90%에서 멈춘다.
#   1) 재표집  -- 같은 프롬프트를 다시 뽑는다. 참조도 피드백도 없다.
#      20개 표본에서 18개가 서로 다른 경로 3가지를 냈던 것이고, 네 계열 중
#      준비 비용이 0이다. 경로가 한 가지뿐인 과제부터 간다.
#   2) BARC    -- 남은 것.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/queue.log

# 사용량을 한 번 읽어 세션%와 주간%를 같이 꺼낸다 -- 두 번 부르면 그 호출도 비용이다.
usage(){ timeout 120 claude -p /usage 2>/dev/null; }
pct(){ sed -n "s/.*$1:[^0-9]*\([0-9]\+\)%.*/\1/p" | head -1; }

# 세션 창이 찰 때까지 밀어붙이지 않는다. 창은 몇 시간마다 풀리므로, 문턱에
# 닿으면 멈추는 게 아니라 기다렸다가 이어서 간다 -- 남은 과제는 그대로 있다.
wait_session(){
  for i in $(seq 1 180); do
    U=$(usage); S=$(echo "$U" | pct "session"); W=$(echo "$U" | pct "week (all models)")
    if [ -n "$W" ] && [ "$W" -ge 90 ]; then
      echo "  weekly ${W}% >= 90, stopping queue" >> "$LOG"; exit 0
    fi
    if [ -z "$S" ] || [ "$S" -lt 85 ]; then
      SESSION_PCT="${S:-?}"; WEEK_PCT="${W:-?}"; return 0
    fi
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] session ${S}% >= 85, waiting for the window" >> "$LOG"
    sleep 600
  done
  return 1
}

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

stage(){  # $1=목록 $2=출력폴더 $3=피드백파일(없으면 "-")
  for attempt in 1 2 3 4 5 6; do
    TODO=$(left "$1" "$2")
    [ -z "$TODO" ] && { echo "  $(basename $1) -> $2 : done" >> "$LOG"; return 0; }
    wait_session || { echo "  gave up waiting on the session window" >> "$LOG"; exit 0; }
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] $(basename $1) -> $2, attempt $attempt, session ${SESSION_PCT}%, weekly ${WEEK_PCT}%, $(echo $TODO | wc -w) left" >> "$LOG"
    if [ "$3" = "-" ]; then
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
        --output_subdir "$2" --write_only_valid --overwrite \
        --parallel 4 --attempts 2 --llm_timeout 1800 >> "$LOG" 2>&1
    else
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
        --output_subdir "$2" --task_feedback_file "$3" --write_only_valid --overwrite \
        --parallel 4 --attempts 2 --llm_timeout 1800 >> "$LOG" 2>&1
    fi
    sleep 10
  done
}

# 지금 돌고 있는 생성(BARC 파일럿)이 끝날 때까지 기다린다.
while pgrep -f "[g]en_rearc_makers_llm.py" > /dev/null; do sleep 120; done
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue start" >> "$LOG"

stage /hdd_data/yunho/cctmp/resample_a.json resample1 -
stage /hdd_data/yunho/cctmp/resample_b.json resample1 -
stage /hdd_data/yunho/cctmp/barc_pilot.json barc1 /hdd_data/yunho/cctmp/barc_feedback.json
stage /hdd_data/yunho/cctmp/barc_rest.json  barc1 /hdd_data/yunho/cctmp/barc_feedback.json
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] queue end" >> "$LOG"
