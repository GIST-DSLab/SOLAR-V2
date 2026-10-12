#!/bin/bash
# code golf 제출본 참조, 400개 × 팀 6개 = 2330 생성.
#
# 팀별로 400개를 먼저 끝내고 다음 팀으로 간다. 과제별로 여섯 팀을 돌면
# 예산이 끊겼을 때 "일부 과제만 여섯 참조"가 되는데, 팀별로 돌면 "전 과제가
# 참조 한둘"이 된다. 뒤쪽이 낫다 -- 첫 참조의 수율이 0.75로 가장 높고,
# 두 번째 참조부터는 겹치기 시작한다.
#
# 세션 90%, 주간 90%에서 멈추고, 창이 풀리면 이어서 간다. 한도에 막혀 바로
# 떨어진 라운드는 재시도로 세지 않는다.
export CLAUDE_CODE_TMPDIR=/hdd_data/yunho/cctmp TMPDIR=/hdd_data/yunho/cctmp
cd /hdd_data/yunho/solar-traj || exit 1
LOG=/hdd_data/yunho/cctmp/cg_all.log
# 묶음 하나가 세션의 5%p쯤 먹으므로, 85%에서 멈추면 한 묶음을 더 돌아도
# 90%를 넘지 않는다. 라운드가 400개짜리면 게이트가 시작 전에만 보기 때문에
# 중간에 99%까지 갔던 적이 있다.
SESSION_MAX=80
WEEK_MAX=98
CHUNK=10
TEAMS="cgi oxjam camjam aggregate itertools lv1dev"

usage(){ timeout 120 claude -p /usage 2>/dev/null; }
sess(){ echo "$1" | sed -n 's/.*session:[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }
week(){ echo "$1" | sed -n 's/.*week (all models):[^0-9]*\([0-9]\+\)%.*/\1/p' | head -1; }

left(){ python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
t = sys.argv[1]
want = set(json.load(open(f"/hdd_data/yunho/cctmp/cg_{t}.json")))
d = Path(f"/hdd_data/yunho/solar-traj/maker/cg_{t}")
have = {p.name for p in d.iterdir() if p.is_dir() and (p/"grid_maker.py").exists()} \
       if d.is_dir() else set()
print(" ".join(sorted(want - have)))
PY
}

wait_budget(){
  while :; do
    # 감시자가 위험하다고 판단하면 여기서 선다. 사람이 강제 종료하는 대신
    # 내가 먼저 비킨다.
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

echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] cg_all start" >> "$LOG"
for team in $TEAMS; do
  tries=0
  while [ "$tries" -lt 4 ]; do
    ALL=$(left "$team")
    [ -z "$ALL" ] && { echo "  $team: done" >> "$LOG"; break; }
    BUD=$(wait_budget)
    TODO=$(echo $ALL | tr ' ' '\n' | head -n "$CHUNK" | tr '\n' ' ')
    BEFORE=$(echo $ALL | wc -w)
    echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] cg_$team  ${BUD}  ${BEFORE} left, chunk $(echo $TODO | wc -w)" >> "$LOG"
    T0=$(date +%s)
    /hdd_data/yunho/cctmp/limited.sh 12G 300 \
      python3 pipeline/gen_rearc_makers_llm.py --tasks $TODO \
      --output_subdir "cg_$team" \
      --task_feedback_file "/hdd_data/yunho/cctmp/cg_${team}_feedback.json" \
      --write_only_valid --overwrite \
      --parallel 3 --attempts 2 --llm_timeout 1800 \
      >> "/hdd_data/yunho/cctmp/cg_$team.log" 2>&1
    AFTER=$(left "$team"); AFTER=$(echo $AFTER | wc -w)
    tries=0   # 묶음이 전진하면 재시도 횟수를 되돌린다
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
echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] cg_all end" >> "$LOG"
