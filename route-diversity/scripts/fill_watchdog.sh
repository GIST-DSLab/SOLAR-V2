#!/bin/bash
# 재부팅·강제종료에도 이어지도록, 러너가 살아있는지 보고 없으면 다시 띄운다.
# cron에서 5분마다 부른다. 러너 자체가 이미 "이미 만든 과제는 건너뛰기"라
# 중복 생성은 없다.
LOCK=/hdd_data/yunho/cctmp/fill.lock
LOG=/hdd_data/yunho/cctmp/fill.log
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0          # 이미 돌고 있으면 아무것도 안 한다

python3 - <<'PY' > /hdd_data/yunho/cctmp/fill_left.txt
import json
from pathlib import Path
left = 0
for a in ["group3", "golf292"]:
    want = set(json.load(open(f"/hdd_data/yunho/cctmp/{a}_feedback.json")))
    d = Path(f"/hdd_data/yunho/solar-traj/maker/{a}")
    have = {p.name for p in d.iterdir() if p.is_dir() and (p/"grid_maker.py").exists()} \
           if d.is_dir() else set()
    left += len(want - have)
print(left)
PY
LEFT=$(cat /hdd_data/yunho/cctmp/fill_left.txt)
[ "${LEFT:-0}" -le 0 ] && exit 0

if ! pgrep -f "cctmp/run_fill.sh" > /dev/null; then
  echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M')] watchdog: restarting, $LEFT left" >> "$LOG"
  setsid nohup /hdd_data/yunho/cctmp/limited.sh 16G 400 /hdd_data/yunho/cctmp/run_fill.sh > /dev/null 2>&1 < /dev/null &
fi
