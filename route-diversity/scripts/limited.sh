#!/bin/bash
# 공유 기계에서 내 작업만 담아두는 껍데기.
#
# MemoryMax 하나만 쓴다. MemoryHigh(상한 전에 회수 압력)를 같이 걸어봤더니
# 죽이는 대신 무한정 멈춰세웠다 -- 3분이 지나도 끝나지 않았다. 그게 바로
# 사고 때의 모습이라(OOM은 안 뜨는데 느려져서 접속이 안 됨) 쓰면 안 된다.
# 상한에 닿으면 깔끔히 죽는 편이 낫다: 내 작업만 사라지고 기계는 멀쩡하다.
#
# CPUWeight와 IOWeight를 낮춰, 남과 경쟁이 붙으면 내가 먼저 양보한다.
# nice/ionice는 그 위의 보험이다.
#
#   limited.sh <메모리상한> <CPU%> <명령...>
MEM="${1:?mem}"; CPU="${2:?cpu}"; shift 2
exec systemd-run --user --scope -q \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  -p CPUQuota="${CPU}%" -p CPUWeight=20 -p IOWeight=20 -p TasksMax=256 \
  nice -n 19 ionice -c3 "$@"
