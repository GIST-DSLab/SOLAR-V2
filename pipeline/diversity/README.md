# 경로 다양성 측정

한 과제를 푸는 유효한 ARCLE 궤적이 몇 가지인지 센다. 배포 게이트가 아니다 --
게이트는 `pipeline/preflight.py`와 `pipeline/choose_maker.py`다.

실험 기록과 지시문, 결과 전부: `docs/ROUTE_DIVERSITY.md`,
자료는 `/hdd_data/yunho/route-diversity/`.

| 스크립트 | 무엇을 하나 |
|---|---|
| `diversity_all.py` | `maker/` 아래 모든 변형을 같은 인스턴스에 돌려 경로 가짓수를 센다. 과제마다 저장하고 다시 돌리면 이어서 간다 |
| `eval84.py` | 특정 갈래가 **새** 경로를 냈는지. 과제 목록과 `NEW` 튜플을 바꿔 쓴다 |
| `mech.py` | 순열·분할·병합이 ARCLE 재생을 통과하는지 |
| `mechroutes.py` | 그 변환 결과가 기존 경로와 **다른지**. 유효한 것과 새로운 것은 다르다 |
| `golfdiff.py` | 임의 마스크 비율과 op 어휘 변화 |
| `depcheck.py` | 정답을 흐트러뜨렸을 때 경로가 바뀌는지, 남의 답을 그려내는지 |

경로의 신원은 (op 이름, 실제로 고른 칸의 집합)의 나열이다. 선택의 표기는
보지 않는다 -- bbox와 같은 칸을 가리키는 셀 목록은 같은 선택이다.
