# 경로 다양성 — 프롬프트·스크립트·측정 결과

한 ARC 과제를 ARCLE 궤적 여러 가지로 푸는 실험의 공개 묶음이다.
무엇을 왜 했는지는 `docs/ROUTE_DIVERSITY.md`에 있고, 여기에는 **그대로 다시
돌릴 수 있는 재료**만 둔다.

모든 경로를 다 만들지는 못했다. 그래서 만드는 방법을 공개한다.

## 어디부터 보면 되나

1. **`templates/PROMPT_ARCHITECTURE.md`** — 먼저 읽을 것.
   프롬프트 뼈대가 어떻게 생겼고, 갈래마다 **어느 자리에 무엇을 넣으면**
   무엇이 달라지는지. 갈래끼리 프롬프트가 많이 달라 보이지만 바뀌는 곳은
   한 군데뿐이다.
2. `templates/codegolf_reference.txt`, `templates/barc_reference.txt` —
   그 한 군데에 들어가는 문자열의 템플릿. 기존 코드와 참조 코드 자리가
   자리표시자로 비어 있다.
3. `prompts/INSTRUCTIONS.md` — 갈래별로 실제로 넣은 지시문 전문.
4. `prompts/*_feedback.json` — 과제별 완성본. 바로 재현에 쓸 수 있다.

## 구성

| 폴더 | 무엇 |
|---|---|
| `templates/` | 프롬프트 뼈대 설명과 슬롯 템플릿 |
| `prompts/` | 갈래별 지시문 전문(`INSTRUCTIONS.md`)과 과제별 JSON |
| `scripts/` | 참조 수집·검증, 프롬프트 조립, 생성 러너, 측정 |
| `results/` | 측정 결과 JSON |
| `logs/` | 실행 로그 |

## 돌리는 법

갈래마다 명령이 같고 두 인자만 다르다.

```bash
python3 pipeline/gen_rearc_makers_llm.py \
  --tasks <task id들> \
  --output_subdir      <arm 이름> \
  --task_feedback_file route-diversity/prompts/<arm>_feedback.json \
  --write_only_valid --overwrite \
  --parallel 3 --attempts 2 --llm_timeout 1800
```

`--parallel`을 올리지 말 것. 60으로 올렸다가 공유 기계를 멈춘 적이 있다.
`scripts/limited.sh`가 cgroup으로 상한을 걸고, `scripts/isolate.py`가 측정에서
과제마다 fork해 폭주하는 maker를 한 과제로 가둔다.

## 참조 재료의 출처

code-golf 저장소는 여기에 싣지 않는다. 각자 받아야 한다.
`scripts/collect_golf.py`가 ARC 원본 train+test 쌍 전부에 돌려서 재현되는
풀이만 걸러내고, 과제 번호와 task id는 원본 쌍의 해시로 맞춘다.

| arm | 저장소 |
|---|---|
| `cg_cgi` | `Seek64/NeurIPS-Code-Golf-2025` |
| `cg_cristianoc` | `cristianoc/arc-code-golf-solutions` |
| `cg_aggregate` | `arc-code-golf/solutions` |
| `cg_oxjam` | `xsot/google-code-golf-2025` |
| `cg_camjam` | `intgrah/golf` |
| `cg_itertools` | `Natanaelel/GoogleCodeGolf2025` |
| `cg_lv1dev` | `lv1-dev/google-code-golf-2025` |
| `cg_sakana` | `SakanaAI/google-code-golf-2025` |
| `cg_cctofu` | `cctofu/golf-championship` |
| `barc1` | `xu3kev/BARC` 의 `seeds/` |
