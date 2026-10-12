# 프롬프트 뼈대와 갈래가 바뀌는 자리

갈래마다 프롬프트 전문이 다 달라 보이지만, **바뀌는 곳은 한 군데뿐**이다.
나머지는 RE-ARC 시절 400개 maker를 쓴 그대로다.

조립하는 코드는 `pipeline/gen_rearc_makers_llm.py`의 `_user_prompt()`.

---

## 전체 구조

```
┌─ SYSTEM (약 26,000자) ──────────────────────────────── 고정, 모든 갈래 공통
│   ARCLE 환경 설명, 작성할 세 함수의 계약
│     sample_colors()  — 에피소드 단위 색 역할
│     generate()       — 인스턴스 생성기
│     derive_operations(I, O) -> (ops, sels)
│   op 목록과 선택(selection) 규약, bbox가 오프셋이라는 것
│   "Op 수는 비용이 아니다" — 금지하는 것은 무의미한 행동이지 긴 경로가 아님
│   겹쳐 그리기는 정당하다 (바탕을 깔고 그 위에 그리기)
│   코드만 내놓고 산문은 쓰지 말 것
└──────────────────────────────────────────────────────────────────────

┌─ USER ────────────────────────────────────────────────────────────────
│ 1. Task ID: {task_id}                                        고정
│
│ 2. {objective}                                               고정
│      "NO WASTED ACTIONS" (기본) 또는 "DSL-FAITHFUL"
│
│ 3. {feedback_block}        ◀◀◀  갈래가 바뀌는 유일한 자리  ▶▶▶
│      고정 머리말:  "Task-specific reviewer feedback (acceptance requirement):"
│      ────────────────────────────────────────────────
│       여기에 갈래별 내용이 들어간다 (아래 표)
│      ────────────────────────────────────────────────
│      고정 꼬리말:  "Infer the rule first, then group actions into
│                     human-meaningful objects, lines, or regions.
│                     Do not raster-scan and reconstruct the known output…"
│      ※ 이 블록은 --task_feedback_file 을 안 주면 통째로 사라진다.
│         그 상태가 "기준"(arc-agi-1)이다.
│
│ 4. {verifier_block}                          --include_verifier (기본 ON)
│      RE-ARC verifier 전문 + "이것은 실행 계획이 아니다" 경고
│      ⚠ 기본값이 켜짐이라 모든 갈래가 이미 Hodel의 읽기를 보고 있다.
│        따라서 code-golf 코드를 넣는 것은 첫 참조가 아니라 "두 번째 참조"다.
│
│ 5. Original RE-ARC generator: ```python {src} ```             고정
│      generators.py의 그 과제 생성기 전문
│
│ 6. Concrete I/O examples: {pairs}                             고정
│
│ 7. {object_hint}                          --include_object_hints (기본 OFF)
│
│ 8. {solver_block}                             --include_solver (기본 OFF)
│      arc-dsl solver — "규칙이 무엇인지"만, 템플릿으로 쓰지 말 것
│
│ 9. "Write the three functions: sample_colors(), generate(),
│     derive_operations()."                                     고정
└──────────────────────────────────────────────────────────────────────
```

---

## 3번 자리에 무엇을 넣으면 무엇이 달라지는가

모든 갈래가 공통으로 다음 두 줄을 **맨 끝에** 붙인다. 이것이 다양성을
품질과 바꾸지 않게 막는 장치다.

> Every selection must be something you can name from the input before you
> know the answer — a rectangle, a connected object, all cells of one colour,
> a row, a column, the cells a copy lands on.
>
> And the operations must be derivable from the input grid and the worked
> examples alone: handed a different answer grid, this function must return
> the same operations.

| 갈래 | 3번 자리에 넣는 것 | 넣는 말 |
|---|---|---|
| 기준 | (비움) | 블록 자체가 없음 |
| 짧게 `golf*` | 현재 maker 코드 | "같은 답에 더 적은 동작으로. 작게 쪼개지 말고 큰 획으로" |
| 다르게 묶기 `group*` | 현재 maker 코드 | "규칙은 맞다. **한 단위로 보는 것**을 바꿔라. 물체 단위면 줄 단위로, 전체면 쪼개서" |
| 인스턴스마다 `var*` | 현재 maker 코드 + 한 인스턴스의 op 나열 | "인스턴스마다 경로가 갈리게. 단 **난수가 아니라 입력에서 유도**" |
| 다른 어휘 `alt*` | 현재 maker 코드 + op 패턴 요약 | "칠하기 말고 복제·이동·뒤집기로 같은 일을" |
| **BARC 참조** `barc1` | 현재 maker 코드 **+ BARC seed 코드** | "남이 다른 어휘로 읽은 것이다. **그 읽기를 따르라**" |
| **code-golf 참조** `cg_*` | 현재 maker 코드 **+ 그 팀의 제출 코드** | "짧은 건 거기서 바이트가 목적이었기 때문. **다른 읽기를 따른 긴 경로가 답**" |

### 참조 갈래와 지시 갈래의 차이는 한 문단

- **지시 갈래**는 "무엇을 바꿔라"를 말로 지정한다 (단위, 어휘, 인스턴스 간 변화)
- **참조 갈래**는 바꿀 방향을 말하지 않는다. **남의 읽기를 보여주고 그걸 따르라**고만 한다
- 그래서 참조 갈래는 **팀을 바꾸면 계속 다른 걸 받을 수 있고**, 지시 갈래는 같은
  지시로 두 번째 경로를 받기 어렵다

---

## 재현에 필요한 플래그

실제로 돌린 명령은 모든 갈래가 동일하고 두 인자만 다르다.

```bash
python3 pipeline/gen_rearc_makers_llm.py \
  --tasks <task id들> \
  --output_subdir   <arm 이름>          ◀ 다름
  --task_feedback_file <arm>_feedback.json   ◀ 다름
  --write_only_valid --overwrite \
  --parallel 3 --attempts 2 --llm_timeout 1800
```

- `--write_only_valid` — 롤아웃이 정답에 닿는 maker만 쓴다
- `--parallel 3` — 메모리 상한 안에서 도는 수. 60으로 올렸다가 기계를 멈춘 적 있음
- 켜지 않은 것: `--include_solver`, `--include_object_hints`, `--builtin_feedback`
- **켜져 있는 것: `--include_verifier` (기본값)**

`<arm>_feedback.json`의 모양:

```json
{ "007bbfb7": { "feedback": "<3번 자리에 들어갈 문자열 전문>" }, ... }
```

과제마다 현재 maker의 `derive_operations` 전문이 박혀 있으므로 파일이 크다
(`group3_feedback.json` 1.1MB, 254개 과제).
이 파일을 만드는 코드는 `scripts/build_*.py`.
