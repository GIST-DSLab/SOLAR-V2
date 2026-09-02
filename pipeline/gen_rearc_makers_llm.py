#!/usr/bin/env python3
"""
LLM-based grid_maker.py generator for all 400 RE-ARC tasks.

Calls `claude -p` (Claude Code CLI) as a subprocess to generate a
task-specific `derive_operations(I, O)` function per task.
No separate API key needed — uses current Claude Code session.

Usage:
    python gen_rearc_makers_llm.py                      # all tasks
    python gen_rearc_makers_llm.py --tasks 0d3d703e 007bbfb7
    python gen_rearc_makers_llm.py --num_examples 5 --overwrite
    python gen_rearc_makers_llm.py --dry_run            # print prompts only
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SOLAR_ROOT    = Path(__file__).resolve().parents[1]
REARC_ROOT    = SOLAR_ROOT / "re-arc"
LOG_DIR       = SOLAR_ROOT / "conv_logs"
_ARCLE_REF_PATH = SOLAR_ROOT / "docs" / "arcle_reference.md"
_ARCLE_REF_TEMPLATE = _ARCLE_REF_PATH.read_text(encoding="utf-8")

LOG_DIR.mkdir(exist_ok=True)

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tasks",        nargs="*", default=None)
parser.add_argument("--num_examples", type=int, default=6,
                    help="I/O examples shown to LLM (default 6)")
parser.add_argument("--overwrite",    action="store_true",
                    help="Regenerate even if grid_maker.py exists")
parser.add_argument("--dry_run",      action="store_true",
                    help="Print prompts without calling claude")
parser.add_argument("--max_grid_dim", nargs=2, type=int, default=[30, 30],
                    metavar=("H", "W"),
                    help="Max grid size (default 30 30)")
parser.add_argument("--min_grid_dim", nargs=2, type=int, default=[1, 1],
                    metavar=("H", "W"),
                    help="Min grid size to accept (default 1 1 - no lower bound)")
parser.add_argument("--parallel",     type=int, default=4,
                    help="Concurrent claude calls (default 4)")
parser.add_argument("--llm_timeout",  type=int, default=2700,
                    help="seconds one LLM call may take before it is killed "
                         "(default 2700). With --attempts 3 a task can spend "
                         "three of these, so lower it when a round has to fit "
                         "in a known window")
parser.add_argument("--save_log",     action="store_true",
                    help="Save full conversation JSON to conv_logs/<task_id>.json")
parser.add_argument("--rand_seed",    type=int, default=42)
parser.add_argument(
    "--trajectory_mode", choices=["efficient", "dsl_faithful"], default="efficient",
    help=("Trajectory objective: efficient=correct/minimal ARCLE actions (default); "
          "dsl_faithful=also preserve solver-level Move/Rotate/Flip families"),
)
parser.add_argument(
    "--include_verifier", action=argparse.BooleanOptionalAction, default=True,
    help="Include the declarative verifier in the user prompt (default: true)",
)
parser.add_argument(
    "--include_solver", action=argparse.BooleanOptionalAction, default=False,
    help="Include the procedural re-arc-llm solver in the user prompt (default: false)",
)
parser.add_argument(
    "--include_object_hints", action=argparse.BooleanOptionalAction, default=False,
    help="Include task-id-derived object-op hints in the prompt (default: false)",
)
parser.add_argument(
    "--builtin_feedback", action=argparse.BooleanOptionalAction, default=False,
    help="Apply the hand-written per-task feedback table baked into this file "
         "(5 tasks; default: false). It was a one-off v6→v7 remediation and does not "
         "generalise. v6/v7 were generated with it ON — pass this to reproduce them.",
)
parser.add_argument(
    "--rule_first", action=argparse.BooleanOptionalAction, default=False,
    help="Make the model state the rule in plain English before the code "
         "block (default: false). Unlike --include_solver this needs no per-task "
         "reference and so extends to unseen tasks.",
)
parser.add_argument(
    "--enforce_solver_ops", action=argparse.BooleanOptionalAction, default=None,
    help=("Require solver-detected Move/Rotate/Flip ops in validation. By default this is "
          "enabled only in --trajectory_mode dsl_faithful"),
)
parser.add_argument(
    "--llm_backend", choices=["claude", "codex"], default="claude",
    help="LLM CLI used to generate task code (default: claude)",
)
parser.add_argument("--output_subdir", type=str, default="arc-from-rearc",
                    help="Output subdirectory under maker/ (default: arc-from-rearc)")
parser.add_argument("--diff_lb",      type=float, default=None,
                    help="RE-ARC generator difficulty lower bound (default: random 0.2~0.5)")
parser.add_argument("--diff_ub",      type=float, default=None,
                    help="RE-ARC generator difficulty upper bound (default: random 0.5~0.8)")
parser.add_argument("--attempts",     type=int, default=3,
                    help="LLM generation/repair attempts per task (default 3)")
parser.add_argument("--task_feedback_file", type=str, default=None,
                    help="JSON mapping task IDs to human feedback and operation constraints")
parser.add_argument("--write_only_valid", action="store_true",
                    help="Do not write a candidate unless validation succeeds")
args = parser.parse_args()

if args.attempts < 1:
    parser.error("--attempts must be at least 1")

TASK_FEEDBACK: dict[str, dict[str, Any]] = {}
if args.task_feedback_file:
    feedback_path = Path(args.task_feedback_file).resolve()
    try:
        loaded_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read --task_feedback_file {feedback_path}: {exc}")
    if not isinstance(loaded_feedback, dict):
        parser.error("--task_feedback_file must contain a JSON object keyed by task ID")
    for feedback_tid, entry in loaded_feedback.items():
        if isinstance(entry, str):
            entry = {"feedback": entry}
        if not isinstance(entry, dict):
            parser.error(f"feedback entry for {feedback_tid} must be a string or object")
        TASK_FEEDBACK[str(feedback_tid)] = entry

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)

MAX_H, MAX_W = args.max_grid_dim
MIN_H, MIN_W = args.min_grid_dim
ARCLE_REF = _ARCLE_REF_TEMPLATE.replace("{MAX_H}", str(MAX_H)).replace("{MAX_W}", str(MAX_W))
OUTPUT_DIR = SOLAR_ROOT / "maker" / args.output_subdir
CLAUDE_SESSION_LIMIT_REACHED = threading.Event()

# ── Parse RE-ARC generators ────────────────────────────────────────────────────
sys.path.insert(0, str(REARC_ROOT))

generators_src = (REARC_ROOT / "generators.py").read_text(encoding="utf-8")
tree           = ast.parse(generators_src)
src_lines      = generators_src.splitlines()

task_ids: list[str]         = []
func_bodies: dict[str, str] = {}

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith("generate_"):
        tid = node.name[len("generate_"):]
        task_ids.append(tid)
        func_bodies[tid] = "\n".join(src_lines[node.lineno - 1 : node.end_lineno])

print(f"Found {len(task_ids)} RE-ARC tasks.")

# ── Parse RE-ARC verifiers ─────────────────────────────────────────────────────
verifiers_src  = (REARC_ROOT / "verifiers.py").read_text(encoding="utf-8")
vtree          = ast.parse(verifiers_src)
vsrc_lines     = verifiers_src.splitlines()
verify_bodies: dict[str, str] = {}

for node in ast.walk(vtree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith("verify_"):
        tid = node.name[len("verify_"):]
        verify_bodies[tid] = "\n".join(vsrc_lines[node.lineno - 1 : node.end_lineno])

print(f"Found {len(verify_bodies)} RE-ARC verifiers.")

# verifier 기준 (grid 생성 방식 기술 — 과검출 있음, hint 용도로만 사용)
_v_shift = {t for t,b in verify_bodies.items() if "shift(" in b or "move(" in b}
_v_rotate = {t for t,b in verify_bodies.items()
             if any(k in b for k in ("rot90(", "rot180(", "rot270("))}
_v_flip = {t for t,b in verify_bodies.items()
           if any(k in b for k in ("hmirror(", "vmirror(", "dmirror(", "cmirror("))}

# ── DSL → ARCLE op mapping (included in prompt for object-level tasks) ────────
DSL_ARCLE_MAP = """\
## RE-ARC DSL → ARCLE op mapping
Translate solver DSL calls DIRECTLY to these ARCLE ops (select object bbox each time):

| DSL call                  | ARCLE op(s)                              | Notes                                      |
|---------------------------|------------------------------------------|--------------------------------------------|
| shift(obj,(dr,dc))        | MoveU=20/MoveD=21/MoveR=22/MoveL=23     | |dr|+|dc| steps; update selection each step|
| move(grid,obj,(dr,dc))    | same as shift                            |                                            |
| rot90(grid)               | Rotate90 = 24                            | select object bbox                         |
| rot180(grid)              | Rotate90 twice (24, 24)                  |                                            |
| rot270(grid)              | Rotate270 = 25                           |                                            |
| hmirror(piece)            | FlipH = 26   (top↔bottom)               | select object bbox                         |
| vmirror(piece)            | FlipV = 27   (left↔right)               | select object bbox                         |
| dmirror / cmirror         | no direct ARCLE op — use Color ops       | repaint cells individually                 |
| fill(grid,v,region)       | FloodFillV = 10+v, seed=[r,c,0,0]       | one seed cell inside connected region      |
| fill_background(bg,v,...) | FloodFillV = 10+v on background region   | seed = any background cell in region       |
| paint_onto_grid(grid,obj) | Color ops per cell — by object, not raster| paint cells grouped by object/region      |
| recolor(v,obj)            | Color = v, grouped by object bbox        |                                            |

Priority rule: Move/Rotate/Flip > FloodFill > Color.
FloodFill for connected regions; Color only for isolated cells or non-contiguous pixels.
When Color is unavoidable, paint by semantic reference point — NOT top-left raster scan.
"""

# ── Parse re-arc-llm solvers ───────────────────────────────────────────────────
_SOLVER_PATH = SOLAR_ROOT / "re-arc-llm" / "solvers.py"
solver_bodies: dict[str, str] = {}
if _SOLVER_PATH.exists():
    _solver_src   = _SOLVER_PATH.read_text(encoding="utf-8")
    _solver_tree  = ast.parse(_solver_src)
    _solver_lines = _solver_src.splitlines()
    for _node in ast.walk(_solver_tree):
        if isinstance(_node, ast.FunctionDef) and _node.name.startswith("solve_"):
            _tid = _node.name[len("solve_"):]
            solver_bodies[_tid] = "\n".join(
                _solver_lines[_node.lineno - 1 : _node.end_lineno]
            )
    print(f"Found {len(solver_bodies)} re-arc-llm solvers.")

# solver 기준 분류 (실제 I→O 변환 의미 — rejection gate 기준)
_s_shift = {t for t,b in solver_bodies.items()
            if any(k in b for k in ("move_object(", "move_until_touching(", "shift_by_vector("))}
_s_rotate = {t for t,b in solver_bodies.items()
             if any(k in b for k in ("rot90(", "rot180(", "rot270("))}
_s_flip = {t for t,b in solver_bodies.items()
           if any(k in b for k in ("horizontal_mirror(", "vertical_mirror(",
                                   "diagonal_mirror(", "counterdiagonal_mirror("))}

# hint 대상 = verifier OR solver (넓게)
shift_move_tasks: set[str]  = _v_shift | _s_shift
rotate_tasks: set[str]      = _v_rotate | _s_rotate
flip_tasks: set[str]        = _v_flip | _s_flip
object_level_tasks: set[str] = shift_move_tasks | rotate_tasks | flip_tasks

# rejection gate = solver 기준만 (과검출 방지)
_gate_shift  = _s_shift
_gate_rotate = _s_rotate
_gate_flip   = _s_flip

for _mod in ["utils", "dsl", "generators"]:
    if _mod in sys.modules:
        del sys.modules[_mod]
from generators import *  # noqa: F401, F403

# ── I/O generation helpers ────────────────────────────────────────────────────

def _gen_io(task_id: str, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    generate_fn = globals().get(f"generate_{task_id}")
    if generate_fn is None:
        return []
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 12:
        tries += 1
        try:
            if args.diff_lb is not None and args.diff_ub is not None:
                _lb, _ub = args.diff_lb, args.diff_ub
            else:
                _lb = random.uniform(0.2, 0.5)
                _ub = random.uniform(0.5, 0.8)
            r = generate_fn(_lb, _ub)
            I = np.array(r["input"],  dtype=np.uint8)
            O = np.array(r["output"], dtype=np.uint8)
            # filter by size bounds
            if I.shape[0] > MAX_H or I.shape[1] > MAX_W: continue
            if O.shape[0] > MAX_H or O.shape[1] > MAX_W: continue
            if I.shape[0] < MIN_H or I.shape[1] < MIN_W: continue
            if O.shape[0] < MIN_H or O.shape[1] < MIN_W: continue
            pairs.append((I, O))
        except Exception:
            continue
    return pairs


def _grid_str(g: np.ndarray) -> str:
    h, w = g.shape
    rows = "[" + "],\n  [".join(", ".join(str(int(v)) for v in row) for row in g) + "]"
    return f"# shape=({h},{w})\n[\n  {rows}\n]"


def _format_examples(pairs: list[tuple]) -> str:
    return "\n\n".join(
        f"Example {i+1}:\nI =\n{_grid_str(I)}\nO =\n{_grid_str(O)}"
        for i, (I, O) in enumerate(pairs)
    )

# ── Prompt builders ────────────────────────────────────────────────────────────

# ── The prompt ────────────────────────────────────────────────────────────────
# This is the prompt the 400 published makers were written with. Earlier drafts
# are not kept here. It states each rule (bgc repair, recolor-vs-move, Copy
# source choice, structural patterns, the padded-repaint and in-place-op-sweep
# waste patterns) at the point in the prompt where it is actually decided,
# rather than appending them as a trailing "additional patterns" section.
SYSTEM_PROMPT = f"""\
You are an expert at writing ARC (Abstraction and Reasoning Corpus) grid transformers \
using the ARCLE reinforcement-learning environment.

You will receive a RE-ARC task: its generator source code plus concrete input→output examples.

Your output: exactly THREE Python functions in a single ```python``` block:

---

### 1. sample_colors(num_examples=None) -> dict
Returns a dict of kwargs used by generate() — color roles and, when the task has discrete
structural cases, a pre-planned per-instance sequence. Called ONCE per episode; the SAME
episode-level values are shared across every generate() call in that episode (examples
AND test). Do NOT mutate or pop() anything here across calls — the caller (not generate())
is responsible for selecting each instance's entry by index and passing it through a plain
kwargs dict. See the instance-plan pattern below for the exact contract.

Rules:
- Include EVERY color variable that the original generator samples randomly (bgc, fgc, col1, col2, etc.)
- At minimum, background color must be fixed if the task has one
- If a color is hardcoded (not randomly sampled), do NOT include it
- Exception: if the task rule depends only on the *presence/pattern* of objects (not their color),
  foreground colors do NOT need to be fixed — only bgc.
- Accept `num_examples=None` even if unused — the caller passes the real value when available.

```python
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {{"bgc": bgc, "fgc": fgc}}
```

### 2. generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict
A modified version of the RE-ARC generator that:
- Replaces hardcoded `30` upper bounds for h/w with `max_h`/`max_w`
- Uses color_kwargs instead of sampling colors internally
- Returns {{"input": grid, "output": grid}}

**Discrete structural variants** (classification label, connected/disconnected, direction,
shape mode, count mode, color relationship, etc.): there is no way for generate() to know
whether THIS call is building an example or the test instance. Do not special-case "the
test call". Have sample_colors() build the full sequence up front (length =
num_examples + 1), return it under `"instance_plan"` as a list of kwargs dicts, and let the
caller merge `instance_plan[j]` into generate()'s kwargs **by index**:
```python
VARIANTS = [
    {{"connected": True,  "direction": "horizontal"}},
    {{"connected": False, "direction": "vertical"}},
]

def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = (num_examples if num_examples else 3)
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test case was shown
    return {{"bgc": bgc, "instance_plan": plan}}

def generate(diff_lb, diff_ub, max_h, max_w, bgc,
             connected=None, direction=None) -> dict:
    if connected is None or direction is None:
        variant = random.choice(VARIANTS)
        connected = variant["connected"]
        direction = variant["direction"]
    ...
```
Don't pop() from instance_plan inside generate() — a single instance can retry generate()
several times (oversized grid, generator exception) before succeeding, and each retry must
see the SAME variant. The caller reads `instance_plan[j]` and merges that dict into the
generate() kwargs. When examples are numerous enough, cover every discrete variant at least
once; choose the test dict from the example dicts so the episode is learnable. Use a single
dict entry with multiple fields when dimensions interact — do not invent separate parallel
plans that can become desynchronized.

### 3. derive_operations(I, O, examples=None) -> tuple[list[int], list[list[int]]]
Returns (ops, sels) — the ARCLE operation sequence transforming I into O.
- `examples` is the episode's demonstrations: a list of (input, output) pairs, the
  same ones a solver is shown before being asked about I. Take the third parameter
  and use them wherever the rule has a convention that I alone does not fix — which
  colour the rule fills with, which colour marks a thing — because that is where a
  solver reads it from. Reading it out of O instead is reading the answer.
- A two-argument `derive_operations(I, O)` still works and is right when the rule
  needs nothing beyond I. Take `examples` only when you would otherwise have to look
  at O for it.
- Last op MUST be Submit = 34
- len(ops) == len(sels)
- EVERY selection is a MASK: the exact set of cells the op must act on. Emit it with
  `sel_of(cells)` — add `from maker.sel_helpers import sel_of` — where `cells` are the
  (r, c) indices you intend, e.g. `sel_of(toindices(obj))` for an object, or the exact
  cells you mean to recolour.
  A bbox `[r, c, h, w]` is accepted ONLY when your intended cells are exactly that full
  rectangle — e.g. reflecting or rotating a whole region, background included. Then say
  so in a comment.
  NEVER let a bbox stand in for a non-rectangular object. Move/Rotate/Flip relocate the
  WHOLE selection and clear it from the background, so a bounding rectangle drags or
  erases everything else inside it; and Color/FloodFill would repaint cells that needed
  no change. Selecting the object's true cells is what makes the op act on the object.
- EVERY SELECTION MUST BE CONSTRUCTIBLE FROM WHAT IS VISIBLE AT THAT STEP — from I, the
  grid as it stands after the ops so far, and the rule you measured from the examples.
  A policy replaying your trajectory sees exactly that and never sees O. So a selection
  shaped like "the cells that are background IN THE FINISHED GRID", or one that carves out
  holes where objects will land later, is FORBIDDEN even when every op is otherwise clean:
  the answer is then hidden in the selection instead of the ops, which is worse because it
  is harder to see. Concretely, never build a full predicted output and then diff against it
  to decide what to select. Select the region you mean by its own description — "the rows the
  Resize just added", "this object's cells", "the whole left half" — and paint it whole.
- WHEN THE RULE IS A TRANSLATION, MOVE THE OBJECT. If a component's shape and colour
  reappear in O shifted by a constant (dr, dc), you MUST express that with Move ops on
  `sel_of(the object's cells)`, NOT by painting the vacated cells to background and
  repainting the object at its destination. This holds for ANY displacement, whatever
  determines it — a measured offset, sliding until it touches something, settling against
  an edge, mirroring a neighbour's position. Do not treat this as a rule about one family
  of tasks; the test is only "same object, new position".
  Erase-and-repaint reaches the same grid while hiding the very thing the task is about:
  nothing in the trajectory shows that the object MOVED. **The Move chain being longer is
  not a reason to prefer erase-and-repaint** — op count is not a cost (see the trajectory
  objective). A displacement of 12 cells written as 12 unit Moves is better data than
  2 Color ops.
  Paint directly only when there is genuinely no translation to express: the object
  changes shape or colour, or it is created/destroyed rather than relocated.
- HOW ARCLE MOVE ACTUALLY WORKS — GRAB ONCE, THEN MOVE WITH AN EMPTY SELECTION.
  A Move carries a selection. The FIRST Move of a slide carries the object's cells:
  ARCLE grabs that object and snapshots the background (the whole grid with just the
  grabbed cells zeroed), then shifts the object one cell. Every FURTHER Move of the same
  slide must carry an EMPTY selection (`sel_of([])` -> {{"cells": []}}, no cells). With an
  empty selection ARCLE KEEPS THE SAME OBJECT GRABBED and re-pastes it over that one
  snapshot, so every cell the object glides OVER is automatically restored to its original
  colour. You do NOT repaint the path.
      cur = list(src_cells)
      ops.append(MOVE_OP); sels.append(sel_of(cur))       # first step GRABS the object
      cur = [(r + dr_step, c + dc_step) for r, c in cur]
      for _ in range(steps - 1):
          ops.append(MOVE_OP); sels.append(sel_of([]))    # empty -> continue same object
          cur = [(r + dr_step, c + dc_step) for r, c in cur]
  NEVER re-select the object at its current position on each step. Re-selecting mid-slide
  re-grabs and re-snapshots the (already holed) grid every step, which bakes a 0-trail
  along the WHOLE path and then forces a bogus full-path repair. One grab, then empties.
- THE ONLY 0s LEFT ARE THE OBJECT'S ORIGINAL FOOTPRINT. Because the snapshot zeroes the
  grabbed cells, the cells where the object STARTED read 0 after the slide (the path does
  NOT — ARCLE already kept it alive). If the background `bgc` is non-zero, repair JUST the
  original footprint the object no longer covers, with ONE Color(bgc):
      hole = sorted(set(src_cells) - set(cur))     # cur = final position after the slide
      if bgc != 0 and hole:
          ops.append(bgc); sels.append(sel_of(hole))
  Do NOT repaint the path the object crossed: ARCLE restored it, and painting over it is
  wasteful and WRONG when the path crossed non-background cells. A non-zero background is
  never a reason to abandon Move for erase-and-repaint.
- To move a SEPARATE object next, start a NEW grab: the next Move carries that object's
  cells (non-empty), which wipes the previous grab and snapshots the background afresh.
- ARCLE CLIPS ANYTHING A MOVE PUSHES OFF THE GRID. When a Move step carries part of the
  object past a grid edge, the off-grid cells are silently dropped and the on-grid part
  keeps moving — nothing errors. So sliding an object partly (or eventually wholly) off
  the edge is a legal, normal Move. A grid boundary is NEVER a reason to switch to
  erase-and-repaint: keep issuing the Move and let ARCLE clip.
- STOP A MOVE CHAIN THE INSTANT IT STOPS CHANGING THE VISIBLE GRID. A Move — or any op —
  that leaves the grid identical to the step before is a redundant no-op and is REJECTED
  by the validator. This bites exactly when clipping starts emptying the object: once the
  object has reached its final on-grid position (or has fully clipped away), the next Move
  changes nothing. Advance only while each step visibly alters the grid, and issue no
  further step after the last visible change. Compute the number of steps from the
  destination you measured; do not over-run it.
- Must work for ANY valid (I, O) pair this task can produce — measure everything
  dynamically from I/O, never hardcode a value you only saw in one example.

**Complete ARCLE op table (O2ARCv2Env-v0):**
```
 0–9  : Color0–Color9      — paint ALL selected cells to color N
10–19 : FloodFill0–9       — BFS-fill connected same-color region from 1 seed cell (op = 10 + color)
20    : MoveU              — move selected object up 1 cell
21    : MoveD              — move selected object down 1 cell
22    : MoveR              — move selected object right 1 cell
23    : MoveL              — move selected object left 1 cell
24    : Rotate90           — rotate 90° CCW = np.rot90(k=1)  ⚠️ SQUARE selection ONLY
25    : Rotate270          — rotate 90° CW  = np.rot90(k=3)  ⚠️ SQUARE selection ONLY
26    : FlipH              — flip left↔right = np.fliplr = vmirror
27    : FlipV              — flip up↔down   = np.flipud = hmirror
28    : CopyI              — copy INPUT region (nonzero cells only) to clipboard
29    : CopyO              — copy CURRENT WORKING GRID region (nonzero only) to clipboard
30    : Paste              — paste clipboard at selection's top-left (transparent: 0s don't overwrite)
31    : CopyInput          — reset working grid to input (mid-sequence only, NEVER at position 0)
32    : ResetGrid          — ⛔ DO NOT USE — clears grid, forces blind pixel-by-pixel repaint of O
33    : ResizeGrid/CropGrid— resize canvas to selection bbox (transparent copy)
34    : Submit             — submit current grid as answer (ALWAYS last op)
```

⚠️ **Op semantics — NOT intuitive, memorize once:**
- op24 (Rotate**90**) = **CCW**. op25 (Rotate**270**) = **CW**. (Opposite of what the names suggest.)
- op26 (Flip**H**) = **left↔right** (fliplr/vmirror). op27 (Flip**V**) = **up↔down** (flipud/hmirror).
- FloodFill's underlying selection MASK must contain exactly 1 True cell or it's a silent
  NOOP. In the `[r, c, h, w]` bbox format you actually write, that always means `h=0, w=0`
  — always use `[r, c, 0, 0]` for FloodFill. (Don't sum the bbox list itself — `sum([r,c,0,0])`
  is unrelated to the mask and is usually not 1.)
- Paste/CropGrid/CopyI/CopyO all treat 0 as "nothing here": zero cells never overwrite, never get copied.

⚠️ **Rotate on non-square selections is broken** (op24/op25 compute the wrong position when
h≠w). If `w=h` is explicit in the generator, rotate directly. If h,w vary independently, pad
to a square first, rotate, then crop — see "Non-square rotate" in the ARCLE Reference for the
exact crop coordinates (they differ for CW vs CCW).

**Op selection priority — choose the highest-level op that fits:**

1. **Move** (20–23): object translates — i.e. the SAME content visibly reappears at a
   DIFFERENT position in O. Select bbox, update selection after each 1-cell step.
   ⚠️ Move is NOT for erasing/removing an object (it disappears in O, period — that's
   Color/FloodFill to bgc, not a Move) and NOT a way to compute a bounding box or offset
   for your own bookkeeping. If the object doesn't reappear elsewhere in O, don't reach
   for Move at all — pick the next-lower op that actually matches what happens.

   bgc is NOT passed in. Infer it from task structure: what does the generator treat as
   background (a fixed color it paints the canvas with before placing objects)? That's
   usually reliable, and where it is not, the demonstrations settle it — every example
   input is drawn on the same background as I. `most_common(1)` on
   I's colors (`Counter(I.flatten().tolist())`) is a REASONABLE FALLBACK ONLY — it breaks
   whenever background isn't the majority color (foreground-heavy grids, stripes/checkerboards,
   near-50/50 color splits). Don't reach for it as the default; use it only when you can't
   determine bgc from the generator, and double-check it against a couple of examples.

   Rule out recolor first — cheap and unambiguous: if every non-bgc position in I is ALSO
   non-bgc at the SAME position in O, it's a pure recolor (FloodFill/Color), definitely
   not Move:
   ```python
   changed = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != O[r, c]]
   maybe_moved = any((I[r, c] == bgc) != (O[r, c] == bgc) for r, c in changed)
   # False -> definitely pure recolor, stop here (never Move)
   # True  -> NOT enough on its own to conclude Move — an add/delete/resize/extend
   #          produces the same signal. Confirm below before committing to Move.
   ```
   Only commit to Move once you've found an actual match: a connected component in I whose
   shape AND color pattern reappears in O shifted by one constant `(dr, dc)` (e.g. compare
   each component's bounding box to O's, or just check `I`'s component shifted by a
   candidate `(dr, dc)` equals the corresponding O region exactly). If no such match exists,
   something was added, removed, or resized — that's not a pure move; paint the affected
   cells directly instead of forcing Move ops onto it.

   After Move ops, ARCLE leaves the vacated part of the ORIGINAL source bbox at 0. If the
   shift is small relative to the object (source and destination bboxes overlap), part of
   that source bbox is now correctly holding the just-moved content — don't touch it, or
   you'll repaint already-correct cells (the same waste this list forbids elsewhere). Only
   repair the truly vacated area: source minus destination.
   ```python
   # obj_r, obj_c, obj_h, obj_w = object's position BEFORE moving (source bbox)
   # dest_r, dest_c = object's position AFTER moving (same obj_h, obj_w)
   src = {{(r, c) for r in range(obj_r, obj_r + obj_h) for c in range(obj_c, obj_c + obj_w)}}
   dst = {{(r, c) for r in range(dest_r, dest_r + obj_h) for c in range(dest_c, dest_c + obj_w)}}
   for (r, c) in src - dst:            # cells only ever in the source, never in the destination
       if O[r, c] != 0:                # ARCLE left this vacated cell at 0; O wants something else
           ops.append(int(O[r, c])); sels.append([r, c, 0, 0])
   ```
2. **Rotate** (24/25): square objects only. op25=CW(rot90), op24=CCW(rot270).
3. **Flip** (26/27): any shape. op26=fliplr(vmirror), op27=flipud(hmirror). rot180 = op26+op27.
4. **FloodFill** (10–19): connected same-color region recolor. Select 1 seed cell [r,c,0,0].
5. **Color** (0–9): isolated cells or sparse non-connected pixels.

**⛔ Forbidden and wasteful patterns — an op you could delete without changing the final
submitted grid is a bug, not a stylistic choice. Check every op you emit against this list:**

- **ResetGrid (op32) or pixel-by-pixel O-copy**: clearing the grid, or iterating every O
  cell and painting it individually, means you don't understand I→O. Identify WHAT
  transformation I undergoes — geometric op, recolor, or targeted paint — and derive that.
- **CopyInput (op31) as the first op**: the grid already IS the input at episode start.
- **Redundant cycles**: no sub-sequence may return the grid to a state it was already in.
  MoveD×3 then MoveU×3 = net zero. Rotate CW then CCW = identity. CopyInput after edits =
  undoes everything before it. This means every op must be NECESSARY for the final result
  — not that every op must monotonically get closer to O along the way. Temporarily moving
  further from O is fine and often required (e.g. clear a region to bgc before pasting into
  it, or expand the canvas before cropping it down) — that's not a cycle as long as the grid
  never returns to an exact state it already visited.
- **An op used only for Python-side bookkeeping**: every op's GRID EFFECT must be part of
  the real transformation. Never emit a Move/Rotate/Flip/Resize call just to help YOUR
  derive_operations code compute a bounding box or offset — do that arithmetic in plain
  Python. Self-check: if you deleted this op, would the final submitted grid change?
- **Painting/filling a region wider than needed "to be safe"**: when repainting an area
  (margin around a moved/erased object, canvas padding after Resize), check each candidate
  cell against its CURRENT value — skip a cell only when it ALREADY HOLDS the colour you are
  about to paint, so the op would do nothing there. This includes bgc-fill-after-Resize: skip
  it when bgc==0 (the zero-padding already IS bgc) — see "Canvas resize" in the ARCLE Reference.
  Judge against the grid AS IT IS NOW, never against the finished output. Do NOT drop a cell
  from the selection because a LATER op will paint over it — laying a base and drawing on top
  is correct, and subtracting the future leaks the answer into the selection (see below).
- **In-place Flip/Rotate sweeping up untouched content**: they transform the WHOLE selected
  region, including any pre-existing non-bgc cells a prior Paste never touched — those get
  mirrored/rotated too and end up wrong. Prefer clearing the region to bgc before pasting
  into it, so a later in-place op has nothing pre-existing to catch. If you can't clear
  first, only correct the specific cells actually inside that op's selection bbox afterward
  — not a blind full-grid I-vs-O scan (same waste as the padded-rectangle problem above).

**When using Color ops — paint by object/region, NOT by raster order:**
Separate object discovery from operation emission. A row/column scan is fine for finding
components, but NEVER append operations directly from a full-grid diff scan. First identify
the anchor/reference object, partition changed cells into semantic target objects or
connected regions, and order those targets by the task relation (adjacency, propagation
direction, containment, or distance from the anchor). Start at the cell adjacent to the
reference and proceed outward when the rule grows a pattern. Finish one target object or
connected region before starting another; do not alternate between objects merely because
their cells interleave in row-major order.

**Copy/Paste — which clipboard source:**
- **CopyI (op28)** copies from the ORIGINAL INPUT. Use it whenever duplicating something
  that still looks like it did in I — most duplication/tiling/mirroring tasks.
- **CopyO (op29)** copies from the CURRENT WORKING GRID (after your own edits so far).
  Rarely correct — only when you specifically need content YOU just produced.
- Paste target only needs the top-left corner: `[dst_r, dst_c, 0, 0]`, regardless of the
  copied region's size.
- ⚠️ **0 is always "nothing" to Copy/Paste/CropGrid, even if bgc != 0 and 0 is being used
  as an ordinary foreground color in this task.** If any cell you're copying is 0 but is
  semantically part of what you're duplicating, it will NOT be copied and Paste will NOT
  write it — the destination keeps whatever was already there. When 0 can appear as real
  object content (not just background), Copy/Paste is unsafe for that region; use a
  geometric op (Move/Rotate/Flip, which relocate the whole selection including 0s) or
  explicit Color0 ops instead.
  For a non-rectangular object use sel_of(its cells) so Move/Rotate/Flip relocate
  only the object, not the whole bbox.

---

**Structural patterns worth recognizing — each applies ONLY when the task actually matches
its description below. These are shapes to recognize, not steps to force onto every task.
If I→O doesn't match one of these descriptions, ignore it and reason from I/O directly:**

- **Duplicate-with-mirror at a new position**: `CopyI` the source bbox → `Paste` at the
  destination → `FlipH`/`FlipV` in place on the destination bbox to mirror it. For 4-quadrant
  symmetric tasks (1 object in I, mirrored copies in O), find the object color by connected-
  component size (the real object is one big BFS component; a center marker is many small
  isolated ones), then for each new quadrant compare its center to the source's to pick
  FlipH vs FlipV.
- **Tiling to a d×d pattern repeated d² times**: compute `d` from O's shape
  (`d = round(sqrt(O.shape[0]))`), not from I's non-bgc extent — the latter is unreliable.
  Locate the d×d tile in I, `CopyI` it, `Paste` at each `(i*d, j*d)` tile origin that should
  be nonzero.
- **Moved object whose source is "contaminated"** (extra connecting lines/pixels of the same
  color alongside the real object body, so I's bbox for that color includes junk): find the
  clean destination bbox in O first, then scan I for a block of the same shape that's
  entirely that color — that's the true source. Move it, then paint the leftover
  "contamination" cells (object-colored in I but not in O) to bgc.

**Read the generator before hardcoding anything.** It shows you exactly how each structural
element is built — translate that into a *dynamic* measurement in derive_operations, never
a literal constant you saw in one example. E.g. `sqd = randint(1, min(w, loci-1))` means the
size varies — detect it from I (`area == h*w`, bbox measurements), don't hardcode `sq=2`.

**Counterfactual rule check — required before coding:** list every generator parameter that
can change structure rather than only color or position. Check that your detection works at
its minimum, maximum, and alternative discrete values. Reject explanations that depend on
incidental facts such as first in raster order, largest component, border contact, or a
particular overlap unless the generator guarantees that fact. Flood-filling or repainting
the observed diff is not a valid shortcut when another legal shape, connectivity,
direction, count, or overlap would break it.

**Plan operation order from information dependencies:** preserve every source object until
its final Copy, Move, measurement, or detection use. Perform CropGrid/Resize or destructive
erasure only after removed information is no longer needed. When safe, keep a causal op and
its cleanup adjacent (for example, repair a moved object's truly vacated region before an
unrelated object), but batch cleanup when one connected region can be handled by one
FloodFill or Color selection. Never destroy information and reconstruct it later unless the
transformation genuinely requires that sequence.

---

**Episode learnability check — required before writing parse():**

Ask: "Given ONLY my N examples, could a solver always determine the correct output for ANY
new test instance this task produces?" If NO for any plausible test instance, fix it in
sample_colors()/generate() — parse() itself is fixed infrastructure, not something you write:

- **Discrete variants**: every structural case that can appear in test MUST be represented
  in the examples when enough example slots exist. Don't rely on random.choice() per
  instance to cover classification labels, connectivity, direction, shape/count modes, or
  interacting cases. Use the `instance_plan` pattern from section 2; the caller merges each
  planned kwargs dict into generate() by instance index.
- **Color-dependent rules**: if the rule depends on a specific color role (not just bgc),
  fix that role in sample_colors() — a test instance with an unseen color combo breaks
  learnability.
- **Structural variants**: if N distinct object shapes/pattern types need different rules,
  ensure every variant that could appear in test also appears in the examples.

Only act when a test instance could plausibly be ambiguous or unseen relative to examples.

---

ARCLE Reference:
{ARCLE_REF}

Output ONLY the three functions in a single ```python``` block. No prose.
"""

# The prompt ends by telling the model to emit code and nothing else, and it obeys: 238 of 240
# v7 responses had zero characters before the code fence, so the concept was never
# stated anywhere — not by the model, and (with the solver hidden) not by the prompt.
# The 5 hand-written built_in_feedback entries worked precisely because they filled
# that gap. This contract makes the model fill it itself, which — unlike the solver,
# which exists only for these 400 tasks — also works on a task we have never seen.
# _extract_code() reads the first ```python block, so a "RULE:" preamble is safe.
# Swapped in for the no-prose line when --rule_first is set (see process_task).
_NO_PROSE_LINE = (
    "Output ONLY the three functions in a single ```python``` block. No prose."
)
_RULE_FIRST_CONTRACT = """\
Your output has TWO parts, in this order.

PART 1 — "RULE:" then 3-8 plain-English sentences stating what this transformation
does: what the objects are, what decides the result, and what varies from one
instance of this task to the next. Write it as a person who has only I and the
worked examples in front of them. Name no ARCLE ops and no DSL calls here — this
part is the concept, not the trace.

PART 2 — exactly THREE Python functions in a single ```python``` block, implementing
the rule you just stated. Where the code and the rule disagree, the rule is what you
meant: fix the code, not the rule."""


def _user_prompt(task_id: str, src: str, pairs: list[tuple]) -> str:
    """Build the user prompt without treating declarative DSL calls as an ARCLE trace."""
    n_show = min(args.num_examples, len(pairs))

    verifier_block = ""
    if args.include_verifier and task_id in verify_bodies:
        verifier_block = (
            "RE-ARC verifier (declarative final I→O specification):\n"
            "⚠️ This is NOT an execution plan. Its shift/rotate/mirror calls may normalize "
            "coordinates, construct temporary objects, or describe output geometry. Do not "
            "translate those calls directly into ARCLE operations; choose operations from the "
            "visible transformation between I and O.\n"
            f"```python\n{verify_bodies[task_id]}\n```\n\n"
        )

    include_solver = args.include_solver or args.trajectory_mode == "dsl_faithful"
    solver_block = ""
    if include_solver and task_id in solver_bodies:
        # The solver says WHAT the rule is; it is never a template for HOW.
        # Earlier drafts said "translate each DSL op to an ARCLE op", which
        # instructed the mimicry we now forbid. Hiding the solver answered that, but left
        # "derive from the visible I→O transformation" as the only remaining
        # instruction — and with no concept to derive from, that collapses into
        # painting the diff. Both failures inscribe information the policy will
        # never have: one from the spec, one from the answer. Hence this third
        # framing: show the concept, refuse the template.
        solver_block = (
            "re-arc-llm solver — the CONCEPT of this task, written as a human's solution\n"
            "to the ORIGINAL ARC task. Read it to learn WHAT the rule is. That is the only\n"
            "thing it is for.\n"
            "\n"
            "It is NOT a template for your op sequence. One DSL call does NOT mean one ARCLE\n"
            "op. The call order reflects DSL constraints, not the concept. Its sizes and\n"
            "coordinates are hardcoded to the original task and will not fit the augmented\n"
            "inputs you must handle. Do not import or call its DSL functions.\n"
            "\n"
            "Use it to understand the rule. Then write derive_operations as a person who had\n"
            "only I and the worked examples — and who solves it the way that person would.\n"
            f"```python\n{solver_bodies[task_id]}\n```\n\n"
        )

    object_hint = ""
    if args.include_object_hints and task_id in object_level_tasks:
        candidates = []
        if task_id in shift_move_tasks:
            candidates.append("Move (20–23)")
        if task_id in rotate_tasks:
            candidates.append("Rotate (24–25)")
        if task_id in flip_tasks:
            candidates.append("Flip (26–27)")
        object_hint = (
            "Possible geometric families mentioned by reference code: "
            + ", ".join(candidates)
            + ". This is an advisory search hint only. Use a family only after I/O confirms "
              "its ARCLE-visible effect; otherwise omit it.\n\n"
        )

    if args.trajectory_mode == "dsl_faithful":
        objective = (
            "Trajectory objective: DSL-FAITHFUL. Preserve solver-level Move/Rotate/Flip "
            "families where validation requires them, while still removing redundant actions."
        )
    else:
        objective = (
            "Trajectory objective: NO WASTED ACTIONS. **Op count is not a cost.** Do not "
            "shorten a trajectory by collapsing a transformation into fewer, less meaningful "
            "ops. What is forbidden is the MEANINGLESS action: an op that returns the grid to "
            "a state it already held, an op with no visible effect at the moment it is issued, "
            "an op that only helps your Python side compute something, or an op added merely "
            "to resemble verifier/solver DSL. "
            "LAYERING IS LEGITIMATE. Laying a base — filling a region with the background, "
            "then drawing on top of it — is exactly how a human works, and the fact that a "
            "later op covers part of the base does NOT make the base op wasted. Do NOT chase "
            "'nothing gets painted twice' by computing the finished grid and subtracting the "
            "cells that will be covered later: that hides the answer inside the SELECTION. "
            "Fill the whole region you mean to fill, then draw over it. "
            "Correctness and visible I→O causality are what matter. "
            "A LONG chain of unit Moves that each carry an object closer to its destination is "
            "CORRECT and PREFERRED over erasing the object and repainting it elsewhere — even "
            "though the erase-and-repaint is shorter. Every op must advance the rule visibly."
        )

    # Hand-written per-task feedback for 5 tasks — a one-off remediation from the v6→v7
    # round, not part of the loop. It works (v7's 23581191 copies the wording almost
    # verbatim), but it does not generalise: task 401 gets nothing. The solver block now
    # carries the concept for all 400 instead, so this stays off unless asked for.
    # v6 and v7 were generated with it ON; pass --builtin_feedback to reproduce them.
    built_in_feedback = {
        "1e0a9b12": (
            "Task-specific quality requirement: do not imitate a verifier rotation. Match the "
            "same shape/color object between I and O and, when it has one constant displacement, "
            "use ARCLE Move steps with the current bbox and repair only the truly vacated cells. "
            "Do not replace a clear translation with Rotate or a generic erase-and-redraw trace."
        ),
        "1190e5a7": (
            "Task-specific quality requirement: derive the output height and width from the "
            "number of horizontal and vertical full-length separator lines in I (plus one). "
            "Line colors may differ; color equality is not the rule. Do not use O.shape as a "
            "substitute for measuring this structure from I."
        ),
        "137eaa0f": (
            "Task-specific quality requirement: the separated input objects encode the output "
            "one object/color at a time around their shared reference-dot position. Preserve "
            "source information, complete one semantic object before the next, and perform the "
            "final Crop/Resize only after all source-dependent work. Do not Resize first and "
            "then repaint O as a generic color grid."
        ),
        "234bbc79": (
            "Task-specific quality requirement: preserve the separated colored snake segments "
            "until their order/shape has been used. Prefer object-level CopyI/Paste or another "
            "source-preserving construction when ARCLE semantics make it safe. Crop/Resize to "
            "the final canvas only after source-dependent operations; do not Resize first and "
            "redraw the answer from O."
        ),
        "23581191": (
            "Task-specific quality requirement: express the real rule: each source point emits "
            "its horizontal and vertical full-grid lines, and cells where the two color families "
            "intersect become color 2 while original points retain their colors. Organize ops by "
            "source-point/line family and intersections, not as a generic flood-fill or arbitrary "
            "top-left diff repaint."
        ),
    }.get(task_id, "") if args.builtin_feedback else ""
    external_feedback = str(TASK_FEEDBACK.get(task_id, {}).get("feedback", "")).strip()
    task_feedback = "\n".join(
        part for part in (built_in_feedback, external_feedback) if part
    )
    feedback_block = (
        "Task-specific reviewer feedback (acceptance requirement):\n"
        f"{task_feedback}\n"
        "Infer the rule first, then group actions into human-meaningful objects, lines, or "
        "regions. Do not raster-scan and reconstruct the known output unless the task truly "
        "has no higher-level structure.\n\n"
        if task_feedback else ""
    )

    return (
        f"Task ID: {task_id}\n\n"
        f"{objective}\n\n"
        f"{feedback_block}"
        f"{verifier_block}"
        f"Original RE-ARC generator:\n"
        f"```python\n{src}\n```\n\n"
        f"Concrete I/O examples:\n"
        f"{_format_examples(pairs[:n_show])}\n"
        f"{object_hint}"
        f"{solver_block}"
        "Write the three functions: sample_colors(), generate(), derive_operations()."
    )


def call_claude(system: str, user: str, timeout: int = 2700,
                log_path: str | None = None) -> str | None:
    """
    Call `claude -p` with system + user prompt via temp files.
    Returns stdout text or None. Saves full conversation to log_path if given.
    """
    import json as _json

    if CLAUDE_SESSION_LIMIT_REACHED.is_set():
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as sf:
        sf.write(system)
        sys_file = sf.name

    # The user prompt goes through a file too, not through a pipe. `claude -p`
    # waits 3 s for stdin and gives up if nothing has arrived ("no stdin data
    # received in 3s, proceeding without it"), which then exits 1 with "Input
    # must be provided". With --parallel this is a coin flip on a loaded box:
    # four writers, tens of KB each, and a prompt that misses the window is a
    # task lost for the round. A regular file is readable the moment it opens.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as uf:
        uf.write(user)
        user_file = uf.name

    cmd = [
        shutil.which("claude") or "claude", "-p",
        "--system-prompt-file", sys_file,
        "--output-format", "text",
        "--dangerously-skip-permissions",
    ]

    stdout = stderr = None
    returncode = None
    try:
        # Not subprocess.run(timeout=...): on a timeout it kills the child and
        # then calls communicate() again to drain the pipes, and if anything the
        # CLI spawned still holds stdout that second call blocks forever — with
        # the timeout already spent, nothing wakes it. One run sat wedged for
        # three hours that way. Own the process group and kill all of it.
        with open(user_file, "r", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                cmd,
                stdin=fh,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    stdout = stderr = ""
                print(f"    claude timed out after {timeout}s")
                return None
        returncode = proc.returncode
        if returncode != 0:
            detail = (stderr or stdout or "<no output>").strip()
            if "session limit" in detail.lower():
                first_limit = not CLAUDE_SESSION_LIMIT_REACHED.is_set()
                CLAUDE_SESSION_LIMIT_REACHED.set()
                if first_limit:
                    print("    Claude session limit reached; skipping remaining calls until next cycle.")
            print(f"    claude failed (exit {returncode}): {detail[:500]}")
            return None
        return stdout
    except FileNotFoundError:
        print("    ERROR: `claude` CLI not found in PATH")
        return None
    finally:
        os.unlink(sys_file)
        os.unlink(user_file)
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "system_prompt": system,
                    "user_prompt":   user,
                    "response":      stdout,
                    "stderr":        stderr,
                    "returncode":    returncode,
                }, f, indent=2, ensure_ascii=False)


def call_codex(system: str, user: str, timeout: int = 2700,
               log_path: str | None = None) -> str | None:
    """Call Codex non-interactively and return only its final response."""
    import json as _json

    out_fd, out_file = tempfile.mkstemp(suffix=".txt")
    os.close(out_fd)
    prompt = (
        "Treat the following SYSTEM INSTRUCTIONS as the highest-priority task "
        "instructions for this isolated code-generation request. Do not edit files or "
        "run commands; return only the requested Python code block.\n\n"
        "===== SYSTEM INSTRUCTIONS =====\n"
        + system
        + "\n\n===== TASK INPUT =====\n"
        + user
    )
    cmd = [
        shutil.which("codex") or "codex", "exec",
        "--ephemeral",
        "--sandbox", "read-only",
        "--skip-git-repo-check",
        "-C", str(SOLAR_ROOT.parent),
        "--output-last-message", out_file,
        "-",
    ]

    stdout = stderr = response = None
    returncode = None
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        if Path(out_file).exists():
            response = Path(out_file).read_text(encoding="utf-8").strip()
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or response or "<no output>").strip()
            print(f"    codex failed (exit {result.returncode}): {detail[:500]}")
            return None
        return response or result.stdout
    except subprocess.TimeoutExpired:
        print("    codex timed out")
        return None
    except FileNotFoundError:
        print("    ERROR: `codex` CLI not found in PATH")
        return None
    finally:
        try:
            os.unlink(out_file)
        except FileNotFoundError:
            pass
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "backend":       "codex",
                    "system_prompt": system,
                    "user_prompt":   user,
                    "response":      response,
                    "stdout":        stdout,
                    "stderr":        stderr,
                    "returncode":    returncode,
                }, f, indent=2, ensure_ascii=False)


def call_llm(system: str, user: str, timeout: int = 2700,
             log_path: str | None = None) -> str | None:
    if args.llm_backend == "codex":
        return call_codex(system, user, timeout=timeout, log_path=log_path)
    return call_claude(system, user, timeout=timeout, log_path=log_path)

# ── Code extraction & validation ───────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "def sample_colors" in code and "def generate" in code and "def derive_operations" in code:
            return code
    # fallback: grab everything from first def
    for marker in ("def sample_colors", "def derive_operations"):
        if marker in text:
            return text[text.index(marker):].strip()
    return None


def _validate(
    code: str,
    pairs: list[tuple],
    required_ops: dict[str, list[int]] | None = None,
    forbidden_ops: dict[str, list[int]] | None = None,
) -> tuple[bool, str]:
    import random as _random
    import importlib.util as _ilu
    import sys as _sys
    if str(SOLAR_ROOT) not in _sys.path:
        _sys.path.insert(0, str(SOLAR_ROOT))

    def _load(name: str):
        p = REARC_ROOT / f"{name}.py"
        spec = _ilu.spec_from_file_location(name, p)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ns: dict[str, Any] = {"np": np, "random": _random}
    for _mname in ("utils", "dsl"):
        _mod = _load(_mname)
        ns.update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")})

    try:
        exec(compile(code, "<llm>", "exec"), ns)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"exec error: {e}"

    for fname in ("sample_colors", "generate", "derive_operations"):
        if ns.get(fname) is None:
            return False, f"{fname} not defined"

    # test generate + derive_operations on a couple of pairs
    _seen_ops: set[int] = set()
    for i, (I, O) in enumerate(pairs[:3]):
        try:
            ops, sels = ns["derive_operations"](I, O)
        except Exception as e:
            return False, f"example {i+1} derive_operations error: {e}"
        for label, family in (forbidden_ops or {}).items():
            used = [int(op) for op in ops if int(op) in family]
            if used:
                return False, (
                    f"example {i+1}: human review forbids unnecessary {label}; "
                    f"trajectory used {used}"
                )
        if not ops or ops[-1] != 34:
            return False, f"example {i+1}: last op must be 34"
        if len(ops) != len(sels):
            return False, f"example {i+1}: len mismatch ops/sels"
        def _sel_ok(op, s):
            if isinstance(s, dict):
                cells = s.get("cells")
                if cells == []:
                    # empty selection = continue the already-grabbed object; ARCLE only
                    # honours this for Move ops (20-23). Reject it on any other op.
                    return int(op) in (20, 21, 22, 23)
                return bool(cells) and all(
                    isinstance(c, (list, tuple)) and len(c) == 2 for c in cells)
            return len(s) == 4
        if any(not _sel_ok(op, s) for op, s in zip(ops, sels)):
            return False, f"example {i+1}: selection must be [r,c,h,w] or {{'cells': [...]}}"
        _seen_ops.update(ops)

    # Required-op gate, checked against the ACTUAL ops derive_operations returned
    # above — not a static string/regex search over the source text. A regex
    # (even a word-boundary one) can't tell `ops.append(20)` apart from an
    # unrelated `MAX_STEPS = 20` or a comment mentioning the op id, and it can't
    # recognize valid variable-based emission at all without over- or under-matching.
    # Real returned op ids are unambiguous either way.
    if required_ops:
        missing = [label for label, ids in required_ops.items()
                   if not any(o in _seen_ops for o in ids)]
        if missing:
            return False, f"missing required ops: {', '.join(missing)}"

    # Test sample_colors + generate, including the generic per-instance plan.
    # category_plan remains supported for already-generated responses;
    # new makers should use instance_plan (a list of kwargs dicts).
    try:
        import inspect as _inspect
        n_ex = args.num_examples
        if "num_examples" in _inspect.signature(ns["sample_colors"]).parameters:
            colors = ns["sample_colors"](num_examples=n_ex)
        else:
            colors = ns["sample_colors"]()
        if not isinstance(colors, dict):
            return False, "sample_colors must return dict"

        category_plan = colors.pop("category_plan", None)
        instance_plan = colors.pop("instance_plan", None)
        if category_plan is not None and instance_plan is not None:
            return False, "sample_colors must return only one of category_plan/instance_plan"

        if instance_plan is not None:
            if not isinstance(instance_plan, (list, tuple)):
                return False, "instance_plan must be a list/tuple of kwargs dicts"
            if len(instance_plan) != n_ex + 1:
                return False, (
                    f"instance_plan length {len(instance_plan)} != num_examples+1 "
                    f"({n_ex + 1})"
                )
            if any(not isinstance(entry, dict) for entry in instance_plan):
                return False, "every instance_plan entry must be a kwargs dict"
            if instance_plan[-1] not in instance_plan[:-1]:
                return False, "instance_plan: test variant (last entry) never appears among examples"
            for j in range(n_ex + 1):
                call_kwargs = dict(colors)
                call_kwargs.update(instance_plan[j])
                result = ns["generate"](0.2, 0.8, MAX_H, MAX_W, **call_kwargs)
                if "input" not in result or "output" not in result:
                    return False, f"generate must return input/output dict (instance {j})"
        elif category_plan is not None:
            if len(category_plan) != n_ex + 1:
                return False, (
                    f"category_plan length {len(category_plan)} != num_examples+1 "
                    f"({n_ex + 1}) — sample_colors(num_examples=...) must size the plan "
                    "to exactly one entry per instance (examples + test)."
                )
            if category_plan[-1] not in category_plan[:-1]:
                return False, "category_plan: test category (last entry) never appears among the examples"
            for j in range(n_ex + 1):
                call_kwargs = dict(colors)
                call_kwargs["category"] = category_plan[j]
                result = ns["generate"](0.2, 0.8, MAX_H, MAX_W, **call_kwargs)
                if "input" not in result or "output" not in result:
                    return False, f"generate must return dict with 'input'/'output' (instance {j})"
        else:
            result = ns["generate"](0.2, 0.8, MAX_H, MAX_W, **colors)
            if "input" not in result or "output" not in result:
                return False, "generate must return dict with 'input'/'output'"
    except Exception as e:
        return False, f"generate/sample_colors error: {e}"

    # ARCLE trajectory simulation: check ops actually produce O from I.
    # Only an ImportError (ARCLE genuinely not installed in this environment) skips
    # the check silently. Any other exception during setup/simulation is a real bug
    # in the generated code and must fail validation, not be swallowed — a bare
    # `except Exception: pass` here previously treated shape mismatches and env
    # errors as if validation had passed.
    try:
        import gymnasium as gym
        import arcle
        import importlib.util as _ilu2
        # NOT a plain `import utils` — sys.modules["utils"] is already bound to
        # re-arc/utils.py (from the module-level `from generators import *` at
        # the top of this file), so a plain import would silently return THAT
        # module instead of this one, and sel_bbox_to_mask below would
        # AttributeError on every single call. Load it by explicit path under a
        # distinct module name so it can't collide. It is this package's own
        # utils.py, beside this file — an absolute path, since the check has to
        # work whatever directory the script was started from.
        _solar_utils_path = Path(__file__).resolve().parent / "utils.py"
        _spec = _ilu2.spec_from_file_location("_solar_utils_for_validate", _solar_utils_path)
        _solar_utils = _ilu2.module_from_spec(_spec)
        _spec.loader.exec_module(_solar_utils)
    except ImportError:
        return True, "ok"  # ARCLE not installed here — skip simulation check

    class _SingleLoader:
        """ARCLE's env expects `data_loader.pick(data_index)` to return a
        (train_in, train_out, test_in, test_out, desc) 5-tuple — see
        arcle/loaders/loader.py. A raw `[(I, O)]` list has no `.pick()` at
        all, so `_env.reset()` crashes with `'list' object has no attribute
        'pick'` on every call. This wraps a single (I, O) pair to match."""
        def __init__(self, I, O):
            self.data = [([], [], [I], [O], {})]

        def pick(self, data_index=None, **kwargs):
            return self.data[0 if data_index is None else data_index]

    _H, _W = MAX_H, MAX_W

    _simulated = 0
    for i, (I, O) in enumerate(pairs[:2]):
        # A fresh env per pair — reusing one env across differently-shaped
        # pairs via `.unwrapped.data_loader = ...; .reset()` leaves internal
        # state (grid_dim) stuck at the FIRST pair's shape, silently
        # comparing the wrong slice on every pair after the first.
        _env = gym.make(
            "ARCLE/O2ARCv2Env-v0",
            render_mode=None,
            data_loader=_SingleLoader(I, O),
            max_grid_size=(_H, _W),
            colors=10,
            max_episode_steps=None,
            max_trial=1,
        )
        try:
            ops, sels = ns["derive_operations"](I, O)

            def _run_ops(run_ops, run_sels, *, collect_states=False):
                obs, _ = _env.reset(options={"prob_index": 0, "adaptation": False})
                states = []
                if collect_states:
                    states.append((
                        int(obs["grid_dim"][0]),
                        int(obs["grid_dim"][1]),
                        obs["grid"].tobytes(),
                    ))
                for run_op, run_sel in zip(run_ops, run_sels):
                    # to_sel_mask, not sel_bbox_to_mask: a selection may be a bbox
                    # [r,c,h,w] OR {"cells": [[r,c],...]} for a non-rectangular
                    # object. Unpacking the dict as a 4-tuple raised
                    # "not enough values to unpack (expected 4, got 1)" and
                    # rejected every mask-using candidate.
                    mask = _solar_utils.to_sel_mask(run_sel, (_H, _W))
                    action = {"selection": mask.astype(bool), "operation": int(run_op)}
                    obs, _rew, _done, _trunc, _info = _env.step(action)
                    if collect_states:
                        states.append((
                            int(obs["grid_dim"][0]),
                            int(obs["grid_dim"][1]),
                            obs["grid"].tobytes(),
                        ))
                gd_h = int(obs["grid_dim"][0])
                gd_w = int(obs["grid_dim"][1])
                final = obs["grid"][:gd_h, :gd_w].astype(int)
                return final, states

            _final, _states = _run_ops(ops, sels, collect_states=True)
            if not np.array_equal(_final, np.asarray(O)):
                _diffs = [
                    (r, c, int(_final[r, c]), int(O[r, c]))
                    for r in range(O.shape[0])
                    for c in range(O.shape[1])
                    if _final[r, c] != O[r, c]
                ]
                _hint = (
                    f"example {i+1}: trajectory produced wrong output "
                    f"({len(_diffs)} cells wrong). "
                    "First mismatches (row,col,got,want): "
                    + ", ".join(f"({r},{c},{g},{w})" for r, c, g, w in _diffs[:6])
                    + ". Hint: FlipH/V operates on the ENTIRE region including "
                    "pre-existing non-bgc cells — if a non-bgc cell (e.g. marker) "
                    "sits inside a flip selection it gets displaced to the wrong position. "
                    "Fix: (1) clear target region with bgc before each paste+flip, "
                    "(2) after all paste+flip ops restore preserved cells "
                    "(same non-bgc color in I and O) with Color ops (Pattern 7)."
                )
                return False, _hint

            # Reject a trajectory that returns to an earlier visible grid state.
            # Copy/Paste can legitimately revisit the same grid while changing or
            # consuming hidden clipboard state, so intervals containing clipboard
            # ops are excluded from this conservative cycle test.
            _first_state_at = {_states[0]: -1}
            for _k, _state in enumerate(_states[1:-1]):  # exclude Submit's state
                _prev = _first_state_at.get(_state)
                if _prev is not None:
                    _interval_ops = ops[_prev + 1:_k + 1]
                    if not any(int(o) in (28, 29, 30) for o in _interval_ops):
                        return False, (
                            f"example {i+1}: redundant cycle/no-op through op index {_k}; "
                            f"the visible grid already had this state after index {_prev}"
                        )
                # Advance the baseline after a legitimate clipboard-only state so
                # a following ordinary no-op is still detected.
                _first_state_at[_state] = _k

            # Implement the prompt's "could I delete this op?" self-check in validation.
            # Exhaustively test ordinary trajectories. For very long pixel-heavy
            # traces, test every structural/destructive op; immediate no-op Color
            # actions are already caught by the repeated-state check above.
            if i == 0:
                _body_len = max(0, len(ops) - 1)  # never delete Submit
                if _body_len <= 80:
                    _delete_candidates = range(_body_len)
                else:
                    _delete_candidates = [
                        k for k, op in enumerate(ops[:-1])
                        if 20 <= int(op) <= 27 or int(op) in (31, 32, 33)
                    ]
                for _k in _delete_candidates:
                    _reduced_ops = ops[:_k] + ops[_k + 1:]
                    _reduced_sels = sels[:_k] + sels[_k + 1:]
                    _reduced_final, _ = _run_ops(_reduced_ops, _reduced_sels)
                    if np.array_equal(_reduced_final, np.asarray(O)):
                        return False, (
                            f"example {i+1}: redundant op at index {_k}: op={ops[_k]}, "
                            f"selection={sels[_k]}; deleting it still produces O"
                        )
            _simulated += 1
        except Exception as e:
            return False, f"simulation error: {e}"
        finally:
            _env.close()

    if _simulated == 0:
        return False, "simulation never ran on any example pair"

    return True, "ok"

# ── Grid-maker template ────────────────────────────────────────────────────────

TEMPLATE = '''\
"""
ARC Task: {task_id} (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
{derive_fn}


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {{len(category_plan)}} != "
                        f"num_examples+1 ({{num_examples + 1}}) for task {task_id}"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {{len(instance_plan)}} != "
                            f"num_examples+1 ({{num_examples + 1}}) for task {task_id}"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {{j}} after 10 attempts "
                                f"for task {task_id}"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task {task_id} "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {{
                "id":         f"{task_id}-rearc-llm_{{_sn + 1}}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }}))

        return dataset
'''

# ── Per-task worker ────────────────────────────────────────────────────────────

def process_task(tid: str) -> str:
    """Generate grid_maker.py for one task. Returns status string."""
    if tid not in func_bodies:
        return f"SKIP {tid}: not in generators.py"

    out_path = OUTPUT_DIR / tid / "grid_maker.py"
    if out_path.exists() and not args.overwrite:
        return f"SKIP {tid}: exists"

    pairs = _gen_io(tid, args.num_examples + 3)
    if not pairs:
        return f"FAIL {tid}: no I/O pairs"

    src  = func_bodies[tid]
    system_prompt = SYSTEM_PROMPT
    if args.rule_first:
        # Replace the closing "emit code, no prose" line with the two-part
        # RULE-then-code contract. Anchored on the exact line so a prompt edit
        # that drops it fails loudly here instead of silently no-op'ing.
        if _NO_PROSE_LINE not in system_prompt:
            raise RuntimeError(
                "--rule_first: expected no-prose line not found in SYSTEM_PROMPT"
            )
        system_prompt = system_prompt.replace(_NO_PROSE_LINE, _RULE_FIRST_CONTRACT)
    user = _user_prompt(tid, src, pairs)

    if args.dry_run:
        return (
            f"{'=' * 28} SYSTEM ({tid}) {'=' * 28}\n{system_prompt}\n"
            f"{'=' * 28} USER ({tid}) {'=' * 28}\n{user}\n"
            f"DRY  {tid}"
        )

    log_path = str(LOG_DIR / f"{tid}.json") if args.save_log else None

    # Efficient mode validates the visible I→O transformation, not reference-code
    # vocabulary. DSL-faithful mode retains the historical solver-op gate as a
    # separate research objective. Either is overridable with
    # --[no-]enforce_solver_ops.
    _required: dict[str, list[int]] = {}
    if args.enforce_solver_ops is None:
        enforce_solver_ops = args.trajectory_mode == "dsl_faithful"
    else:
        enforce_solver_ops = args.enforce_solver_ops

    if enforce_solver_ops:
        if tid in _gate_shift:  _required["Move(20-23)"]   = [20, 21, 22, 23]
        if tid in _gate_rotate: _required["Rotate(24-25)"] = [24, 25]
        if tid in _gate_flip:   _required["Flip(26-27)"]   = [26, 27]

    _forbidden: dict[str, list[int]] = {}
    for spec in TASK_FEEDBACK.get(tid, {}).get("forbid_ops", []):
        if isinstance(spec, int):
            _forbidden[f"op {spec}"] = [spec]
        elif isinstance(spec, dict):
            label = str(spec.get("label", "operation family"))
            values = spec.get("ops", [])
            if isinstance(values, list) and all(isinstance(v, int) for v in values):
                _forbidden[label] = values

    code: str | None = None
    ok = False
    reason = ""
    rejection_hint = ""

    for attempt in range(args.attempts):
        if not rejection_hint:
            cur_user = user
        else:
            required_fix = (
                "1. Use the required ARCLE object-level ops listed above.\n"
                if _required else
                "1. Do not add reference-code ops to satisfy a hint; simplify the visible I→O trajectory.\n"
            )
            cur_user = (
                user + f"\n\n⛔ Attempt {attempt} rejected: {rejection_hint}\n"
                "Fix derive_operations:\n"
                + required_fix
                + "2. Re-read the generator code — identify every parameter that varies per instance "
                  "(object sizes, positions, counts, colors). Detect these DYNAMICALLY from I and O "
                  "instead of hardcoding values seen in examples. A correct derive_operations must "
                  "work for ANY valid (I, O) pair this generator can produce."
            )
        raw = call_llm(system_prompt, cur_user, timeout=args.llm_timeout,
                       log_path=log_path)
        code = _extract_code(raw) if raw else None
        if not code:
            continue

        # correctness + required-op gate (checked against real returned ops, not source text)
        ok, reason = _validate(
            code, pairs, required_ops=_required, forbidden_ops=_forbidden
        )
        if ok:
            break
        rejection_hint = reason

    if not code:
        return f"FAIL {tid}: no code extracted"

    if not ok and args.write_only_valid:
        return f"WARN {tid}: {reason} (candidate not written)"

    content = TEMPLATE.format(task_id=tid, derive_fn=code)
    (OUTPUT_DIR / tid).mkdir(exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    if ok:
        return f"OK   {tid}"
    return f"WARN {tid}: {reason}"


# ── Main loop ──────────────────────────────────────────────────────────────────
# Guarded so this module can be imported (e.g. to inspect SYSTEM_PROMPT or
# call _user_prompt directly) without kicking off real generation.

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = args.tasks if args.tasks else task_ids

    counters = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "DRY": 0}

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(process_task, tid): tid for tid in targets}
        for future in as_completed(futures):
            result = future.result()
            print(result, flush=True)
            key = result.split()[0].strip(":")
            counters[key] = counters.get(key, 0) + 1

    print(f"\nDone - " + "  ".join(f"{k}={v}" for k, v in counters.items() if v))
