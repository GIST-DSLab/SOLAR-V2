#!/usr/bin/env python3
"""
Stage-3 LLM critic for LLM-generated grid makers.

Stage 1 (answer correctness) is verify_grid_makers.py — A/B/C gates.
Stage 2 (static trajectory checks) is _validate() inside gen_rearc_makers_llm.py,
which already rejects repeated-state cycles and delete-one-op redundancy at
generation time.  What survives both is "right answer, possibly bad process":
the cases a human would have to eyeball.  This script hands those to a separate
critic LLM that never saw the generation prompt.

The critic is deliberately NOT the generator: it gets the maker's code, the
episode it actually produces, a natural-language rendering of the trajectory,
and objective static facts — then must answer in a fixed JSON schema drawn from
the V3 prompt's own forbidden-pattern list, so a finding can be fed straight
back to gen_rearc_makers_llm.py as a rejection hint.

Trajectories are rolled out here rather than read from the .json corpus written
by gen_rearc_trajectories_v2.py: that corpus stores (state, next_action) pairs,
pads to 30x30 with fill value 10, and only keeps trajectories that passed — all
three make it the wrong input for a process audit.

Usage:
    python critique_makers_llm.py --subfolder arc-from-rearc-v6 --tasks 0d3d703e --save_log
    python critique_makers_llm.py --subfolder arc-from-rearc-v6 --parallel 4 --out critique_v6.json
    python critique_makers_llm.py --tasks 0d3d703e --dry_run          # print prompt, no LLM call

Requires the solar_ldcq env (arcle, gymnasium).  `claude -p` inherits
CLAUDE_CONFIG_DIR from that env's activate hook, so the critic runs on the same
account as gen_rearc_makers_llm.py.
"""
import argparse
import ast
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import gymnasium as gym

SOLAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils

REARC_ROOT = SOLAR_ROOT / "re-arc"
LOG_DIR = SOLAR_ROOT / "critique_logs"

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subfolder",    type=str, default="arc-agi-1")
parser.add_argument("--tasks",        nargs="+", default=None)
parser.add_argument("--num_samples",  type=int, default=1,
                    help="Episodes per task shown to the critic (default 1 — each costs an LLM call)")
parser.add_argument("--num_examples", type=int, default=3)
parser.add_argument("--max_grid_dim", nargs=2, type=int, default=[30, 30])
parser.add_argument("--rand_seed",    type=int, default=42)
parser.add_argument("--parallel",     type=int, default=2,
                    help="Concurrent claude calls (default 2)")
parser.add_argument("--timeout",      type=int, default=900,
                    help="Per-call timeout in seconds")
parser.add_argument("--max_steps_shown", type=int, default=60,
                    help="Trajectory steps rendered in full before eliding the middle")
parser.add_argument("--out",          type=str, default=None,
                    help="Write full verdicts to this JSON path")
parser.add_argument("--dry_run",      action="store_true",
                    help="Print the prompt and exit without calling the LLM")
parser.add_argument("--dump_payloads", type=str, default=None,
                    help=("Roll out each maker and write the built payloads to this JSON "
                          "path, without calling the LLM. Lets another tool (e.g. the "
                          "best-of judge) compare several maker versions from one rollout "
                          "each instead of one critique call each."))
parser.add_argument("--save_log",     action="store_true",
                    help="Save each conversation to critique_logs/<task_id>.json")
args = parser.parse_args()

H, W = args.max_grid_dim
MAKER_BASE = SOLAR_ROOT / "maker" / args.subfolder

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)

if args.save_log:
    LOG_DIR.mkdir(exist_ok=True)

CLAUDE_SESSION_LIMIT_REACHED = threading.Event()
PRINT_LOCK = threading.Lock()

# ── critic output schema ──────────────────────────────────────────────────────
# One code per forbidden/wasteful pattern named in SYSTEM_PROMPT_V3, so a finding
# maps back onto the rule the maker was supposed to follow.
FINDING_CODES = {
    "ANSWER_RECONSTRUCTION":     "The rule/parameters used to produce O are taken from O, not measured from I. Drawing after inferring the rule from I is fine; drawing without measuring it is not.",
    "RASTER_ORDER_PAINT":        "Cells edited in top-left raster order rather than by object/region.",
    "OBJECT_SCATTER":            "Edits to one object are split across non-adjacent stretches of the trajectory.",
    "REDUNDANT_CYCLE":           "A sub-sequence returns the grid to a state it already visited.",
    "BOOKKEEPING_OP":            "An op emitted only to help the Python side compute something; no real grid effect.",
    "OVERPAINT_WIDER_THAN_DIFF": "Painted/filled cells that already held their target value.",
    "INPLACE_OP_SWEEP":          "In-place Flip/Rotate caught pre-existing content it should not have transformed.",
    "WRONG_CLIPBOARD_SOURCE":    "CopyO used where CopyI was correct, or vice versa.",
    "HARDCODED_CONSTANT":        "A value true of only some instances is hardcoded instead of measured from I/O.",
    "DSL_MIMICRY":               "Op sequence transliterates the verifier DSL rather than deriving from I/O.",
    "MOVE_MISUSE":               "Move used for something that is not a translation (erase, add, resize).",
    "INFO_DESTROY_THEN_REVIVE":  "Information destroyed by Crop/Resize/erase, then re-read from the input.",
    "LEARNABILITY_GAP":          "A structural variant reachable in test is absent from the examples.",
    "WRONG_RULE":                "Right answer, wrong concept: the trajectory implements a different rule that happens to match O here (e.g. flood-fill where the rule is line-intersection).",
    "UNIDIOMATIC_OP":            "An op used against its idiom: Move for anything but a translation, Copy/Paste for anything but replication, Resize as anything but a last resort, flood-fill where a bounded region-recolor is meant.",
    "INCONSISTENT_WORKSPACE":    "Where the task reshapes/reduces the grid, the build location is not consistent (should be top-left, or resize-first, not wherever each instance happens to land).",
    "CONCEPT_NOT_LEGIBLE":       "Right answer, right rule, but the ops do not express the concept the re-arc solver reveals — a valid but opaque route (e.g. growing-bbox CopyO doubling, or redraw where an object-unit Move is the concept) hides the rule. Not answer-drawing. Fire only when a more legible idiomatic route exists AND was available.",
}
VERDICTS = {"PASS", "REVISE", "FAIL"}
LEVELS = {"high", "medium", "low"}

# ── RE-ARC verifier sources (AST slice, no import) ────────────────────────────
# Same extraction as gen_rearc_makers_llm.py:154-164, copied rather than imported
# because that module parses argv and mkdirs at import time.
def load_verify_bodies() -> dict:
    vsrc = (REARC_ROOT / "verifiers.py").read_text(encoding="utf-8")
    vsrc_lines = vsrc.split("\n")
    bodies = {}
    for node in ast.walk(ast.parse(vsrc)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("verify_"):
            tid = node.name[len("verify_"):]
            bodies[tid] = "\n".join(vsrc_lines[node.lineno - 1 : node.end_lineno])
    return bodies


VERIFY_BODIES = load_verify_bodies()


# ── re-arc-llm solver sources (AST slice, no import) ──────────────────────────
# The solver is the readable CONCEPT — a human's solution to the original ARC
# task. The critic reads it to state "the rule you failed to measure is …"
# accurately, instead of reverse-engineering it from the opaque verifier DSL.
# (Generators must NOT be handed the solver — the pilot showed it makes them draw
# the answer; its value is on the critic side only.) It won't run here (PEP 695
# syntax needs 3.12+, arc_types.py is absent) but we only need its text.
def load_solver_bodies() -> dict:
    path = SOLAR_ROOT / "re-arc-llm" / "solvers.py"
    if not path.exists():
        return {}
    ssrc = path.read_text(encoding="utf-8")
    slines = ssrc.split("\n")
    bodies = {}
    for node in ast.walk(ast.parse(ssrc)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("solve_"):
            tid = node.name[len("solve_"):]
            bodies[tid] = "\n".join(slines[node.lineno - 1 : node.end_lineno])
    return bodies


SOLVER_BODIES = load_solver_bodies()


# ── original ARC task pairs ───────────────────────────────────────────────────
# The human ground truth. RE-ARC augments these; where the augmentation drifts
# from the original concept, the original is the reference. The critic needs them
# to judge whether a task is one a human solves by inference (so drawing the
# answer without measuring I is wrong) or one where drawing after inference is
# legitimate — a distinction the verifier alone cannot make. Located via the
# arcle package so the path is not hardcoded to one env layout.
def load_orig_arc_pairs() -> dict:
    try:
        import arcle
    except Exception:
        return {}
    base = Path(arcle.__file__).parent / "arcs" / "ARC" / "data"
    pairs = {}
    for split_dir in (base / "training", base / "evaluation"):
        if not split_dir.is_dir():
            continue
        for p in split_dir.glob("*.json"):
            if p.stem in pairs:
                continue
            try:
                pairs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return pairs


ORIG_ARC_PAIRS = load_orig_arc_pairs()

# ── maker source slicing ──────────────────────────────────────────────────────
# Every generated maker carries these two banners; the region between them is the
# only part the LLM wrote, and the only part worth criticising.
_LLM_BANNER = re.compile(r"^# ─+ LLM-generated.*$", re.M)
_GM_BANNER = re.compile(r"^# ─+ GridMaker.*$", re.M)


def extract_llm_code(src: str) -> str:
    """Return the LLM-authored region of a grid_maker.py, or the whole file."""
    start = _LLM_BANNER.search(src)
    end = _GM_BANNER.search(src)
    if start and end and start.end() < end.start():
        return src[start.end() : end.start()].strip("\n")
    return src


# ── grid helpers ──────────────────────────────────────────────────────────────

def render_grid(g) -> str:
    g = np.asarray(g)
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def infer_bgc(grid) -> int:
    """Most common colour. A heuristic the V3 prompt itself calls unreliable —
    labelled as such wherever it reaches the critic."""
    return Counter(np.asarray(grid).flatten().tolist()).most_common(1)[0][0]


def find_objects(grid, bgc: int) -> list:
    """4-connected same-colour components of non-background cells."""
    g = np.asarray(grid)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bgc:
                continue
            colour = int(g[r, c])
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == colour:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [x for x, _ in cells]
            cs = [y for _, y in cells]
            objs.append({
                "id": len(objs) + 1,
                "color": colour,
                "size": len(cells),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "cells": set(cells),
            })
    return objs


def sel_origin(sel) -> tuple:
    """Top-left (r, c) anchor of a selection.

    bbox -> its origin; {"cells": [...]} -> the min cell. Used for raster-order
    comparisons, so a non-rectangular object selection needs an anchor too.
    """
    if isinstance(sel, dict) and "cells" in sel:
        cells = sel["cells"]
        if not cells:
            return (0, 0)
        return (min(int(r) for r, _ in cells), min(int(c) for _, c in cells))
    return (sel[0], sel[1])


def sel_cells(sel, shape) -> set:
    """Cells a selection covers, clipped to the grid.

    Accepts a [r, c, h, w] bbox or {"cells": [[r,c],...]} — the non-rectangular
    object selection a maker emits via sel_of().
    """
    gh, gw = shape
    if isinstance(sel, dict) and "cells" in sel:
        return {(int(rr), int(cc)) for rr, cc in sel["cells"]
                if 0 <= int(rr) < gh and 0 <= int(cc) < gw}
    r, c, dh, dw = sel
    return {
        (rr, cc)
        for rr in range(max(0, r), min(gh, r + dh + 1))
        for cc in range(max(0, c), min(gw, c + dw + 1))
    }


# ── rollout ───────────────────────────────────────────────────────────────────

def crop(obs) -> np.ndarray:
    gh = int(obs["grid_dim"][0])
    gw = int(obs["grid_dim"][1])
    return obs["grid"][:gh, :gw].astype(int).copy()


def rollout(env, obs, ops, sels):
    """Grid state after every op. states[k] --ops[k]--> states[k+1]."""
    states = [crop(obs)]
    for op, sel in zip(ops, sels):
        mask = solar_utils.to_sel_mask(sel, (H, W))
        action = {"selection": mask.astype(bool), "operation": int(op)}
        obs, _r, _term, _trunc, _info = env.step(action)
        states.append(crop(obs))
    return states


# ── trajectory → natural language ─────────────────────────────────────────────

def op_category(op: int) -> str:
    if op < 10:
        return "Color"
    if op < 20:
        return "FloodFill"
    if op < 24:
        return "Move"
    if op < 26:
        return "Rotate"
    if op < 28:
        return "Flip"
    if op < 32:
        return "Copy/Paste"
    if op == 32:
        return "ResetGrid"
    if op == 33:
        return "ResizeGrid"
    return "Submit"


def describe_step(k, op, sel, before, after, objs) -> str:
    name = solar_utils.mapping_operation(int(op))
    head = f"[{k:>3}] {name:<11} sel={list(sel)}"

    if before.shape != after.shape:
        return f"{head}  grid {before.shape} -> {after.shape}"

    diff = np.argwhere(before != after)
    if len(diff) == 0:
        return f"{head}  NO-OP (grid unchanged)"

    touched = sel_cells(sel, before.shape)
    hits = [o["id"] for o in objs if o["cells"] & touched]
    obj_note = f"  touches obj#{','.join(map(str, hits))}" if hits else "  touches no object in I"

    shown = "; ".join(
        f"({r},{c}) {int(before[r, c])}->{int(after[r, c])}" for r, c in diff[:4]
    )
    more = f" +{len(diff) - 4} more" if len(diff) > 4 else ""
    return f"{head}  {len(diff)} cell(s): {shown}{more}{obj_note}"


def summarize_trajectory(ops, sels, states, objs) -> str:
    lines = []
    n = len(ops)
    keep = args.max_steps_shown
    for k in range(n):
        if n > keep and keep // 2 <= k < n - keep // 2:
            if k == keep // 2:
                lines.append(f"      ... {n - keep} steps elided ...")
            continue
        lines.append(describe_step(k, ops[k], sels[k], states[k], states[k + 1], objs))
    return "\n".join(lines)


# ── objective static facts ────────────────────────────────────────────────────
# Observations only — no pass/fail. The critic does the judging; these exist so it
# judges against measurements instead of its own reading of the grids.

def static_facts(ops, sels, states) -> dict:
    facts = {}
    facts["n_ops"] = len(ops)
    facts["categories"] = dict(Counter(op_category(int(o)) for o in ops))

    # Submit never changes the grid; counting it would report every trajectory as
    # ending in a no-op and a cycle. Same exemption as _validate and
    # analyze_trajectory_quality.
    body = len(ops) - 1 if ops and int(ops[-1]) == 34 else len(ops)

    noop = [k for k in range(body)
            if states[k].shape == states[k + 1].shape and np.array_equal(states[k], states[k + 1])]
    facts["noop_steps"] = noop

    # Repeated grid state = the grid returned somewhere it had already been.
    seen = {}
    repeats = []
    for k, s in enumerate(states[: body + 1]):
        key = (s.shape, s.tobytes())
        if key in seen:
            repeats.append([seen[key], k])
        seen[key] = k
    facts["repeated_state_spans"] = repeats

    # Longest run of Color/FloodFill ops whose selection walks forward in raster
    # order. Long runs are what "painted by raster scan, not by object" looks
    # like; short ones are normal. The critic decides which this is.
    best = run = 0
    prev = None
    for op, sel in zip(ops, sels):
        if int(op) < 20:
            key = sel_origin(sel)
            run = run + 1 if prev is not None and key >= prev else 1
            prev = key
            best = max(best, run)
        else:
            run, prev = 0, None
    facts["longest_raster_ordered_paint_run"] = best

    facts["resetgrid_used"] = any(int(o) == 32 for o in ops)
    facts["copyinput_at_index_0"] = bool(ops) and int(ops[0]) == 31
    facts["submit_last"] = bool(ops) and int(ops[-1]) == 34
    return facts


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You audit ARC grid makers written for the ARCLE environment by another LLM.

The maker already produces the CORRECT answer: the op sequence below was re-run
in ARCLE just now and its final grid equals O exactly. Do not re-check
correctness — spending your attention there is waste. Your only question is
whether the SOLUTION PROCESS is honest:

    Does this trajectory perform the task's actual transformation, or does it
    reach the right grid by a route that would not generalise and that a human
    reviewer would reject?

You are given the TRUE RULE from two readable sources, so you never have to guess
it from the opaque verifier:
- The ORIGINAL ARC task shows the task as a human sees it — the worked examples
  they reason from. It is the reference when the maker's augmented examples drift.
- The re-arc-llm SOLVER is a human's solution: the concept in high-level form.
Read both, state the rule to yourself, then check whether the maker's trajectory
actually measures that rule from I — or merely arrives at O by another route. When
you name a rule the maker "failed to measure", it must be the rule these sources
show, not one you inferred from O.

The maker was written against a prompt that forbids the patterns below. Judge
against these, and nothing else:

{chr(10).join(f"- {k}: {v}" for k, v in FINDING_CODES.items())}

ANSWER_RECONSTRUCTION — apply this test, do not over-fire it:
    The forbidden thing is not drawing, and not referencing O. It is reaching O
    WITHOUT MEASURING THE RULE FROM I. Ask:

      Where do the colours/positions the trajectory paints come from — did the
      code measure the rule or its parameters from I (and the examples), or read
      them straight out of O?

    - Reads no structure from I; loops over O and paints each differing cell
      (`for r,c: v = O[r,c]; paint v at r,c`) → ANSWER_RECONSTRUCTION.
    - Measures the rule from I — a colour map, a count, an object's shape, a
      line/region — then paints by that rule → NOT a finding, even if O is the
      paint target and even if the final sweep looks like drawing. Some ARC tasks
      (see the original examples) are genuinely solved by inferring the rule and
      then rendering it; that is the human solution, not a shortcut.
    - Decide which case you are in using the ORIGINAL ARC examples (is the rule
      inferable from examples?) and the SOLVER (what a legitimate computation
      path looks like). An O-read count is a hint, never the verdict — a
      colour-map task reads O and is still sound; a copy-O task reads O and is
      not. Judge the source of the RULE, not the presence of O.

WRONG_RULE vs ANSWER_RECONSTRUCTION — both give the right grid by a bad route,
but tell them apart:
    - ANSWER_RECONSTRUCTION measures no rule — it copies O.
    - WRONG_RULE measures a rule from I, but the WRONG one: a different
      transformation that coincides with O on this episode. Check the maker's
      rule against the concept in the original ARC examples and the solver. If the
      maker floods a region where the true rule draws intersecting lines
      (23581191), or reads O.shape where the true rule counts separator lines
      (1190e5a7), that is WRONG_RULE even though nothing copies O. A WRONG_RULE at
      high severity is a FAIL — it would not generalise.

CONCEPT_NOT_LEGIBLE — the answer is right, the rule is right, but the trajectory
takes a valid-yet-opaque route that hides the rule. Use the re-arc solver below to
identify WHAT the rule is — then judge legibility against the VISIBLE I→O change,
not against the solver's choice of primitives. Perfect mimicry is NOT required; a
different-but-legible route passes.

THE SOLVER IS A DECLARATIVE SPEC, NOT AN EXECUTION PLAN — the same caveat as the
verifier (see DSL_MIMICRY). Its DSL may express a visible motion as `sort`,
`replace`, `rot*`, index arithmetic, or a set operation, purely because that is
short to write in that DSL. A solver written that way does NOT license a trajectory
that erases and repaints. Ask what CHANGED between I and O, in the grid the human
sees:
  - the SAME cell-set (shape + colour) present at a new position  -> a TRANSLATION,
    whatever the solver called it. Expect Move.
  - cells that change colour in place, or a pattern that did not exist before
    -> genuinely a paint.
If the solver says `sort` but the evidence shows each object arriving intact at a
new position, the concept is MOVEMENT and you must judge it as such.

CONCRETE TRIGGERS — when you observe one of these, FIRE this code at >= medium.
Do NOT file the observation under a lesser code (OVERPAINT_WIDER_THAN_DIFF,
UNIDIOMATIC_OP) and move on: the opaque route IS the finding, and describing it
in your summary without firing is a miss.
  - The rule is a TRANSLATION and the trajectory ERASES the object's old region
    and REPAINTS it at the new location (Color/FloodFill sweep) instead of
    selecting the object's exact cells and Move-ing them. Fire even when the
    erase is harmless / hits nothing else. "Erase here, repaint there" is never
    a translation made legible.
    Decide "is it a translation?" from the I->O evidence, NOT from the solver's
    vocabulary. A clear-then-repile, clear-then-restack, or paint-the-settled-
    block pair IS erase-and-repaint however the maker's comments describe it, and
    a maker phrasing it as "rebuild the column" does not make it legible.
    Note also: ARCLE Move leaves 0 in the vacated cells, so a correct Move-based
    trajectory over a non-zero background ENDS with one Color op repairing that
    trail. That repair op is part of the translation — do not read it as an
    erase, and do not treat the need for it as a reason the maker was right to
    avoid Move.
  - A periodic fill done by growing-bbox CopyO "doubling" that never exposes the
    period (stamp the base unit at each period offset instead).
  - The rule names a whole OBJECT but the ops only ever address its bounding
    rectangle, so the object never appears as an object.

EXCEPTIONS, narrowly:
  - Drawing is legitimate when it is the ONLY route or genuinely the idiomatic
    one. A task solved by inferring the rule and then painting the result is
    fine — but that is painting a NEW pattern, not erasing an object and
    repainting the SAME object elsewhere. The forbidden thing is reaching O
    without measuring the rule from I (see ANSWER_RECONSTRUCTION); drawing per
    se is not.
  - Do not fire merely because the maker's method differs from the solver's.
  - A selection may be an object's EXACT cells, so two objects that merely TOUCH
    are still separable. Only genuinely overlapping cells make isolation
    impossible — only then is a repaint/Copy-Paste workaround legitimate.

UNIDIOMATIC_OP — the op works but violates what the op is FOR. Move is for
translating an object; Copy/Paste for replicating one; Resize is a last resort
(when the grid truly changes size), not a convenience; flood-fill is for
recolouring a bounded region, not for painting arbitrary cells. A forced Move that
merely displaces one cell where a Color op is the natural edit, a double Resize,
or a Rotate with no reason are UNIDIOMATIC_OP. (An op that is simply deletable —
removing it still yields O — is caught by stage 2 already; this is about ops that
are load-bearing but wrong for the job.)

Key ARCLE semantics you need (the generator's prompt asserts these; assume them):
- A selection is an ARBITRARY CELL MASK — the exact set of cells the op acts on. A
  maker emits it with `sel_of(cells)`. It need NOT be a rectangle, so an object of
  any shape can be selected and Moved as itself. Never excuse a redraw on the
  grounds that "the object is not rectangular" — that limit no longer exists.
  A bbox `[r, c, h, w]` is the shorthand for a FULL rectangle, h/w being offsets
  EXCLUDING the cell itself: [2,3,0,0] is the single cell (2,3).
- Move/Rotate/Flip relocate the whole selection and CLEAR it from the background,
  leaving 0 (not the background colour) in every vacated cell. Over a non-zero
  background a correct Move chain therefore ends with a Color op repairing that
  trail; that op is part of the translation, not an erase.
- op24 Rotate90 = CCW, op25 Rotate270 = CW. op26 FlipH = left<->right,
  op27 FlipV = up<->down. Rotate on a non-square selection is broken.
- op28 CopyI copies from the ORIGINAL INPUT; op29 CopyO from the CURRENT grid.
- Copy/Paste/Crop treat colour 0 as "nothing": 0 cells are never copied or pasted.
- op32 ResetGrid is forbidden outright. op31 CopyInput at index 0 is a no-op.

Rules of evidence:
- Cite an op index or a line of the maker's code in every finding. A finding you
  cannot anchor to a specific op or line is not a finding — drop it.
- The static facts block is measured, not inferred. Trust it over your own
  reading of the grids. In particular: an op listed in noop_steps really did
  nothing; longest_raster_ordered_paint_run really is that long.
- Background colour, where quoted, is a most-common-colour guess and is wrong on
  foreground-heavy or near-even grids. Do not build a finding on it alone.
- You see ONE episode. A constant that matches this episode may still be
  hardcoded — check the code against the generator's parameter ranges before
  calling HARDCODED_CONSTANT, and say so in the evidence.
- Judge the process, not the style. Verbose but honest code is a PASS.

Verdicts:
- PASS   — process is sound; no finding above medium severity.
- REVISE — a real flaw, but the approach is basically right.
- FAIL   — the trajectory does not derive I->O (reconstruction, DSL transliteration,
           or a rule that only works on this episode).

Reply with ONE ```json``` block and no prose, matching exactly:

```json
{{
  "verdict": "PASS" | "REVISE" | "FAIL",
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence",
  "findings": [
    {{
      "code": "<one of the codes above>",
      "severity": "high" | "medium" | "low",
      "op_indices": [<int>, ...],
      "evidence": "<what you saw, citing op index or code line>"
    }}
  ]
}}
```

"findings" is [] when the verdict is PASS. Report only what you can defend."""


def build_user_prompt(tid, ex_in, ex_out, pr_in, pr_out, ops, sels, states, code) -> str:
    bgc = infer_bgc(pr_in)
    objs = find_objects(pr_in, bgc)
    facts = static_facts(ops, sels, states)

    parts = [f"# Task {tid}\n"]

    parts.append("## Original ARC task (human ground truth)\n")
    orig = ORIG_ARC_PAIRS.get(tid)
    if orig:
        parts.append(
            f"This task, as a human sees it: {len(orig.get('train', []))} worked "
            "example(s) below, then a test input to solve. RE-ARC augments this task; "
            "where the maker's examples drift from the concept shown here, this original "
            "is the reference. Use it to decide whether the rule is meant to be INFERRED "
            "from the examples (so reaching O without measuring the rule from I is "
            "ANSWER_RECONSTRUCTION) or whether drawing after inference is the natural "
            "solution.\n"
        )
        for j, pr in enumerate(orig.get("train", [])):
            parts.append(f"orig example {j} input:\n{render_grid(pr['input'])}\n")
            parts.append(f"orig example {j} output:\n{render_grid(pr['output'])}\n")
    else:
        parts.append("(original ARC task not found — judge from the examples below)\n")

    parts.append("## re-arc-llm solver (readable concept — WHAT the rule is)\n")
    solver_src = SOLVER_BODIES.get(tid)
    if solver_src:
        parts.append("```python\n" + solver_src + "\n```\n")
        parts.append(
            "A human's solution to the original task. Read it to know the true rule so "
            "you can state precisely what the maker measured or failed to measure. It is "
            "the CONCEPT, not a template: its call order and hardcoded sizes are DSL "
            "artefacts, and a maker that transliterates it op-for-op is committing "
            "DSL_MIMICRY.\n"
        )
    else:
        parts.append("(no solver available for this task)\n")

    parts.append("## RE-ARC verifier (ground-truth rule, DSL form)\n")
    parts.append("```python\n" + VERIFY_BODIES.get(tid, "<not found>") + "\n```\n")
    parts.append(
        "The maker must derive this rule from I and O. Transliterating this op list "
        "into ARCLE ops is DSL_MIMICRY.\n"
    )

    parts.append("## Training examples the maker produced\n")
    for j, (ei, eo) in enumerate(zip(ex_in, ex_out)):
        parts.append(f"example {j} input:\n{render_grid(ei)}\n")
        parts.append(f"example {j} output:\n{render_grid(eo)}\n")

    parts.append("## Test instance — the trajectory below solves this one\n")
    parts.append(f"input:\n{render_grid(pr_in)}\n")
    parts.append(f"output:\n{render_grid(pr_out)}\n")

    parts.append(f"## Objects in the test input (4-connected, bgc={bgc} by most-common guess)\n")
    if objs:
        for o in objs:
            parts.append(
                f"obj#{o['id']}: color={o['color']} size={o['size']} "
                f"bbox=(r{o['bbox'][0]}..{o['bbox'][2]}, c{o['bbox'][1]}..{o['bbox'][3]})"
            )
    else:
        parts.append("(none — the bgc guess may be wrong)")
    parts.append("")

    parts.append("## Trajectory, rolled out in ARCLE\n")
    parts.append("Each line is: [index] OpName sel=[r,c,h,w]  what changed  which object it hit.\n")
    parts.append("```\n" + summarize_trajectory(ops, sels, states, objs) + "\n```\n")

    parts.append("## Static facts (measured)\n")
    parts.append("```json\n" + json.dumps(facts, indent=2) + "\n```\n")

    parts.append("## The maker's LLM-written code\n")
    parts.append("```python\n" + code + "\n```\n")

    parts.append(
        "Audit the trajectory against the forbidden patterns. Reply with the JSON block only."
    )
    return "\n".join(parts)


# ── claude -p subprocess ──────────────────────────────────────────────────────

def call_claude(system: str, user: str, log_path: str | None = None) -> str | None:
    """Mirrors gen_rearc_makers_llm.py:call_claude so both stages hit the same
    account and CLI surface."""
    if CLAUDE_SESSION_LIMIT_REACHED.is_set():
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as sf:
        sf.write(system)
        sys_file = sf.name

    cmd = [
        shutil.which("claude") or "claude", "-p",
        "--system-prompt-file", sys_file,
        "--output-format", "text",
        "--dangerously-skip-permissions",
    ]

    stdout = stderr = None
    returncode = None
    try:
        result = subprocess.run(
            cmd, input=user, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=args.timeout,
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        if returncode != 0:
            detail = (stderr or stdout or "<no output>").strip()
            if "session limit" in detail.lower():
                if not CLAUDE_SESSION_LIMIT_REACHED.is_set():
                    CLAUDE_SESSION_LIMIT_REACHED.set()
                    print("    Claude session limit reached; skipping remaining calls.")
            print(f"    claude failed (exit {returncode}): {detail[:300]}")
            return None
        return stdout
    except subprocess.TimeoutExpired:
        print(f"    claude timed out after {args.timeout}s")
        return None
    except FileNotFoundError:
        print("    ERROR: `claude` CLI not found in PATH")
        return None
    finally:
        Path(sys_file).unlink(missing_ok=True)
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "system_prompt": system,
                    "user_prompt":   user,
                    "response":      stdout,
                    "stderr":        stderr,
                    "returncode":    returncode,
                }, f, indent=2, ensure_ascii=False)


# ── verdict parsing ───────────────────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_verdict(text: str) -> tuple[dict | None, str]:
    """Constrain the critic to the schema. Returns (verdict, error)."""
    if not text:
        return None, "empty response"

    m = _JSON_FENCE.search(text)
    raw = m.group(1) if m else None
    if raw is None:                      # tolerate a bare object
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return None, "no JSON object in response"
        raw = text[s : e + 1]

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"malformed JSON: {e}"

    if obj.get("verdict") not in VERDICTS:
        return None, f"verdict must be one of {sorted(VERDICTS)}, got {obj.get('verdict')!r}"
    if obj.get("confidence") not in LEVELS:
        return None, f"confidence must be one of {sorted(LEVELS)}, got {obj.get('confidence')!r}"
    if not isinstance(obj.get("findings"), list):
        return None, "findings must be a list"

    for f in obj["findings"]:
        if not isinstance(f, dict):
            return None, "each finding must be an object"
        if f.get("code") not in FINDING_CODES:
            return None, f"unknown finding code {f.get('code')!r}"
        if f.get("severity") not in LEVELS:
            return None, f"severity must be one of {sorted(LEVELS)}, got {f.get('severity')!r}"
        if not isinstance(f.get("op_indices"), list):
            return None, "op_indices must be a list"
        if not str(f.get("evidence", "")).strip():
            return None, "evidence must be non-empty"

    obj.setdefault("summary", "")
    return obj, ""


# ── per-task driver ───────────────────────────────────────────────────────────

def build_payload(tid: str) -> dict:
    """Load the maker and roll out one episode into a critic prompt.

    MUST run serially: every grid_maker.py purges 'utils'/'dsl'/'generators' from
    sys.modules and re-imports them off re-arc, so two concurrent loads race on
    global interpreter state and one of them dies with KeyError: 'utils'.
    """
    maker_path = MAKER_BASE / tid / "grid_maker.py"
    if not maker_path.exists():
        return {"task_id": tid, "error": "grid_maker.py not found"}

    src = maker_path.read_text(encoding="utf-8")
    code = extract_llm_code(src)

    spec = importlib.util.spec_from_file_location(f"gm_{tid}", str(maker_path))
    gm_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gm_mod)
    except Exception as e:
        return {"task_id": tid, "error": f"load: {e}"[:200]}

    loader = gm_mod.GridMaker(
        rand_seed=args.rand_seed,
        num_samples=args.num_samples,
        num_examples=args.num_examples,
    )
    try:
        loader.parse(num_samples=args.num_samples, num_examples=args.num_examples,
                     max_grid_dim=[H, W])
    except Exception as e:
        return {"task_id": tid, "error": f"parse: {e}"[:200]}

    env = gym.make(
        "ARCLE/O2ARCv2Env-v0", render_mode=None, data_loader=loader,
        max_grid_size=(H, W), colors=10, max_episode_steps=None, max_trial=1,
    )

    for i, sample in enumerate(loader.data):
        ex_in, ex_out, pr_in, pr_out, desc = sample
        if not pr_in or not pr_out:
            continue
        ops, sels = desc["operations"], desc["selections"]
        pr_in0, pr_out0 = np.asarray(pr_in[0]), np.asarray(pr_out[0])
        if pr_out0.shape[0] > H or pr_out0.shape[1] > W:
            continue

        try:
            obs, _info = env.reset(options={"prob_index": i, "adaptation": False})
            states = rollout(env, obs, ops, sels)
        except Exception as e:
            return {"task_id": tid, "error": f"rollout: {e}"[:200]}

        # Stage 1 over again, on this exact episode. The critic prompt tells the
        # LLM the answer is already correct and forbids re-checking it, so that
        # had better be true — and the recorded A-gate results come from another
        # machine and another seed. A maker that fails here needs regeneration,
        # not a process audit.
        final = states[-1]
        if final.shape != pr_out0.shape or not np.array_equal(final, pr_out0):
            return {"task_id": tid,
                    "error": "stage-1 fail: trajectory does not reproduce O (skipped)"}

        user = build_user_prompt(tid, ex_in, ex_out, pr_in0, pr_out0,
                                 ops, sels, states, code)
        return {"task_id": tid, "user": user, "n_ops": len(ops)}

    return {"task_id": tid, "error": "no usable episode"}


def run_critic(payload: dict) -> dict:
    """The LLM call. Safe to run concurrently — touches no global state."""
    tid = payload["task_id"]
    if "error" in payload:
        return payload

    user = payload["user"]
    log_path = str(LOG_DIR / f"{tid}.json") if args.save_log else None
    raw = call_claude(SYSTEM_PROMPT, user, log_path=log_path)
    verdict, err = parse_verdict(raw or "")

    if verdict is None and raw:
        # One retry, telling it exactly what was wrong with the shape.
        raw = call_claude(
            SYSTEM_PROMPT,
            user + f"\n\nYour previous reply was rejected: {err}\n"
                   "Reply with ONE ```json``` block matching the schema exactly.",
            log_path=log_path,
        )
        verdict, err = parse_verdict(raw or "")

    if verdict is None:
        return {"task_id": tid, "error": f"critic: {err}"[:200]}

    verdict["task_id"] = tid
    verdict["n_ops"] = payload["n_ops"]
    return verdict


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not MAKER_BASE.is_dir():
        sys.exit(f"no such maker dir: {MAKER_BASE}")

    tids = sorted(d.name for d in MAKER_BASE.iterdir() if d.is_dir())
    if args.tasks:
        tids = [t for t in tids if t in args.tasks]
        missing = set(args.tasks) - set(tids)
        if missing:
            print(f"warning: not in {args.subfolder}: {' '.join(sorted(missing))}")
    if not tids:
        sys.exit("no tasks selected")

    print(f"Critiquing {len(tids)} task(s) in '{args.subfolder}' "
          f"({args.parallel} parallel)\n")

    # Serial: rolling out a maker mutates sys.modules (see build_payload).
    payloads = [build_payload(tid) for tid in tids]

    if args.dump_payloads:
        Path(args.dump_payloads).write_text(
            json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
        ok = sum(1 for p in payloads if "error" not in p)
        print(f"wrote {len(payloads)} payload(s) ({ok} rolled out, "
              f"{len(payloads) - ok} error) -> {args.dump_payloads}")
        return

    if args.dry_run:
        for p in payloads:
            if "error" in p:
                print(f"{p['task_id']:<10} SKIP   {p['error']}")
                continue
            print("=" * 30, f" SYSTEM ({p['task_id']}) ", "=" * 30)
            print(SYSTEM_PROMPT)
            print("=" * 30, f" USER ({p['task_id']}) ", "=" * 30)
            print(p["user"])
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for res in pool.map(run_critic, payloads):
            results.append(res)
            with PRINT_LOCK:
                tid = res["task_id"]
                if "error" in res:
                    print(f"{tid:<10} ERROR  {res['error']}")
                else:
                    codes = ",".join(f["code"] for f in res["findings"]) or "-"
                    print(f"{tid:<10} {res['verdict']:<6} ({res['confidence']:<6}) {codes}")

    # ── summary ──
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    print("\n" + "=" * 60)
    print(f"critiqued {len(ok)} / {len(results)}   errors {len(errs)}")
    if ok:
        by_verdict = Counter(r["verdict"] for r in ok)
        for v in ("PASS", "REVISE", "FAIL"):
            n = by_verdict.get(v, 0)
            print(f"  {v:<7} {n:>4}  ({100 * n / len(ok):.1f}%)")

        code_counts = Counter(f["code"] for r in ok for f in r["findings"])
        if code_counts:
            print("\nfindings by code:")
            for code, n in code_counts.most_common():
                print(f"  {n:>4}  {code}")

        flagged = [r["task_id"] for r in ok if r["verdict"] != "PASS"]
        if flagged:
            print(f"\nnot PASS ({len(flagged)}): {' '.join(flagged)}")
    if errs:
        print(f"\nerrors ({len(errs)}): {' '.join(r['task_id'] for r in errs)}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
