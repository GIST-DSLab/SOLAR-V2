"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# ----------------------------------------------------------------------------
# Task 3ac3eb23
#
# Input : a canvas of bgc with a few single marker cells sitting on ONE edge
#         line of the grid (top row / bottom row / left col / right col — the
#         generator draws them on the top row and then rotates the whole grid).
# Output: from every marker a "diamond chain" grows away from that edge:
#         at even distance d from the edge the marker colour sits on the
#         marker's own line, at odd distance it sits on the two lines beside it.
#
# The verifier normalises the grid with a reflection (dmirror / cmirror /
# hmirror) so the markers lie on the near edge, draws, and reflects back.
# The trajectory does exactly that: FlipV (op27) when the markers are on the
# bottom row, FlipH (op26) when they are on the right column, then one Color op
# per marker painting its whole chain, then the same flip back (only when that
# flip still changes the grid — for an odd extent the drawn grid is already
# symmetric and a second flip would be an invisible no-op).
# ----------------------------------------------------------------------------

MF_VARIANTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # keep colour 0 out of the foreground when the background is non-zero, so a
    # whole-grid Flip never has to carry a "0 is real content" cell.
    if bgc == 0:
        pool = [c for c in cols if c != 0]
    else:
        pool = [c for c in cols if c != bgc and c != 0]
    k = random.randint(2, min(5, len(pool)))
    fgcols = random.sample(pool, k)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(MF_VARIANTS):
        examples = [{"mf_name": v} for v in MF_VARIANTS]
        examples += [{"mf_name": random.choice(MF_VARIANTS)}
                     for _ in range(n_ex - len(MF_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"mf_name": v} for v in random.sample(MF_VARIANTS, n_ex)]
    plan = [dict(e) for e in examples]
    plan.append(dict(random.choice(examples)))  # test orientation was shown
    return {"bgc": bgc, "fgcols": fgcols, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    ba = max(a, int(round(a + (b - a) * diff_lb)))
    bb = min(b, int(round(a + (b - a) * diff_ub)))
    if bb < ba:
        ba, bb = bb, ba
    ba = max(a, min(b, ba))
    bb = max(a, min(b, bb))
    return random.randint(ba, bb)


def _rot90cw(g):
    return [list(r) for r in zip(*g[::-1])]


def _rot180(g):
    return [list(r)[::-1] for r in g[::-1]]


def _rot270ccw(g):
    return [list(r) for r in zip(*g)][::-1]


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgcols=None, mf_name=None) -> dict:
    cols = list(range(10))
    if fgcols is None:
        if bgc == 0:
            pool = [c for c in cols if c != 0]
        else:
            pool = [c for c in cols if c != bgc and c != 0]
        fgcols = random.sample(pool, min(3, len(pool)))
    if mf_name is None:
        mf_name = random.choice(MF_VARIANTS)

    # a rot90/rot270 swaps the two dimensions, so keep both inside both caps
    if mf_name in ("rot90", "rot270"):
        hb = wb = max(3, min(max_h, max_w))
    else:
        hb, wb = max(3, max_h), max(3, max_w)

    h = _unifint(diff_lb, diff_ub, (3, hb))
    w = _unifint(diff_lb, diff_ub, (3, wb))

    nlocs = _unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]

    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x <= locj + 2)]
        col = random.choice(fgcols)
        gi[0][locj] = col
        for p in range(0, h, 2):
            go[p][locj] = col
        for p in range(1, h, 2):
            go[p][locj - 1] = col
            go[p][locj + 1] = col

    if mf_name == "rot90":
        gi, go = _rot90cw(gi), _rot90cw(go)
    elif mf_name == "rot180":
        gi, go = _rot180(gi), _rot180(go)
    elif mf_name == "rot270":
        gi, go = _rot270ccw(gi), _rot270ccw(go)

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the canvas colour the generator paints before placing markers
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    marks = [(r, c, int(I[r, c]))
             for r in range(h) for c in range(w) if int(I[r, c]) != bgc]

    full = [0, 0, h - 1, w - 1]   # bbox == the ENTIRE grid (a true full rectangle)
    cur = I.copy()

    mrows = {r for r, _, _ in marks}
    mcols = {c for _, c, _ in marks}

    # which edge do the markers sit on?  (markers are never on a corner, so this
    # is unambiguous).  Reflect the grid so they end up on the top row / left col.
    if mrows == {0}:
        axis, flip_op = "row", None
    elif mrows == {h - 1}:
        axis, flip_op = "row", 27          # FlipV: bottom row -> top row
    elif mcols == {0}:
        axis, flip_op = "col", None
    else:
        axis, flip_op = "col", 26          # FlipH: right col -> left col

    if flip_op == 27:
        ops.append(27); sels.append(full)
        cur = np.flipud(cur)
        marks = [(h - 1 - r, c, col) for r, c, col in marks]
    elif flip_op == 26:
        ops.append(26); sels.append(full)
        cur = np.fliplr(cur)
        marks = [(r, w - 1 - c, col) for r, c, col in marks]

    # grow one diamond chain per marker, away from the edge it sits on
    marks.sort(key=(lambda t: t[1]) if axis == "row" else (lambda t: t[0]))
    for r0, c0, col in marks:
        cells = []
        if axis == "row":
            for r in range(0, h, 2):                 # spine, on the marker's column
                cells.append((r, c0))
            for r in range(1, h, 2):                 # the two arms
                for cc in (c0 - 1, c0 + 1):
                    if 0 <= cc < w:
                        cells.append((r, cc))
        else:
            for c in range(0, w, 2):                 # spine, on the marker's row
                cells.append((r0, c))
            for c in range(1, w, 2):                 # the two arms
                for rr in (r0 - 1, r0 + 1):
                    if 0 <= rr < h:
                        cells.append((rr, c))
        cells = sorted(set(cells))
        cells = [p for p in cells if int(cur[p[0], p[1]]) != col]  # marker itself already holds col
        if not cells:
            continue
        ops.append(int(col)); sels.append(sel_of(cells))
        for rr, cc in cells:
            cur[rr, cc] = col

    # reflect back — unless the drawn grid is already symmetric under that flip,
    # in which case the op would change nothing at all.
    if flip_op == 27 and not np.array_equal(cur, np.flipud(cur)):
        ops.append(27); sels.append(full)
        cur = np.flipud(cur)
    elif flip_op == 26 and not np.array_equal(cur, np.fliplr(cur)):
        ops.append(26); sels.append(full)
        cur = np.fliplr(cur)

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
    return ops, sels


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
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
