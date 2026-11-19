"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of
from utils import unifint, choice, randint
from dsl import (interval, box, backdrop, inbox, canvas, fill, connect, totuple,
                 identity, rot90, rot180, rot270)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxc = random.choice([c for c in cols if c not in (bgc, 0)])
    inc = random.choice([c for c in cols if c not in (bgc, boxc)])
    dotc = random.choice([c for c in cols if c not in (bgc, boxc, inc)])
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, boxc=1, inc=2, dotc=3, **kw) -> dict:
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    hi = unifint(diff_lb, diff_ub, (4, h - 1))
    wi = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    bx = box(frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)}))
    ins = backdrop(inbox(bx))
    c = canvas(bgc, (h, w))
    go = fill(c, boxc, bx)
    go = fill(go, inc, ins)
    cutoff = randint(loci + 2, loci + hi - 2)
    bx2 = box(frozenset({(loci, locj), (cutoff, locj + wi - 1)}))
    ins2 = backdrop(inbox(bx2))
    gi = fill(c, boxc, bx2)
    gi = fill(gi, inc, ins2)
    locc = choice(totuple(connect((loci + hi - 1, locj), (loci + hi - 1, locj + wi - 1))))
    gi = fill(gi, dotc, {locc})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _scene(I):
    """Find the hollow box (outline colour + uniform inside) and the lone dot."""
    for cand in np.unique(I).tolist():
        cells = np.argwhere(I == cand)
        a0, b0 = int(cells[:, 0].min()), int(cells[:, 1].min())
        a1, b1 = int(cells[:, 0].max()), int(cells[:, 1].max())
        bh, bw = a1 - a0 + 1, b1 - b0 + 1
        if bh < 3 or bw < 3 or len(cells) != 2 * bh + 2 * bw - 4:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        if not (np.all(sub[0, :] == cand) and np.all(sub[-1, :] == cand)
                and np.all(sub[:, 0] == cand) and np.all(sub[:, -1] == cand)):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or len(np.unique(inner)) != 1 or int(inner[0, 0]) == cand:
            continue
        cinc = int(inner[0, 0])
        rest = [x for x in np.unique(I).tolist() if x not in (cand, cinc)]
        singles = [x for x in rest if int((I == x).sum()) == 1]
        if len(rest) != 2 or len(singles) != 1:
            continue
        bgc = [x for x in rest if x != singles[0]][0]
        d = np.argwhere(I == singles[0])[0]
        return cand, cinc, bgc, (a0, b0, a1, b1), (int(d[0]), int(d[1]))
    raise ValueError("no box found")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    boxc, inc, bgc, (r0, c0, r1, c1), (rd, cd) = _scene(I)
    ops, sels = [], []

    # Reading: the wall facing the dot is one unit; it slides out until it lies on
    # the dot's line, and the box body is stretched behind it (the vacated lane
    # takes the interior colour, the two side walls extend along it).
    if cd > c1:        # wall = right column, slide right
        wall = [(r, c1) for r in range(r0, r1 + 1)]; op, dr, dc, n = 22, 0, 1, cd - c1
        gap_in = [(r, c) for r in range(r0 + 1, r1) for c in range(c1, cd)]
        gap_side = [(r0, c) for c in range(c1, cd)] + [(r1, c) for c in range(c1, cd)]
    elif cd < c0:      # left column, slide left
        wall = [(r, c0) for r in range(r0, r1 + 1)]; op, dr, dc, n = 23, 0, -1, c0 - cd
        gap_in = [(r, c) for r in range(r0 + 1, r1) for c in range(cd + 1, c0 + 1)]
        gap_side = [(r0, c) for c in range(cd + 1, c0 + 1)] + [(r1, c) for c in range(cd + 1, c0 + 1)]
    elif rd > r1:      # bottom row, slide down
        wall = [(r1, c) for c in range(c0, c1 + 1)]; op, dr, dc, n = 21, 1, 0, rd - r1
        gap_in = [(r, c) for r in range(r1, rd) for c in range(c0 + 1, c1)]
        gap_side = [(r, c0) for r in range(r1, rd)] + [(r, c1) for r in range(r1, rd)]
    else:              # top row, slide up
        wall = [(r0, c) for c in range(c0, c1 + 1)]; op, dr, dc, n = 20, -1, 0, r0 - rd
        gap_in = [(r, c) for r in range(rd + 1, r0 + 1) for c in range(c0 + 1, c1)]
        gap_side = [(r, c0) for r in range(rd + 1, r0 + 1)] + [(r, c1) for r in range(rd + 1, r0 + 1)]

    G = I.copy()
    if boxc != 0:
        # grab the wall once, then keep sliding it with empty selections
        ops.append(op); sels.append(sel_of(wall))
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))
        for r, c in wall:
            G[r, c] = 0                       # the grabbed footprint is left at 0
        for r, c in wall:
            G[r + dr * n, c + dc * n] = boxc
    else:
        # a colour-0 wall cannot be grabbed by Move: redraw it on the dot's line
        dest = [(r + dr * n, c + dc * n) for r, c in wall]
        ops.append(0); sels.append(sel_of(dest))
        for r, c in dest:
            G[r, c] = 0

    # the lane the wall left behind: body interior, then the two side walls
    # (skip only cells that already hold the colour being painted)
    for col, cells in ((inc, gap_in), (boxc, gap_side)):
        todo = [(r, c) for r, c in cells if G[r, c] != col]
        if todo:
            ops.append(int(col)); sels.append(sel_of(todo))
            for r, c in todo:
                G[r, c] = col

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
