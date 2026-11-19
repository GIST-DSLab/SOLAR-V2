"""
ARC Task: a48eeaf7 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, dotc=2) -> dict:
    def unifint(lb, ub):
        a, b = lb, ub
        if b < a:
            b = a
        d = random.uniform(diff_lb, diff_ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    h = unifint(8, max(8, max_h))
    w = unifint(8, max(8, max_w))
    ih = unifint(2, h // 2)
    iw = unifint(2, w // 2)
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
            go[r][c] = sqc
    A = [(x, locj - 1) for x in range(loci, loci + ih)]
    Ap = [(x, random.randint(0, locj - 2)) for x in range(loci, loci + ih)]
    B = [(x, locj + iw) for x in range(loci, loci + ih)]
    Bp = [(x, random.randint(locj + iw + 1, w - 1)) for x in range(loci, loci + ih)]
    C = [(loci - 1, x) for x in range(locj, locj + iw)]
    Cp = [(random.randint(0, loci - 2), x) for x in range(locj, locj + iw)]
    D = [(loci + ih, x) for x in range(locj, locj + iw)]
    Dp = [(random.randint(loci + ih + 1, h - 1), x) for x in range(locj, locj + iw)]
    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    num = unifint(1, len(srarr))
    locs = set(random.sample(range(len(srarr)), num))
    for j in range(len(srarr)):
        if j in locs:
            s, d = srarr[j], dearr[j]
            gi[s[0]][s[1]] = dotc
            go[d[0]][d[1]] = dotc
    ncorn = unifint(0, 4)

    def diag(start, dr, dc):
        out = []
        r, c = start
        while 0 <= r < h and 0 <= c < w:
            out.append((r, c))
            r += dr
            c += dc
        return out

    specs = [((loci - 1, locj - 1), (loci - 2, locj - 2), (-1, -1)),
             ((loci - 1, locj + iw), (loci - 2, locj + iw + 1), (-1, 1)),
             ((loci + ih, locj - 1), (loci + ih + 1, locj - 2), (1, -1)),
             ((loci + ih, locj + iw), (loci + ih + 1, locj + iw + 1), (1, 1))]
    for k in range(ncorn):
        tgt, st, (dr, dc) = specs[k]
        go[tgt[0]][tgt[1]] = dotc
        r, c = random.choice(diag(st, dr, dc))
        gi[r][c] = dotc
    gi = np.array(gi)
    go = np.array(go)
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: four gravity sweeps.  Every dot falls straight toward the block
    and stops on the block's one-cell outer ring (a dot on a diagonal falls
    to the ring's corner, vertical leg first, then horizontal leg).  Instead
    of walking dot by dot, the dots are carried one DIRECTION at a time:
    all dots that still have to travel downward are grabbed together and
    pushed one cell, then the ones still short of the ring are grabbed again
    and pushed another cell, and so on; then the same for up, right, left.
    Each push leaves the cells the dots just left at 0, which are
    immediately repainted with the background.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    cells_of = {}
    for r in range(h):
        for c in range(w):
            cells_of.setdefault(int(I[r, c]), []).append((r, c))

    def is_rect(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    rect_cols = [c for c, cl in cells_of.items() if is_rect(cl)]
    sqc = max(rect_cols, key=lambda c: len(cells_of[c]))
    others = [c for c in cells_of if c != sqc]
    bgc = max(others, key=lambda c: len(cells_of[c]))
    dotc = max([c for c in others if c != bgc], key=lambda c: len(cells_of[c]))

    block = cells_of[sqc]
    r0 = min(p[0] for p in block); r1 = max(p[0] for p in block)
    c0 = min(p[1] for p in block); c1 = max(p[1] for p in block)

    # Where each dot comes to rest: clamp onto the outer ring along the
    # straight line (or the diagonal) toward the block.
    def rest(cell):
        sr, sc = cell
        tr = min(max(sr, r0 - 1), r1 + 1)
        tc = min(max(sc, c0 - 1), c1 + 1)
        return (tr, tc)

    dots = {p: rest(p) for p in cells_of[dotc]}
    ops, sels = [], []

    if dotc == 0:
        # A 0-coloured dot cannot ride a Move (only non-zero cells travel):
        # lift it with the background, set it down with Color(0).
        for (sr, sc), (tr, tc) in sorted(dots.items()):
            if (sr, sc) == (tr, tc):
                continue
            ops.append(int(bgc)); sels.append(sel_of([(sr, sc)]))
            ops.append(0); sels.append(sel_of([(tr, tc)]))
        ops.append(34); sels.append([0, 0, h - 1, w - 1])  # whole grid
        return ops, sels

    cur = {p: p for p in dots}   # original -> current position
    sweeps = [(21, 1, 0), (20, -1, 0), (22, 0, 1), (23, 0, -1)]  # D, U, R, L
    for op, dr, dc in sweeps:
        while True:
            active = []
            for p, (tr, tc) in dots.items():
                r, c = cur[p]
                if (dr and (tr - r) * dr > 0) or (dc and (tc - c) * dc > 0):
                    active.append(p)
            if not active:
                break
            before = [cur[p] for p in active]
            ops.append(op); sels.append(sel_of(before))   # grab the still-falling dots
            for p in active:
                r, c = cur[p]
                cur[p] = (r + dr, c + dc)
            after = set(cur[p] for p in active)
            vacated = sorted(set(before) - after)
            if bgc != 0 and vacated:
                ops.append(int(bgc)); sels.append(sel_of(vacated))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # whole grid, intentional
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
                        f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
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
                                f"for task a48eeaf7"
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
                    f"Failed to build a complete episode for task a48eeaf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a48eeaf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
