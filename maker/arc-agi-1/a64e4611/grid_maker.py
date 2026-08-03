"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    h = unifint(diff_lb, diff_ub, (min(18, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(18, max_w), max_w))

    # --- noise canvas -------------------------------------------------------
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = [[noisec] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    for i, j in random.sample(inds, nbgc):
        gi[i][j] = bgc
    addn, addb = set(), set()
    for i in range(h - 2):
        for j in range(w - 2):
            blk = [gi[i + di][j + dj] for di in range(3) for dj in range(3)]
            if all(v == bgc for v in blk):
                addn.add((i + random.randint(0, 2), j + random.randint(0, 2)))
            elif all(v == noisec for v in blk):
                addb.add((i + random.randint(0, 2), j + random.randint(0, 2)))
    for i, j in addn:
        gi[i][j] = noisec
    for i, j in addb:
        gi[i][j] = bgc
    go = [row[:] for row in gi]

    # --- vertical bar -------------------------------------------------------
    m = min(h, w)
    marg = 6                                   # keeps every arm long enough to be readable
    dim_ub = max(3, min(8, m - 2 * marg - 1))
    dim = random.randint(random.randint(3, dim_ub), dim_ub)
    locj = random.randint(marg, max(marg, m - dim - marg - 1))
    spi = random.choice((0, random.randint(3, h // 2)))
    for j in range(locj, locj + dim):
        for r in range(spi, h):
            gi[r][j] = bgc
            go[r][j] = bgc
    r0 = spi + 1 if spi > 0 else spi
    for j in range(locj + 1, locj + dim - 1):
        for r in range(r0, h):
            go[r][j] = 3

    # --- horizontal arms ----------------------------------------------------
    sgns = random.choice(((-1,), (1,), (-1, 1)))
    blocks = []
    sl = random.choice((spi, random.randint(spi + 3, h - 6)))
    blocks.append((sgns, sl, random.randint(3, min(8, h - sl - 3))))
    if len(sgns) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
        sl2 = random.choice((spi, random.randint(spi + 3, h - 6)))
        blocks.append(((-sgns[0],), sl2, random.randint(3, min(8, h - sl2 - 3))))
    for sgs, a, hh in blocks:
        for sgn in sgs:
            for r in range(a, a + hh):
                cs = range(0, locj + 1) if sgn == -1 else range(locj, w)
                for c in cs:
                    gi[r][c] = bgc
                    if go[r][c] != 3:
                        go[r][c] = bgc
        for sgn in sgs:
            for r in range(a + 1 if a > 0 else a, a + hh - 1):
                cs = range(0, locj + dim - 1) if sgn == -1 else range(locj + 1, w)
                for c in cs:
                    go[r][c] = 3

    # --- keep the bar/arm borders unambiguous against random noise ----------
    def mark(r, c):
        if 0 <= r < h and 0 <= c < w:
            gi[r][c] = noisec
            go[r][c] = noisec

    if spi > 0:
        mark(spi - 1, locj)
    mark(h - 1, locj - 1)
    mark(h - 1, locj + dim)
    for sgs, a, hh in blocks:
        for sgn in sgs:
            c = 0 if sgn == -1 else w - 1
            mark(a - 1, c)
            mark(a + hh, c)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    def runs(seq):
        out = []
        for v in seq:
            if out and out[-1][-1] == v - 1:
                out[-1].append(v)
            else:
                out.append([v])
        return out

    # 1. the vertical bar: the only block of >=3 neighbouring columns that is one
    #    solid colour from mid-grid down to the bottom edge.  Its colour is the
    #    bar colour, its extent gives the bar's left edge and width.
    bar = None
    for c in np.unique(I):
        solid = [j for j in range(w) if bool(np.all(I[h // 2:, j] == c))]
        for run in runs(solid):
            if len(run) >= 3 and (bar is None or len(run) > len(bar[1])):
                bar = (int(c), run)
    bgc, cols = bar
    locj, dim = cols[0], len(cols)

    # 2. the bar's top edge
    top = h - 1
    while top > 0 and bool(np.all(I[top - 1, locj:locj + dim] == bgc)):
        top -= 1

    # 3. the horizontal arms: rows that are bar-coloured from the bar out to a
    #    side edge, grouped into contiguous arms
    def arms(mask_of_row):
        rows = [r for r in range(top, h) if bool(np.all(mask_of_row(r)))]
        return [(rr[0], rr[-1]) for rr in runs(rows) if len(rr) >= 3]

    left = arms(lambda r: I[r, :locj] == bgc)
    right = arms(lambda r: I[r, locj + dim:] == bgc)

    ops, sels = [], []

    def paint(r0, c0, r1, c1):
        ops.append(3)
        sels.append([r0, c0, r1 - r0, c1 - c0])

    # the interior of each bar gets colour 3: inset one cell from every wall the
    # bar has, but not where the bar runs off the grid edge.
    # vertical bar first (it is the anchor the arms grow out of)
    paint(top + 1 if top > 0 else 0, locj + 1, h - 1, locj + dim - 2)
    # then each arm, from the bar outward to its own edge
    for a, b in left:
        paint(a + 1 if a > 0 else a, 0, b - 1, locj)
    for a, b in right:
        paint(a + 1 if a > 0 else a, locj + dim - 1, b - 1, w - 1)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
