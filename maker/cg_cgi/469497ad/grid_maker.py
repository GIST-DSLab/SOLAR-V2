"""
ARC Task: 469497ad (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 2]
    bgc, sqc = random.sample(cols, 2)
    rem = [c for c in cols if c not in (bgc, sqc)]
    ccols = random.sample(rem, 6)
    return {"bgc": bgc, "sqc": sqc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return random.randint(int(round(lo)), int(round(hi)))
    if ccols is None:
        cols = [c for c in range(10) if c not in (2, bgc, sqc)]
        ccols = random.sample(cols, 6)
    h = min(unifint(diff_lb, diff_ub, (3, 6)), max_h)
    w = min(unifint(diff_lb, diff_ub, (3, 6)), max_w)
    gi = np.full((h, w), bgc, dtype=int)
    sqh = random.randint(1, h - 2)
    sqw = random.randint(1, w - 2)
    si = random.randint(0, h - sqh - 2)
    sj = random.randint(0, w - sqw - 2)
    gi[si:si + sqh, sj:sj + sqw] = sqc
    numcub = min(min(min(h, w) + 1, min(max_h, max_w) // max(h, w)), 7)
    numc = unifint(diff_lb, diff_ub, (2, max(2, numcub)))
    numaccc = numc - 1
    cc = list(ccols)[:numaccc]
    gi = np.rot90(gi, 2).copy()
    locs = random.sample(range(1, min(h, w)), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(cc, locs):
        gi[0, l:] = c
        gi[l:, 0] = c
    gi = np.rot90(gi, 2).copy()
    go = np.kron(gi, np.ones((numc, numc), dtype=int))
    rows, cs = np.where(go == sqc)
    r0, c0, r1, c1 = rows.min(), cs.min(), rows.max(), cs.max()
    H, W = go.shape
    out = go.copy()
    for (r, c), (dr, dc) in [((r1, c1), (1, 1)), ((r0, c0), (-1, -1)),
                             ((r0, c1), (-1, 1)), ((r1, c0), (1, -1))]:
        r += dr; c += dc
        while 0 <= r < H and 0 <= c < W:
            if go[r, c] == bgc:
                out[r, c] = 2
            r += dr; c += dc
    k = random.randint(0, 3)
    return {"input": np.rot90(gi, k).tolist(), "output": np.rot90(out, k).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    k = ho // hi   # upscale factor (= number of colours in I minus one)

    def ring_colors(r0, c0, r1, c1):
        out = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                    if 0 <= r < hi and 0 <= c < wi:
                        out.add(int(I[r, c]))
        return out

    # square = colour whose bbox is wrapped by ONE colour (that colour is the background)
    best = None
    for col in sorted(set(I.flatten().tolist())):
        cells = np.argwhere(I == col)
        r0, c0 = [int(v) for v in cells.min(0)]
        r1, c1 = [int(v) for v in cells.max(0)]
        ring = ring_colors(r0, c0, r1, c1)
        if len(ring) == 1:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area < best[0]:
                best = (area, (r0, c0, r1, c1), next(iter(ring)))
    _, (sr0, sc0, sr1, sc1), bgc = best

    ops, sels = [], []
    G = np.zeros((ho, wo), dtype=int)   # tracks the current working grid
    G[:hi, :wi] = I

    # step 1: grow the canvas to k-times size (whole rectangle)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # step 2: stretch row by row -- each input row becomes a k-tall band; each run
    # of equal colour in that row becomes one k-times wider stripe (full rectangle).
    # Skip a stripe only if it already holds that colour in the CURRENT grid.
    for r in range(hi):
        c = 0
        while c < wi:
            e = c
            while e + 1 < wi and I[r, e + 1] == I[r, c]:
                e += 1
            col = int(I[r, c])
            R0, C0, R1, C1 = r * k, c * k, (r + 1) * k - 1, (e + 1) * k - 1
            if (G[R0:R1 + 1, C0:C1 + 1] != col).any():
                ops.append(col); sels.append([R0, C0, R1 - R0, C1 - C0])
                G[R0:R1 + 1, C0:C1 + 1] = col
            c = e + 1

    # step 3: one diagonal ray per corner of the stretched square, marking 2 on
    # every background cell the ray crosses (one op per ray).
    rr0, cc0 = sr0 * k, sc0 * k
    rr1, cc1 = (sr1 + 1) * k - 1, (sc1 + 1) * k - 1
    rays = [((rr0, cc0), (-1, -1)), ((rr0, cc1), (-1, 1)),
            ((rr1, cc1), (1, 1)), ((rr1, cc0), (1, -1))]
    for (r, c), (dr, dc) in rays:
        cells = []
        r += dr; c += dc
        while 0 <= r < ho and 0 <= c < wo:
            if I[r // k, c // k] == bgc:
                cells.append((r, c))
            r += dr; c += dc
        if cells:
            ops.append(2); sels.append(sel_of(cells))
            for a, b in cells:
                G[a, b] = 2

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
                        f"num_examples+1 ({num_examples + 1}) for task 469497ad"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 469497ad"
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
                                f"for task 469497ad"
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
                    f"Failed to build a complete episode for task 469497ad "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"469497ad-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
