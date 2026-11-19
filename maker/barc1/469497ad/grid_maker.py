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
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))
    cols = [c for c in range(10) if c != 2]
    h = unifint(diff_lb, diff_ub, (3, min(6, max_h)))
    w = unifint(diff_lb, diff_ub, (3, min(6, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    sqh = random.randint(1, h - 2)
    sqw = random.randint(1, w - 2)
    si = random.randint(0, h - sqh - 2)
    sj = random.randint(0, w - sqw - 2)
    gi[si:si + sqh, sj:sj + sqw] = sqc
    numcub = min(min(h, w) + 1, max_h // h, max_w // w, 7)
    if numcub < 2:
        raise ValueError("grid too small")
    numc = unifint(diff_lb, diff_ub, (2, numcub))
    remcols = [c for c in cols if c not in (bgc, sqc)]
    ccols = random.sample(remcols, numc - 1)
    gi = np.rot90(gi, 2).copy()
    locs = random.sample(range(1, min(h, w)), numc - 2)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi[0, l:] = c
        gi[l:, 0] = c
    gi = np.rot90(gi, 2).copy()
    go = np.kron(gi, np.ones((numc, numc), dtype=int))
    rs, cs = np.where(go == sqc)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    H, W = go.shape
    for (r, c), (dr, dc) in [((r1, c1), (1, 1)), ((r0, c0), (-1, -1)),
                             ((r0, c1), (-1, 1)), ((r1, c0), (1, -1))]:
        r += dr; c += dc
        while 0 <= r < H and 0 <= c < W:
            if go[r, c] == bgc:
                go[r, c] = 2
            r += dr; c += dc
    k = random.randint(0, 3)
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    palette = sorted(set(I.flatten().tolist()))
    k = len(palette) - 1                      # magnification = number of non-background colours
    ho, wo = hi * k, wi * k

    def ring_colors(r0, c0, r1, c1):
        out = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if (r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)) and 0 <= r < hi and 0 <= c < wi:
                    out.add(int(I[r, c]))
        return out

    # the square: a colour whose bbox is wrapped by ONE uniform colour (= background)
    best = None
    for col in palette:
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
    grid = np.zeros((ho, wo), dtype=int)     # tracks the visible working grid
    grid[:hi, :wi] = I

    # 1. enlarge canvas to the magnified size (full rectangle)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. magnify every input pixel into its k x k block (full rectangle per pixel)
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            blk = grid[r * k:(r + 1) * k, c * k:(c + 1) * k]
            if (blk == v).all():
                continue                      # block already holds this colour
            ops.append(v); sels.append([r * k, c * k, k - 1, k - 1])
            blk[:] = v

    # 3. red diagonal line from just outside each corner of the magnified square,
    #    drawn outward until it meets a non-background colour or the edge
    rr0, cc0 = sr0 * k, sc0 * k
    rr1, cc1 = (sr1 + 1) * k - 1, (sc1 + 1) * k - 1
    rays = [((rr0, cc0), (-1, -1)), ((rr1, cc1), (1, 1)),
            ((rr1, cc0), (1, -1)), ((rr0, cc1), (-1, 1))]
    for (r, c), (dr, dc) in rays:
        line = []
        r += dr; c += dc
        while 0 <= r < ho and 0 <= c < wo and grid[r, c] == bgc:
            line.append((r, c))
            r += dr; c += dc
        if line:
            ops.append(2); sels.append(sel_of(line))
            for a, b in line:
                grid[a, b] = 2

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
