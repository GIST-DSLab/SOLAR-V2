"""
ARC Task: 3906de3d (RE-ARC) — LLM-generated grid_maker
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = int(round(a + (b - a) * diff_lb))
    hi = int(round(a + (b - a) * diff_ub))
    if lo > hi:
        lo, hi = hi, lo
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    return random.randint(lo, hi)


def _rot90ccw(g):
    h = len(g)
    w = len(g[0])
    return [[g[r][w - 1 - c] for r in range(h)] for c in range(w)]


# ── 1. sample_colors ─────────────────────────────────────────────────────────

ROTS = [0, 1, 2, 3]          # the generator's rotf: identity / 90 / 180 / 270


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, linc = random.sample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


# ── 2. generate ──────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc,
             rot=None, instance_plan=None) -> dict:
    if rot is None and instance_plan:      # caller that forwards the plan wholesale
        rot = random.choice([p["rot"] for p in instance_plan])
    if rot is None:
        rot = random.choice(ROTS)
    mh = max(5, min(30, int(max_h)))
    mw = max(5, min(30, int(max_w)))
    if rot % 2 == 1:                       # odd rotations swap the axes
        h_ub, w_ub = mw, mh
    else:
        h_ub, w_ub = mh, mw

    h = _unifint(diff_lb, diff_ub, (5, h_ub))
    w = _unifint(diff_lb, diff_ub, (5, w_ub))
    oh = _unifint(diff_lb, diff_ub, (2, h // 2))
    ow = _unifint(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)

    gi = [[bgc] * w for _ in range(h)]
    rng = list(range(locj, locj + ow))
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):                    # the comb, flush with row 0
        for r in range(co):
            gi[r][jj] = boxc

    numlns = _unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = set(random.sample(rng, numlns))
    go = [row[:] for row in gi]
    for jj, co in zip(rng, cutoffs):                    # loose bars, flush with row h-1
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            for r in range(h - lineh, h):
                gi[r][jj] = linc
            for r in range(co, co + lineh):             # ... slid up onto the comb
                go[r][jj] = linc

    for _ in range(rot):
        gi = _rot90ccw(gi)
        go = _rot90ccw(go)
    return {"input": gi, "output": go}


# ── 3. derive_operations ─────────────────────────────────────────────────────

MOVE_OF_K = {0: 20, 1: 22, 2: 21, 3: 23}   # canonical "up" expressed in I's frame


def _analyse(I):
    """Find (k, boxc, linc) such that np.rot90(I, k) shows the comb flush against
    the top edge and the loose bars flush against the bottom edge."""
    hi, wi = I.shape
    bgc = int(I[0, 0])                      # all four corners are always background
    cands = sorted(set(I.flatten().tolist()) - {bgc})
    idx = np.arange(hi * wi).reshape(hi, wi)
    best = None
    for k in range(4):
        J = np.rot90(I, k)
        M = np.rot90(idx, k)                # canonical (r,c) -> original flat index
        hc, wc = J.shape
        for bc in cands:
            for lc in cands:
                if lc == bc:
                    continue
                boxcols = sorted({c for c in range(wc) if (J[:, c] == bc).any()})
                lincols = sorted({c for c in range(wc) if (J[:, c] == lc).any()})
                if not boxcols or not lincols:
                    continue
                if boxcols != list(range(boxcols[0], boxcols[-1] + 1)):
                    continue                # comb spans a contiguous band of columns
                if not set(lincols) <= set(boxcols):
                    continue                # every bar hangs under a comb tooth
                ok = True
                for c in boxcols:           # comb teeth: contiguous from the top edge
                    rows = [r for r in range(hc) if J[r, c] == bc]
                    if not rows or rows != list(range(0, len(rows))):
                        ok = False
                        break
                if not ok:
                    continue
                for c in lincols:           # bars: contiguous, flush with bottom edge
                    rows = [r for r in range(hc) if J[r, c] == lc]
                    if rows != list(range(hc - len(rows), hc)):
                        ok = False
                        break
                    if rows[0] <= int((J[:, c] == bc).sum()):
                        ok = False          # must have room to slide
                        break
                if ok:
                    cand = (len(lincols), k, bc, lc, J, M)
                    if best is None or cand[0] < best[0]:
                        best = cand
    if best is None:
        return None
    _, k, bc, lc, J, M = best
    return k, bc, lc, J, M, bgc


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    info = _analyse(I)
    if info is None:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    k, boxc, linc, J, M, bgc = info
    hc, wc = J.shape
    move_op = MOVE_OF_K[k]
    dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[move_op]

    # ARCLE grabs `grid * selection`, so a bar painted with colour 0 would be an
    # empty object and the Move would delete it. Carry it under a spare colour.
    temp = None
    if linc == 0:
        temp = next(c for c in range(1, 10) if c not in (bgc, boxc, linc))

    for c in range(wc):                      # one bar at a time, along the comb
        rows = [r for r in range(hc) if J[r, c] == linc]
        if not rows:
            continue
        bl = int((J[:, c] == boxc).sum())    # length of this comb tooth
        shift = rows[0] - bl                 # slide until the bar meets the tooth
        if shift <= 0:
            continue
        cells = [tuple(divmod(int(M[r, c]), wi)) for r in rows]
        final = [(r + dr * shift, c2 + dc * shift) for r, c2 in cells]

        if temp is not None:
            ops.append(temp)                 # make the bar carriable
            sels.append(sel_of(cells))

        ops.append(move_op)                  # grab the bar ...
        sels.append(sel_of(cells))
        for _ in range(shift - 1):           # ... and keep gliding it (empty sel)
            ops.append(move_op)
            sels.append(sel_of([]))

        hole = sorted(set(cells) - set(final))
        if bgc != 0 and hole:                # only the vacated footprint reads 0
            ops.append(bgc)
            sels.append(sel_of(hole))

        if temp is not None:
            ops.append(linc)                 # restore the bar's own colour
            sels.append(sel_of(final))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
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
                                f"for task 3906de3d"
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
                    f"Failed to build a complete episode for task 3906de3d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3906de3d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
