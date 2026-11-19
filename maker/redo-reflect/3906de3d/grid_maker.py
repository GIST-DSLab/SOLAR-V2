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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ---------------------------------------------------------------- colors ----
ROTS = [0, 1, 2, 3]   # discrete structural variant: global orientation of the scene


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # bgc and linc are kept non-zero: they are the two colours that live inside the
    # strips we reflect, and ARCLE's object ops treat 0 as "nothing there".
    bgc = random.choice([c for c in cols if c != 0])
    linc = random.choice([c for c in cols if c != 0 and c != bgc])
    boxc = random.choice([c for c in cols if c != bgc and c != linc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


# -------------------------------------------------------------- generator ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def unifint(bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    # a 90/270 rotation swaps the axes, so swap the caps before sampling
    hlim = max_w if rot % 2 == 1 else max_h
    wlim = max_h if rot % 2 == 1 else max_w
    hlim = max(5, min(30, int(hlim)))
    wlim = max(5, min(30, int(wlim)))

    h = unifint((5, hlim))
    w = unifint((5, wlim))
    oh = unifint((2, h // 2))
    ow = unifint((3, w - 2))
    locj = random.randint(1, w - ow - 1)

    gi = [[bgc] * w for _ in range(h)]
    cutoffs = {}
    for jj in range(locj, locj + ow):
        co = random.randint(1, oh - 1)
        cutoffs[jj] = co
        for r in range(co):
            gi[r][jj] = boxc

    go = [row[:] for row in gi]

    numlns = unifint((1, ow - 1))
    lnlocs = random.sample(list(range(locj, locj + ow)), numlns)
    for jj in lnlocs:
        co = cutoffs[jj]
        lineh = random.randint(1, h - co - 1)
        for r in range(h - lineh, h):          # input: line hangs off the far edge
            gi[r][jj] = linc
        for r in range(co, co + lineh):        # output: line hangs under the box
            go[r][jj] = linc

    A = np.rot90(np.array(gi, dtype=int), rot)
    B = np.rot90(np.array(go, dtype=int), rot)
    return {"input": A.tolist(), "output": B.tolist()}


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O):
    """
    Rule: the scene is a comb of teeth (the 'box') growing from one grid edge; each
    tooth's column may carry a bar stuck to the OPPOSITE edge.  In the output that
    column's free part (everything past the tooth) is REVERSED: the bar ends up
    hanging directly off the tooth and the empty run behind it.  So per affected
    column/row we simply reflect that one strip -- FlipV for vertical strips,
    FlipH for horizontal ones.  Nothing else changes.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- normalise orientation (teeth pointing "down" from the top row) --------
    transposed = len(set(I[0].tolist())) == 1          # teeth grow sideways
    J = I.T if transposed else I
    bgc = int(J[0, 0])
    flipped = int((J[0] == bgc).sum()) > int((J[-1] == bgc).sum())   # teeth at bottom
    K = J[::-1] if flipped else J
    hk, wk = K.shape

    def orig(r, c):
        rr = (hk - 1 - r) if flipped else r            # back into J coordinates
        return (c, rr) if transposed else (rr, c)      # back into I coordinates

    flip_op = 26 if transposed else 27                 # 26 = FlipH (left<->right)
                                                       # 27 = FlipV (up<->down)

    grid = I.copy()
    box_cols = [c for c in range(wk) if K[0, c] != bgc]

    for c in box_cols:
        boxc = int(K[0, c])
        co = 1
        while co < hk and int(K[co, c]) == boxc:       # first free cell past the tooth
            co += 1
        end_col = int(K[hk - 1, c])
        if end_col == bgc:                             # this tooth carries no bar
            continue
        r = hk - 1
        while r >= co and int(K[r, c]) == end_col:     # bar length
            r -= 1
        bar_top = r + 1
        if bar_top <= co:                              # already flush: flip would be a no-op
            continue

        strip = [orig(rr, c) for rr in range(co, hk)]  # the whole free strip of this column
        ops.append(flip_op)
        sels.append(sel_of(strip))

        vals = [int(grid[p]) for p in strip]
        for p, v in zip(strip, reversed(vals)):
            grid[p] = v

    # safety net: on well-formed instances this emits nothing
    if grid.shape == O.shape and not np.array_equal(grid, O):
        diff = [(r, c) for r in range(ho) for c in range(wo) if grid[r, c] != O[r, c]]
        for col in sorted({int(O[r, c]) for (r, c) in diff}):
            cells = [(r, c) for (r, c) in diff if int(O[r, c]) == col]
            ops.append(col)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])                # bbox == whole grid, exactly intended
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
