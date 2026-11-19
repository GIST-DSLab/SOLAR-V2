"""
ARC Task: 8f2ea7aa (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(a, hi_i)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    remcols = [c for c in range(10) if c != bgc]
    dmax = 5
    while dmax > 2 and dmax * dmax > min(max_h, max_w):
        dmax -= 1
    d = _unifint(diff_lb, diff_ub, (2, dmax))
    d2 = d * d
    gi = np.full((d2, d2), bgc, dtype=int)
    go = np.full((d2, d2), bgc, dtype=int)
    minig = np.full((d, d), bgc, dtype=int)
    inds = [(i, j) for i in range(d) for j in range(d)]
    mp = d2 // 2
    dev = _unifint(diff_lb, diff_ub, (0, mp))
    devs = random.choice((1, -1))
    num = mp + devs * dev
    num = max(min(num, d2), 0)
    locs = set(random.sample(inds, num))

    def shp(s):
        if not s:
            return (0, 0)
        rs = [p[0] for p in s]
        cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(locs) != (d, d):
        locs.add(random.choice(sorted(set(inds) - locs)))
    ncols = _unifint(diff_lb, diff_ub, (1, 9))
    cols = random.sample(remcols, ncols)
    for (i, j) in locs:
        minig[i, j] = random.choice(cols)
    itv = list(range(0, d2, d))
    pr, pc = random.choice(itv), random.choice(itv)
    gi[pr:pr + d, pc:pc + d] = minig
    for (i, j) in locs:
        go[i * d:i * d + d, j * d:j * d + d] = minig
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[counts.argmax()])
    nz = np.argwhere(I != bgc)
    r0 = int(nz[:, 0].min())
    r1 = int(nz[:, 0].max())
    c0 = int(nz[:, 1].min())
    c1 = int(nz[:, 1].max())
    d = int(max(r1 - r0 + 1, c1 - c0 + 1))
    pattern = I[r0:r0 + d, c0:c0 + d]
    ops = []
    sels = []

    non_bgc_cells = [(i, j) for i in range(d) for j in range(d)
                     if int(pattern[i, j]) != bgc]
    # Where the pattern already sits in I; if that block is itself a
    # destination (its pattern cell is non-bgc) the source copy is kept as-is.
    src_tile = (r0 // d, c0 // d)
    src_is_dest = (r0 % d == 0 and c0 % d == 0 and src_tile in non_bgc_cells)
    has_nonzero = any(int(pattern[i, j]) != 0 for (i, j) in non_bgc_cells)

    # 1. Copy pattern from input (only if it has anything Paste can carry)
    if has_nonzero:
        ops.append(28)
        sels.append([r0, c0, d - 1, d - 1])
    # 2. Fill canvas with bgc — needed only to erase the source when the
    #    source block is not one of the destinations
    if not src_is_dest:
        ops.append(bgc)
        sels.append([0, 0, ho - 1, wo - 1])
    # 3. Paste pattern at each destination tile (skip the one already holding it)
    targets = [(i, j) for (i, j) in non_bgc_cells
               if not (src_is_dest and (i, j) == src_tile)]
    if has_nonzero:
        for (i, j) in targets:
            ops.append(30)
            sels.append([i * d, j * d, 0, 0])
    # 4. 0-cells don't travel with Paste: paint them per replicated tile
    if bgc != 0:
        zero_cells = [(ii, jj) for ii in range(d) for jj in range(d)
                      if int(pattern[ii, jj]) == 0]
        if zero_cells:
            for (i, j) in targets:
                ops.append(0)
                sels.append(sel_of([(i * d + ii, j * d + jj) for (ii, jj) in zero_cells]))
    # 5. Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
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
                                f"for task 8f2ea7aa"
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
                    f"Failed to build a complete episode for task 8f2ea7aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8f2ea7aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
