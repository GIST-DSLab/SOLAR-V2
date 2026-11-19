"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    variants = [{"mirror": False}, {"mirror": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, mirror=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)
    if mirror is None:
        mirror = random.choice([True, False])
    lim = min(max_h, max_w, 30)
    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc, sqc)]
    nbars = unifint(diff_lb, diff_ub, (3, min(8, (lim) // 3)))
    ccols = random.sample(remcols, nbars)
    w = unifint(diff_lb, diff_ub, (nbars + 3, lim))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, lim - nbars))
    go = [[col] * nbars for col in ccols]
    gi = [[col] * w for col in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(gi) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    h2 = nbars + hmarg
    n = nbars
    loci = random.randint(1, h2 - n - 2)
    locj = random.randint(1, w - n - 2)
    gi = np.array(gi, dtype=int)
    gi[loci - 1:loci + n + 1, locj - 1:locj + n + 1] = bgc
    gi[loci:loci + n, locj:locj + n] = sqc
    go = np.array(go, dtype=int)
    if mirror:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (the golf entry's): the far edge line across the bars -- the last column
    when the bars are rows, the last row when they are columns -- is never touched by
    the square or its margin, so it lists every bar, in order, as one cell each with
    background gaps in between.  The answer is that list with the gaps squeezed out,
    repeated as many times as there are bars.
    Ops: slide each bar's edge cell along the edge line into slot k (one grab, then
    empty Moves); crop to the n x n block ending at that edge; copy the squeezed
    line and paste it into every other line of the block.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vertical = len(set(I[hi - 1, :].tolist())) > len(set(I[:, wi - 1].tolist()))
    G = I.T.copy() if vertical else I.copy()      # work as if bars were rows
    H, W = G.shape
    edge = G[:, W - 1].tolist()                    # last column: always intact
    vals, cnts = np.unique(edge, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])               # gaps dominate the edge line
    bars = [(r, int(v)) for r, v in enumerate(edge) if v != bgc]
    n = len(bars)

    def P(r, c):                                   # canonical -> real coordinates
        return (c, r) if vertical else (r, c)

    MV = 23 if vertical else 20                    # toward slot 0: left / up
    ops, sels = [], []
    cur = G.copy()

    # 1. squeeze: slide each bar's edge cell into its slot, in bar order
    for k, (r, col) in enumerate(bars):
        if r == k or col == 0:                     # in place / colour-0 cannot be grabbed
            continue
        ops.append(MV); sels.append(sel_of([P(r, W - 1)]))
        for _ in range(r - k - 1):
            ops.append(MV); sels.append(sel_of([]))
        cur[r, W - 1] = 0
        cur[k, W - 1] = col

    # 2. crop to the n x n block whose last line is the squeezed edge (full rectangle)
    r0, c0 = P(0, W - n)
    ops.append(33); sels.append([r0, c0, n - 1, n - 1])
    blk = cur[0:n, W - n:W].copy()

    # 3. a colour-0 bar: paint its whole slot line 0 (Paste never writes 0)
    for k, (r, col) in enumerate(bars):
        if col == 0 and np.any(blk[k, :] != 0):
            ops.append(0); sels.append(sel_of([P(k, j) for j in range(n)]))
            blk[k, :] = 0

    # 4. copy the squeezed line and paste it over every other line of the block
    line = blk[:, n - 1].copy()
    pasted = False
    for j in range(n - 1):
        cand = blk[:, j].copy()
        nz = line != 0
        cand[nz] = line[nz]
        if np.array_equal(cand, blk[:, j]):
            continue
        if not pasted:
            a, b = P(0, n - 1)
            # the whole squeezed line (full rectangle n x 1 / 1 x n)
            ops.append(29); sels.append([a, b, (n - 1) if not vertical else 0,
                                         0 if not vertical else (n - 1)])
            pasted = True
        a, b = P(0, j)
        ops.append(30); sels.append([a, b, 0, 0])
        blk[:, j] = cand

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
