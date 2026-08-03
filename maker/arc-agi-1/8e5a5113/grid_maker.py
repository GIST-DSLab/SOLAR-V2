"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers
def _rot90cw(g):
    """DSL rot90 == clockwise == np.rot90(k=3)."""
    return [list(r) for r in zip(*g[::-1])]


# ---------------------------------------------------------------- 1. colors
VARIANTS = [{"portrait_flag": False}, {"portrait_flag": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barc = random.choice(rem)
    rem = [c for c in rem if c != barc]
    patcols = random.sample(rem, random.randint(1, min(8, len(rem))))

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "patcols": patcols, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, patcols,
             portrait_flag=None) -> dict:
    if portrait_flag is None:
        portrait_flag = random.choice((True, False))

    # grid is built landscape first, then optionally rotated -> swap limits
    if portrait_flag:
        Hlim, Wlim = max_w, max_h
    else:
        Hlim, Wlim = max_h, max_w

    dmax = min(9, Hlim, (Wlim - 1) // 2)
    dmax = max(2, dmax)
    d = unifint(diff_lb, diff_ub, (2, dmax))

    k = 4 if d < 7 else 3
    nmax = min(k, (Wlim + 1) // (d + 1))
    nmax = max(2, nmax)
    num = unifint(diff_lb, diff_ub, (2, nmax))

    ncols = unifint(diff_lb, diff_ub, (1, min(8, len(patcols))))
    pcols = random.sample(list(patcols), ncols)

    # pattern panel
    c = [[bgc for _ in range(d)] for _ in range(d)]
    inds = [(i, j) for i in range(d) for j in range(d)]
    ncells = unifint(diff_lb, diff_ub, (1, d * d - 1))
    for (i, j) in random.sample(inds, ncells):
        c[i][j] = random.choice(pcols)

    bgpanel = [[bgc for _ in range(d)] for _ in range(d)]
    fillinidx = random.randint(0, num - 1)

    ipanels, opanels = [], []
    cur = [row[:] for row in c]
    for j in range(num):
        if j > 0:
            cur = _rot90cw(cur)
        opanels.append([row[:] for row in cur])
        ipanels.append([row[:] for row in cur] if j == fillinidx
                       else [row[:] for row in bgpanel])

    def _assemble(panels):
        g = [[] for _ in range(d)]
        for j, p in enumerate(panels):
            if j > 0:
                for r in range(d):
                    g[r].append(barc)
            for r in range(d):
                g[r].extend(p[r])
        return g

    gi = _assemble(ipanels)
    go = _assemble(opanels)

    if portrait_flag:
        gi = _rot90cw(gi)
        go = _rot90cw(go)

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ---------------------------------------------------------------- 3. ops
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # panel geometry: d x d panels separated by 1-cell bar lines
    vertical = hi > wi
    if vertical:
        d = wi
        num = (hi + 1) // (d + 1)
        origins = [(j * (d + 1), 0) for j in range(num)]
    else:
        d = hi
        num = (wi + 1) // (d + 1)
        origins = [(0, j * (d + 1)) for j in range(num)]

    panels = [I[r:r + d, c:c + d] for (r, c) in origins]
    # the only non-uniform panel is the pattern source; the rest are pure bgc
    src = 0
    bgc = 0
    for j, p in enumerate(panels):
        if len(set(p.flatten().tolist())) > 1:
            src = j
        else:
            bgc = int(p[0, 0])

    ops, sels = [], []
    sr, sc = origins[src]
    src_panel = I[sr:sr + d, sc:sc + d]
    has_zero = bool((src_panel == 0).any())

    # copy the pattern panel from INPUT (full d x d rectangle)
    ops.append(28); sels.append([sr, sc, d - 1, d - 1])

    for t in range(num):
        if t == src:
            continue
        tr, tc = origins[t]
        # Paste is transparent to 0; if the pattern owns 0-cells and bgc != 0,
        # blank the panel first so those cells end up 0 as required.
        if has_zero and bgc != 0:
            ops.append(0); sels.append([tr, tc, d - 1, d - 1])
        ops.append(30); sels.append([tr, tc, 0, 0])
        # each panel is the previous one rotated clockwise
        kk = (t - src) % 4
        if kk == 3:
            ops.append(24); sels.append([tr, tc, d - 1, d - 1])  # one CCW
        else:
            for _ in range(kk):
                ops.append(25); sels.append([tr, tc, d - 1, d - 1])  # CW

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
