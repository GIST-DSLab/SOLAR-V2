"""
ARC Task: 6d0160f0 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    hmax = 2
    while (hmax + 1) * (hmax + 1) + hmax <= max_h and hmax < 5:
        hmax += 1
    wmax = 2
    while (wmax + 1) * (wmax + 1) + wmax <= max_w and wmax < 5:
        wmax += 1
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a * (h + 1), b * (w + 1)) for a, b in denseinds}
    locs = sample(totuple(sparseinds), noccs)
    trgtl = choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
    place = choice(totuple(denseinds))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = sample(candss, ntrgt)
    trgrem = {(choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    for rl in remlocs:
        tlo = choice(totuple(ccols))
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = sample(totuple(denseinds), ncells)
        obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        gi = paint(gi, shift(obj, rl))
    return {"input": gi, "output": go}


def _cell_size(n):
    # full = k*k + k - 1 for k in 2..5
    for k in range(2, 6):
        if k * k + k - 1 == n:
            return k
    for k in range(1, n + 1):
        if k * k + k - 1 >= n:
            return k
    return 1


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    h = _cell_size(H)
    w = _cell_size(W)
    line_rows = set(range(h, H, h + 1))
    line_cols = set(range(w, W, w + 1))
    linc = int(I[h, 0]) if h < H else int(I[0, w])

    def interior(G):
        return [int(G[r, c]) for r in range(G.shape[0]) for c in range(G.shape[1])
                if r not in line_rows and c not in line_cols]

    # background: what fills the compartments of the example outputs
    bgc = None
    if examples:
        cnt = Counter()
        for _, eo in examples:
            cnt.update(interior(np.asarray(eo, dtype=int)))
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None:
        bgc = Counter(interior(I)).most_common(1)[0][0]

    # compartment containing the 4 marker
    fours = [(r, c) for r in range(H) for c in range(W)
             if I[r, c] == 4 and r not in line_rows and c not in line_cols]
    fr, fc = fours[0]
    ti, tj = fr // (h + 1), fc // (w + 1)
    sr, sc = ti * (h + 1), tj * (w + 1)
    pi, pj = fr - sr, fc - sc          # position of 4 inside its compartment
    dr, dc = pi * (h + 1) - sr, pj * (w + 1) - sc

    ops, sels = [], []
    # 1. clear contents of every other compartment
    others = [(r, c) for r in range(H) for c in range(W)
              if r not in line_rows and c not in line_cols
              and not (r // (h + 1) == ti and c // (w + 1) == tj)
              and I[r, c] != bgc]
    if others:
        ops.append(bgc); sels.append(sel_of(others))
    # 2. line-coloured cells in the target compartment are dropped
    tcells = [(r, c) for r in range(sr, sr + h) for c in range(sc, sc + w)]
    linecells = [(r, c) for r, c in tcells if I[r, c] == linc and I[r, c] != bgc]
    if linecells:
        ops.append(bgc); sels.append(sel_of(linecells))
    obj = [(r, c) for r, c in tcells if I[r, c] != bgc and I[r, c] != linc]
    # 3. move the target pattern to the compartment indicated by the 4
    if dr != 0 or dc != 0:
        steps = [(20 if dr < 0 else 21, -1 if dr < 0 else 1, 0)] * abs(dr) + \
                [(23 if dc < 0 else 22, 0, -1 if dc < 0 else 1)] * abs(dc)
        cur = list(obj)
        first = True
        for op, a, b in steps:
            ops.append(op); sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + a, c + b) for r, c in cur]
        hole = sorted(set(obj) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))
        # zero-coloured pattern cells are transparent to Move: paint them at destination
        zeros = [(r + dr, c + dc) for r, c in obj if I[r, c] == 0]
        if zeros:
            ops.append(0); sels.append(sel_of(zeros))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
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
                                f"for task 6d0160f0"
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
                    f"Failed to build a complete episode for task 6d0160f0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d0160f0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
