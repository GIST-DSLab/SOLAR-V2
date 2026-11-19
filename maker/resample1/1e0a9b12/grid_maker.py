"""
ARC Task: 1e0a9b12 (RE-ARC) — LLM-generated grid_maker
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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
        w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
        nc = unifint(diff_lb, diff_ub, (1, w))
        gi = canvas(bgc, (h, w))
        remcols = remove(bgc, cols)
        scols = [choice(remcols) for j in range(nc)]
        slocs = sample(interval(0, w, 1), nc)
        inds = totuple(connect(ORIGIN, (h - 1, 0)))
        for c, l in zip(scols, slocs):
            nc2 = randint(1, h - 1)
            sel = sample(inds, nc2)
            gi = fill(gi, c, shift(sel, tojvec(l)))
        go = replace(ff(replace(gi, bgc, -1)), -1, bgc)
        if colorcount(gi, bgc) > argmax(remove(bgc, palette(gi)), lbind(colorcount, gi)):
            break
    return {'input': gi, 'output': go}


def _bgc(I, examples):
    from collections import Counter
    # Background is the canvas colour; the generator guarantees it outnumbers
    # every other colour, and all example inputs share it.
    cnt = Counter(np.asarray(I).flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei).flatten().tolist())
    return cnt.most_common(1)[0][0]


def derive_operations(I, O, examples=None):
    # Gravity: in every column, the non-background cells fall to the bottom,
    # keeping their vertical order. Each maximal vertical run of non-bg cells
    # moves down as one object by the number of bg cells below it. Runs are
    # processed bottom-up so the space below is already settled.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = _bgc(I, examples)
    G = I.copy()
    ops, sels = [], []
    for c in range(wi):
        col = G[:, c]
        runs = []
        r = 0
        while r < hi:
            if col[r] != bgc:
                s = r
                while r < hi and col[r] != bgc:
                    r += 1
                runs.append((s, r))  # rows s..r-1
            else:
                r += 1
        floor = hi  # first row occupied by settled cells
        for s, e in reversed(runs):
            n = e - s
            d = floor - e
            floor -= n
            if d == 0:
                continue
            src = [(rr, c) for rr in range(s, e)]
            dst = [(rr + d, c) for rr in range(s, e)]
            vals = [int(G[rr, c]) for rr in range(s, e)]
            nz = [p for p, v in zip(src, vals) if v != 0]
            if nz:
                # grab the run once, then continue with empty selections
                ops.append(21); sels.append(sel_of(src))
                for _ in range(d - 1):
                    ops.append(21); sels.append(sel_of([]))
                # simulate: snapshot with src zeroed, paste nonzero cells
                snap = G.copy()
                for p in src:
                    snap[p] = 0
                for (rr, cc), v in zip(src, vals):
                    if v != 0:
                        snap[rr + d, cc] = v
                G = snap
                # cells of the run that are colour 0 were not carried by Move
                zfix = [(rr + d, cc) for (rr, cc), v in zip(src, vals)
                        if v == 0 and G[rr + d, cc] != 0]
                if zfix:
                    ops.append(0); sels.append(sel_of(zfix))
                    for p in zfix:
                        G[p] = 0
                # repair only the original footprint the run no longer covers
                hole = sorted(set(src) - set(dst))
                if bgc != 0 and hole:
                    ops.append(bgc); sels.append(sel_of(hole))
                    for p in hole:
                        G[p] = bgc
            else:
                # run made only of colour 0 (bgc != 0): Move cannot carry 0s
                hole = sorted(set(src) - set(dst))
                ops.append(bgc); sels.append(sel_of(hole))
                for p in hole:
                    G[p] = bgc
                new = sorted(set(dst) - set(src))
                ops.append(0); sels.append(sel_of(new))
                for p in new:
                    G[p] = 0
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
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
                                f"for task 1e0a9b12"
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
                    f"Failed to build a complete episode for task 1e0a9b12 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e0a9b12-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
