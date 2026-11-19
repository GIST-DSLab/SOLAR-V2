"""
ARC Task: 44f52bb0 (RE-ARC) — LLM-generated grid_maker
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


def _hmirror(g):
    return tuple(g[::-1])


def _vmirror(g):
    return tuple(tuple(r[::-1]) for r in g)


def _dmirror(g):
    return tuple(zip(*g))


def _cmirror(g):
    return _vmirror(_dmirror(_vmirror(g)))


def _rot90(g):
    return tuple(row[::-1] for row in zip(*g))


def _rot180(g):
    return tuple(tuple(row[::-1]) for row in g[::-1])


def _rot270(g):
    return tuple(tuple(row[::-1]) for row in zip(*g[::-1]))[::-1]


def _identity(g):
    return g


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo, hi = int(round(lo)), int(round(hi))
    lo = max(a, min(b, lo))
    hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    random.shuffle(remcols)
    n_ex = num_examples if num_examples else 3
    variants = [{"issymm": True}, {"issymm": False}]
    if n_ex >= 2:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols_pool": remcols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols_pool=None,
             issymm=None, **kwargs) -> dict:
    if ccols_pool is None:
        ccols_pool = [c for c in range(10) if c != bgc]
    if issymm is None:
        issymm = random.choice((True, False))
    h = _unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    ncols = _unifint(diff_lb, diff_ub, (2, min(9, len(ccols_pool))))
    ccols = random.sample(list(ccols_pool), ncols)
    while True:
        numcells = _unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = [(i, j) for i in range(h) for j in range(w)]
        gi = tuple(tuple(bgc for _ in range(w)) for _ in range(h))
        tries = 0
        while gi == _hmirror(gi) and tries < 200:
            tries += 1
            cells = random.sample(inds, numcells)
            g = [[bgc] * w for _ in range(h)]
            for a, b in cells:
                col = random.choice(ccols)
                g[a][b] = col
                g[a][w - 1 - b] = col
            gi = tuple(tuple(r) for r in g)
        if gi == _hmirror(gi):
            continue
        if not issymm:
            numpert = _unifint(diff_lb, diff_ub, (1, h * (w // 2)))
            cands = [(i, j) for i in range(h) for j in range(w // 2)]
            locs = random.sample(cands, numpert)
            g = [list(r) for r in gi]
            pal = set(ccols) | {bgc}
            for a, b in locs:
                col = g[a][b]
                g[a][b] = random.choice(sorted(pal - {col}))
            gi = tuple(tuple(r) for r in g)
            if gi == _vmirror(gi) or gi == _hmirror(gi):
                continue
            go = ((7,),)
        else:
            go = ((1,),)
        break
    mfs = [_identity, _dmirror, _cmirror, _vmirror, _hmirror, _rot90, _rot180, _rot270]
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)

    # Rule measured from I alone: is the grid its own left-right or up-down mirror?
    symmetric = np.array_equal(I, np.fliplr(I)) or np.array_equal(I, np.flipud(I))
    color = 1 if symmetric else 7

    ops, sels = [], []
    # Crop the canvas down to the single 1x1 answer cell (full rectangle = one cell).
    ops.append(33); sels.append([0, 0, 0, 0])
    # After the crop the cell holds I[0,0] (0 if it was 0); paint the verdict colour
    # only if it is not already there.
    if int(I[0, 0]) != color:
        ops.append(color); sels.append(sel_of([(0, 0)]))
    ops.append(34); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
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
                                f"for task 44f52bb0"
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
                    f"Failed to build a complete episode for task 44f52bb0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44f52bb0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
