"""
ARC Task: 8d510a79 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = round(random.uniform(lo, hi))
    return int(max(a, min(b, v)))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    barcol = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barcol": barcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barcol=None, mirror=None, **kw) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)
    if barcol is None:
        barcol = random.choice([c for c in cols if c != bgc])
    if mirror is None:
        mirror = random.choice((True, False))
    # when mirrored, h becomes width and w becomes height
    H, W = (max_w, max_h) if mirror else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (5, max(5, H)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, W)))
    barloci = random.randint(2, h - 3)
    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[barloci][j] = barcol
    go = [row[:] for row in gi]
    jinds = list(range(w))
    numtop = _unifint(diff_lb, diff_ub, (1, w - 1))
    numbot = _unifint(diff_lb, diff_ub, (1, w - 1))
    for t in random.sample(jinds, numtop):
        loci = random.randint(0, barloci - 2)
        col = random.choice((1, 2))
        gi[loci][t] = col
        rng = range(0, loci + 1) if col == 1 else range(loci, barloci)
        for r in rng:
            go[r][t] = col
    for t in random.sample(jinds, numbot):
        loci = random.randint(barloci + 2, h - 1)
        col = random.choice((1, 2))
        gi[loci][t] = col
        rng = range(loci, h) if col == 1 else range(barloci + 1, loci + 1)
        for r in rng:
            go[r][t] = col
    if mirror:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # find the bar: a full uniform row/column whose colour is neither bgc nor 1/2
    bar = None
    for r in range(hi):
        v = int(I[r, 0])
        if v not in (bgc, 1, 2) and (I[r] == v).all():
            bar = ("row", r)
            break
    if bar is None:
        for c in range(wi):
            v = int(I[0, c])
            if v not in (bgc, 1, 2) and (I[:, c] == v).all():
                bar = ("col", c)
                break
    kind, b = bar
    ops, sels = [], []
    markers = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] in (1, 2)]
    # order markers along the bar
    markers.sort(key=lambda rc: (rc[1], rc[0]) if kind == "row" else (rc[0], rc[1]))
    for (r, c) in markers:
        col = int(I[r, c])
        # 2 extends toward the bar (stopping adjacent to it); 1 extends away to the grid edge
        if kind == "row":
            if r < b:
                seg = range(r + 1, b) if col == 2 else range(0, r)
            else:
                seg = range(b + 1, r) if col == 2 else range(r + 1, hi)
            cells = [(rr, c) for rr in seg]
        else:
            if c < b:
                seg = range(c + 1, b) if col == 2 else range(0, c)
            else:
                seg = range(b + 1, c) if col == 2 else range(c + 1, wi)
            cells = [(r, cc) for cc in seg]
        cells = [p for p in cells if I[p] != col]
        if cells:
            ops.append(col); sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
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
                                f"for task 8d510a79"
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
                    f"Failed to build a complete episode for task 8d510a79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d510a79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
