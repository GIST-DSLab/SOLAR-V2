"""
ARC Task: b7249182 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, ca, cb = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None, mirror=None) -> dict:
    from dsl import (interval, canvas, fill, connect, replace, vconcat, hmirror,
                     asobject, shift, paint, dmirror)
    from utils import unifint
    cols = interval(0, 10, 1)
    if bgc is None or ca is None or cb is None:
        bgc, ca, cb = random.sample(cols, 3)
    if mirror is None:
        mirror = random.choice((True, False))
    if mirror:
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(7, min(30, hb))
    wb = max(5, min(30, wb))
    h = unifint(diff_lb, diff_ub, (7, hb))
    w = unifint(diff_lb, diff_ub, (5, wb))
    ih = unifint(diff_lb, diff_ub, (3, (h - 1) // 2))
    subg = canvas(bgc, (ih, 5))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    subg = fill(subg, ca, connect((0, 2), (ih - 2, 2)))
    subg = fill(subg, ca, connect((ih - 2, 0), (ih - 2, 4)))
    subg = fill(subg, ca, {(ih - 1, 0)})
    subga = fill(subg, ca, {(ih - 1, 4)})
    subgb = replace(subga, ca, cb)
    subg = vconcat(subga, hmirror(subgb))
    loci = random.randint(0, h - 2 * ih)
    locj = random.randint(0, w - 5)
    obj = asobject(subg)
    obj = shift(obj, (loci, locj))
    gi = fill(gi, ca, {(loci, locj + 2)})
    gi = fill(gi, cb, {(loci + 2 * ih - 1, locj + 2)})
    go = paint(go, obj)
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pts = sorted([(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc])
    (ra, cA), (rb, cB) = pts[0], pts[1]
    transposed = (ra == rb)  # same row -> horizontal arrangement
    if transposed:
        # work in transposed coordinates: (r,c) -> (c,r)
        p1, p2 = sorted([(cA, ra), (cB, rb)])
    else:
        p1, p2 = (ra, cA), (rb, cB)
    r1, c = p1
    r2, _ = p2
    ca = int(I[p1[1], p1[0]]) if transposed else int(I[p1])
    cb = int(I[p2[1], p2[0]]) if transposed else int(I[p2])
    ih = (r2 - r1 + 1) // 2

    def T(cells):
        return [(cc, rr) for rr, cc in cells] if transposed else list(cells)

    ops, sels = [], []
    # --- upper/left arm in ca ---
    stem_a = [(r, c) for r in range(r1 + 1, r1 + ih - 1)]
    bar_a = [(r1 + ih - 2, cc) for cc in range(c - 2, c + 3)]
    legs_a = [(r1 + ih - 1, c - 2), (r1 + ih - 1, c + 2)]
    # --- lower/right arm in cb ---
    stem_b = [(r, c) for r in range(r1 + ih + 2, r2)]
    bar_b = [(r1 + ih + 1, cc) for cc in range(c - 2, c + 3)]
    legs_b = [(r1 + ih, c - 2), (r1 + ih, c + 2)]
    for col, cells in ((ca, stem_a), (ca, bar_a), (ca, legs_a),
                       (cb, stem_b), (cb, bar_b), (cb, legs_b)):
        if cells:
            ops.append(col); sels.append(sel_of(T(cells)))
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
                        f"num_examples+1 ({num_examples + 1}) for task b7249182"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b7249182"
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
                                f"for task b7249182"
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
                    f"Failed to build a complete episode for task b7249182 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b7249182-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
