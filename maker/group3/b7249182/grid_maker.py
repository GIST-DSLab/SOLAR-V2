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

try:
    from dsl import *
    from utils import unifint
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ca, cb = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transposed": False}, {"transposed": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ca=1, cb=2,
             transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice((True, False))
    # before dmirror: h rows, w cols; after transpose dims swap
    lim_h = max_w if transposed else max_h
    lim_w = max_h if transposed else max_w
    lim_h = max(7, min(30, lim_h))
    lim_w = max(5, min(30, lim_w))
    h = unifint(diff_lb, diff_ub, (7, lim_h))
    w = unifint(diff_lb, diff_ub, (5, lim_w))
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
    loci = randint(0, h - 2 * ih)
    locj = randint(0, w - 5)
    obj = asobject(subg)
    obj = shift(obj, (loci, locj))
    gi = fill(gi, ca, {(loci, locj + 2)})
    gi = fill(gi, cb, {(loci + 2 * ih - 1, locj + 2)})
    go = paint(go, obj)
    if transposed:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour (only two marker cells differ from it)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    (ra, ca_), (rb, cb_) = marks[0], marks[1]

    if ca_ == cb_:
        vertical = True
        fixed = ca_
        p_lo, p_hi = min(ra, rb), max(ra, rb)
        col_lo, col_hi = int(I[p_lo, fixed]), int(I[p_hi, fixed])
    else:
        vertical = False
        fixed = ra
        p_lo, p_hi = min(ca_, cb_), max(ca_, cb_)
        col_lo, col_hi = int(I[fixed, p_lo]), int(I[fixed, p_hi])

    ih = (p_hi - p_lo + 1) // 2
    mid = p_lo + ih          # first cell of the high half

    def cell(p, s):
        return (p, s) if vertical else (s, p)

    def rect(p0, p1, s0, s1):
        return [cell(p, s) for p in range(p0, p1 + 1) for s in range(s0, s1 + 1)]

    ops, sels = [], []

    # 1. the high marker shoots a line all the way to the low marker
    ops.append(col_hi); sels.append(sel_of(rect(p_lo + 1, p_hi - 1, fixed, fixed)))
    # 2. the low marker claims its half of that line
    ops.append(col_lo); sels.append(sel_of(rect(p_lo + 1, mid - 1, fixed, fixed)))
    # 3. bracket box (4 x 5) centred on the meeting point: low half in col_lo
    ops.append(col_lo); sels.append(sel_of(rect(mid - 2, mid - 1, fixed - 2, fixed + 2)))
    # 4. high half of the box in col_hi
    ops.append(col_hi); sels.append(sel_of(rect(mid, mid + 1, fixed - 2, fixed + 2)))
    # 5. hollow out the box interior (2 x 3) back to background
    ops.append(bgc); sels.append(sel_of(rect(mid - 1, mid, fixed - 1, fixed + 1)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
