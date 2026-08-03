"""
ARC Task: 1fad071e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import deque, Counter

VARIANTS = [{"nbl": 0}, {"nbl": 1}, {"nbl": 2}, {"nbl": 3}, {"nbl": 4}, {"nbl": 5}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, otherc = sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
    else:
        idxs = sample(list(range(len(VARIANTS))), n_ex)
        examples = [dict(VARIANTS[i]) for i in idxs]
    plan = examples + [dict(choice(examples))]
    return {"bgc": bgc, "otherc": otherc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, otherc=None, nbl=None) -> dict:
    if nbl is None:
        nbl = randint(0, 5)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (nbl, max(nbl, (h * w) // 10)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    ofcfrbinds = {1: set(), otherc: set()}
    while succ < nobjs and tr < maxtr:
        tr += 1
        col = choice((1, otherc))
        oh = randint(1, 3)
        ow = randint(1, 3)
        if bcount < nbl:
            col = 1
            oh, ow = 2, 2
        else:
            while col == 1 and oh == ow == 2:
                col = choice((1, otherc))
                oh = randint(1, 3)
                ow = randint(1, 3)
        bd = backdrop(frozenset({(0, 0), (oh - 1, ow - 1)}))
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        bd = shift(bd, loc)
        if bd.issubset(inds) and len(mapply(dneighbors, bd) & ofcfrbinds[col]) == 0:
            succ += 1
            inds = inds - bd
            ofcfrbinds[col] = ofcfrbinds[col] | mapply(dneighbors, bd) | bd
            gi = fill(gi, col, bd)
            if col == 1 and oh == ow == 2:
                bcount += 1
    go = (repeat(1, bcount) + repeat(bgc, 5 - bcount),)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the colour the canvas was painted with before objects were placed
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # count the 2x2 blocks of colour 1 in I (4-connected, univalued components)
    seen = np.zeros((hi, wi), dtype=bool)
    k = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != 1 or seen[r, c]:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 1:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            hh = max(rs) - min(rs) + 1
            ww = max(cs) - min(cs) + 1
            if len(comp) == 4 and hh == 2 and ww == 2:
                k += 1

    ops, sels = [], []

    # shrink canvas down to the 1x5 tally bar
    ops.append(33); sels.append([0, 0, 0, 4])
    strip = [int(I[0, j]) for j in range(5)]

    # left segment: k cells of colour 1 (one per 2x2 block found)
    if k > 0 and any(strip[j] != 1 for j in range(k)):
        ops.append(1); sels.append([0, 0, 0, k - 1])
    # right segment: remainder stays background
    if k < 5 and any(strip[j] != bgc for j in range(k, 5)):
        ops.append(bgc); sels.append([0, k, 0, 4 - k])

    ops.append(34); sels.append([0, 0, 0, 4])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1fad071e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1fad071e"
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
                                f"for task 1fad071e"
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
                    f"Failed to build a complete episode for task 1fad071e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1fad071e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
