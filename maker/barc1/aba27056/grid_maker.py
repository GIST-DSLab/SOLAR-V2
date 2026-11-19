"""
ARC Task: aba27056 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of clockwise quarter turns applied to the canonical (hole-on-top) cup


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return int(min(max(a, round(a + (b - a) * d)), b))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, rot=None, **kw) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    hm, wm = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (6, min(30, hm)))
    w = _unifint(diff_lb, diff_ub, (6, min(30, wm)))
    gi = np.full((h, w), bgc, dtype=int)
    oh = random.randint(3, h)
    ow = _unifint(diff_lb, diff_ub, (5, w - 1))
    loci = _unifint(diff_lb, diff_ub, (0, h - oh))
    locj = random.randint(0, w - ow)
    r0, c0, r1, c1 = loci, locj, loci + oh - 1, locj + ow - 1
    k = random.randint(0, (ow - 4) // 2)
    ha, hb = sorted((locj + 2 + k, locj + ow - 3 - k))  # DSL connect is order-free
    go = gi.copy()
    go[r0:r1 + 1, c0:c1 + 1] = 4
    for g in (gi, go):
        g[r0, c0:c1 + 1] = sqc; g[r1, c0:c1 + 1] = sqc
        g[r0:r1 + 1, c0] = sqc; g[r0:r1 + 1, c1] = sqc
    gi[r0, ha:hb + 1] = bgc
    go[0:r0 + 1, ha:hb + 1] = 4
    r, c = r0 - 1, hb + 1
    while r >= 0 and c < w:
        go[r, c] = 4; r -= 1; c += 1
    r, c = r0 - 1, ha - 1
    while r >= 0 and c >= 0:
        go[r, c] = 4; r -= 1; c -= 1
    gi = np.rot90(gi, -rot); go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the canvas colour (the cup is only a thin outline).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pts = np.argwhere(I != bgc)
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    G = I.copy()

    # 1) Fill the cup: every background cell inside the cup's bounding box
    #    (its interior plus the opening in its wall) becomes yellow.
    inside = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if I[r, c] == bgc]
    ops.append(4); sels.append(sel_of(inside))
    for r, c in inside:
        G[r, c] = 4

    # 2) The opening = the inside cells lying on the bbox boundary; its side gives the direction.
    hole = sorted(p for p in inside if p[0] in (r0, r1) or p[1] in (c0, c1))
    if all(p[0] == r0 for p in hole):
        d = (-1, 0)
    elif all(p[0] == r1 for p in hole):
        d = (1, 0)
    elif all(p[1] == c0 for p in hole):
        d = (0, -1)
    else:
        d = (0, 1)

    # 3) Shoot a straight yellow line out of every opening pixel.
    for (hr, hc) in hole:
        line = []
        r, c = hr + d[0], hc + d[1]
        while 0 <= r < hi and 0 <= c < wi:
            line.append((r, c)); r += d[0]; c += d[1]
        if line:
            ops.append(4); sels.append(sel_of([(hr, hc)] + line))
            for p in line:
                G[p] = 4

    # 4) From the two extreme opening pixels, shoot diagonals in all four directions,
    #    each stopping at the first non-background cell of the current grid.
    ends = [hole[0], hole[-1]] if len(hole) > 1 else [hole[0]]
    for (er, ec) in ends:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            ray = []
            r, c = er + dr, ec + dc
            while 0 <= r < hi and 0 <= c < wi and G[r, c] == bgc:
                ray.append((r, c)); r += dr; c += dc
            if ray:
                ops.append(4); sels.append(sel_of(ray))
                for p in ray:
                    G[p] = 4

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
                        f"num_examples+1 ({num_examples + 1}) for task aba27056"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task aba27056"
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
                                f"for task aba27056"
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
                    f"Failed to build a complete episode for task aba27056 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"aba27056-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
