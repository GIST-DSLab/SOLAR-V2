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

ROTS = [0, 1, 2, 3]

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a: return a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))
    if rot is None:
        rot = random.choice(ROTS)
    # after rotation by 90/270 the dims swap; bound accordingly
    H, W = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = unifint(diff_lb, diff_ub, (6, max(6, H)))
    w = unifint(diff_lb, diff_ub, (6, max(6, W)))
    gi = np.full((h, w), bgc, dtype=int)
    oh = random.randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = random.randint(0, w - ow)
    r0, c0, r1, c1 = loci, locj, loci + oh - 1, locj + ow - 1
    box = set()
    for c in range(c0, c1 + 1): box |= {(r0, c), (r1, c)}
    for r in range(r0, r1 + 1): box |= {(r, c0), (r, c1)}
    k = random.randint(0, (ow - 4) // 2)
    ja, jb = sorted((locj + 2 + k, locj + ow - 3 - k))
    hole = [(loci, c) for c in range(ja, jb + 1)]
    for r, c in box: gi[r, c] = sqc
    for r, c in hole: gi[r, c] = bgc
    go = np.full((h, w), bgc, dtype=int)
    go[r0:r1 + 1, c0:c1 + 1] = 4
    for r, c in box: go[r, c] = sqc
    for r, c in hole: go[:r + 1, c] = 4
    hr, hc = hole[-1]
    r, c = hr - 1, hc + 1
    while r >= 0 and c < w: go[r, c] = 4; r -= 1; c += 1
    hr, hc = hole[0]
    r, c = hr - 1, hc - 1
    while r >= 0 and c >= 0: go[r, c] = 4; r -= 1; c -= 1
    gi = np.rot90(gi, -rot); go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    FOUR = 4

    # Background: the colour that is NOT a hollow outline. The box colour has every
    # cell on the border of its own bounding box; background has cells strictly inside.
    bgc = None
    for col in set(I.flatten().tolist()):
        pts = np.argwhere(I == col)
        a, b = pts.min(0); c, d = pts.max(0)
        on_border = all(r in (a, c) or q in (b, d) for r, q in pts)
        if on_border and c - a >= 2 and d - b >= 2:
            continue
        bgc = col
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pts = np.argwhere(I != bgc)
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    # Find the open edge (gap) and its outward direction.
    edges = [
        ("top", [(r0, c) for c in range(c0, c1 + 1)]),
        ("bottom", [(r1, c) for c in range(c0, c1 + 1)]),
        ("left", [(r, c0) for r in range(r0, r1 + 1)]),
        ("right", [(r, c1) for r in range(r0, r1 + 1)]),
    ]
    side, hole = max(((s, [p for p in L if I[p] == bgc]) for s, L in edges),
                     key=lambda x: len(x[1]))
    vertical = side in ("top", "bottom")   # light travels along columns

    # Line-by-line: each interior line running in the light's direction.
    # A line under the gap continues out through the gap to the grid border.
    if vertical:
        gap = {c for _, c in hole}
        for c in range(c0 + 1, c1):
            ra, rb = r0 + 1, r1 - 1
            if c in gap:
                if side == "top": ra = 0
                else: rb = hi - 1
            ops.append(FOUR); sels.append(sel_of([(r, c) for r in range(ra, rb + 1)]))
    else:
        gap = {r for r, _ in hole}
        for r in range(r0 + 1, r1):
            ca, cb = c0 + 1, c1 - 1
            if r in gap:
                if side == "left": ca = 0
                else: cb = wi - 1
            ops.append(FOUR); sels.append(sel_of([(r, c) for c in range(ca, cb + 1)]))

    # Two diagonal rays from the gap's ends, each painted as one whole line.
    d = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}[side]
    t = (0, 1) if vertical else (1, 0)
    p1, p2 = hole[0], hole[-1]
    for (pr, pc), s in ((p1, -1), (p2, 1)):
        dr, dc = d[0] + s * t[0], d[1] + s * t[1]
        r, c = pr + dr, pc + dc
        ray = []
        while 0 <= r < hi and 0 <= c < wi:
            ray.append((r, c)); r += dr; c += dc
        if ray:
            ops.append(FOUR); sels.append(sel_of(ray))

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
