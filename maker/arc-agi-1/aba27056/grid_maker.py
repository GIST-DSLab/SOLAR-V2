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

VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    if rot in (1, 3):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(6, hb)
    wb = max(6, wb)
    h = unifint(diff_lb, diff_ub, (6, hb))
    w = unifint(diff_lb, diff_ub, (6, wb))
    canv = canvas(bgc, (h, w))
    oh = randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    maxk = (ow - 4) // 2
    k = randint(0, maxk)
    hole = connect((loci, locj + 2 + k), (loci, locj + ow - 3 - k))
    gi = fill(canv, sqc, bx)
    gi = fill(gi, bgc, hole)
    go = fill(canv, 4, backdrop(bx))
    go = fill(go, sqc, bx)
    bar = mapply(rbind(shoot, (-1, 0)), hole)
    go = fill(go, 4, bar)
    go = fill(go, 4, shoot(add((-1, 1), urcorner(hole)), (-1, 1)))
    go = fill(go, 4, shoot(add((-1, -1), ulcorner(hole)), (-1, -1)))
    rotf = [identity, rot90, rot180, rot270][rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background = the canvas colour; the only other colour draws the box outline.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pts = np.argwhere(I != bgc)
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    # 1) The box's enclosed interior becomes the light (4) region.
    ops.append(4)
    sels.append([r0 + 1, c0 + 1, r1 - r0 - 2, c1 - c0 - 2])

    # 2) Locate the gap: the one box edge whose line still shows background.
    edges = [
        ([(r0, c) for c in range(c0, c1 + 1)], (-1, 0), (0, 1)),   # top    -> shines up
        ([(r1, c) for c in range(c0, c1 + 1)], (1, 0), (0, 1)),    # bottom -> shines down
        ([(r, c0) for r in range(r0, r1 + 1)], (0, -1), (1, 0)),   # left   -> shines left
        ([(r, c1) for r in range(r0, r1 + 1)], (0, 1), (1, 0)),    # right  -> shines right
    ]
    hole, d, t, best = None, None, None, 0
    for line, dd, tt in edges:
        hc = [p for p in line if I[p[0], p[1]] == bgc]
        if len(hc) > best:
            best, hole, d, t = len(hc), hc, dd, tt

    p1, p2 = hole[0], hole[-1]

    # 3) The straight beam: the gap itself plus everything it lights up
    #    perpendicularly outward, up to the grid border.
    rmin = min(p1[0], p2[0]); rmax = max(p1[0], p2[0])
    cmin = min(p1[1], p2[1]); cmax = max(p1[1], p2[1])
    if d[0] < 0:
        rmin = 0
    elif d[0] > 0:
        rmax = hi - 1
    if d[1] < 0:
        cmin = 0
    elif d[1] > 0:
        cmax = wi - 1
    ops.append(4)
    sels.append([rmin, cmin, rmax - rmin, cmax - cmin])

    # 4) The two diagonal rays spreading from the gap's two ends, one ray at a time.
    rays = [
        ((p1[0] + d[0] - t[0], p1[1] + d[1] - t[1]), (d[0] - t[0], d[1] - t[1])),
        ((p2[0] + d[0] + t[0], p2[1] + d[1] + t[1]), (d[0] + t[0], d[1] + t[1])),
    ]
    for (sr, sc), (dr, dc) in rays:
        r, c = sr, sc
        while 0 <= r < hi and 0 <= c < wi:
            ops.append(4)
            sels.append([r, c, 0, 0])
            r += dr
            c += dc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
