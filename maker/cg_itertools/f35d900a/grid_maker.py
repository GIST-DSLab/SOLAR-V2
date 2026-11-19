"""
ARC Task: f35d900a (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 5]
    bgc, c1, c2 = random.sample(cols, 3)
    return {"bgc": bgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if bgc is None or c1 is None or c2 is None:
        cols = [c for c in range(10) if c != 5]
        bgc, c1, c2 = random.sample(cols, 3)

    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    oh = unifint(diff_lb, diff_ub, (4, h))
    ow = unifint(diff_lb, diff_ub, (4, w))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    r0, c0, r1, c1_ = loci, locj, loci + oh - 1, locj + ow - 1

    gi = np.full((h, w), bgc, dtype=int)
    gi[r0, c0] = c1
    gi[r1, c1_] = c1
    gi[r0, c1_] = c2
    gi[r1, c0] = c2
    go = gi.copy()

    def ring(r, c):
        return [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if (rr, cc) != (r, c) and 0 <= rr < h and 0 <= cc < w]

    for (r, c) in [(r0, c1_), (r1, c0)]:
        for (rr, cc) in ring(r, c):
            go[rr, cc] = c1
    for (r, c) in [(r0, c0), (r1, c1_)]:
        for (rr, cc) in ring(r, c):
            go[rr, cc] = c2

    corners = [(r0, c0), (r0, c1_), (r1, c0), (r1, c1_)]
    box = set()
    for c in range(c0, c1_ + 1):
        box.add((r0, c)); box.add((r1, c))
    for r in range(r0, r1 + 1):
        box.add((r, c0)); box.add((r, c1_))
    for cr in corners:
        rem = [x for x in corners if x != cr]
        for (r, c) in box:
            d = abs(r - cr[0]) + abs(c - cr[1])
            dm = min(abs(r - x[0]) + abs(c - x[1]) for x in rem)
            if d <= dm and d > 1 and d % 2 == 0:
                go[r, c] = 5
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: generator paints the canvas with bgc and only adds 4 dots
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    rs = [r for r, _ in dots]
    cs = [c for _, c in dots]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]  # walk around the box

    palette = sorted({int(I[r, c]) for r, c in dots})

    def other_of(col):
        if len(palette) < 2:
            return col
        return palette[0] if col == palette[1] else palette[1]

    ops, sels = [], []

    # 1) Each corner dot gets a ring in the other dot's colour.
    for (r, c) in corners:
        ring = [(rr, cc)
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
                if (rr, cc) != (r, c) and 0 <= rr < h and 0 <= cc < w]
        if ring:
            ops.append(other_of(int(I[r, c])))
            sels.append(sel_of(ring))

    # 2) From every corner, a dotted trail of 5s goes along both of its box edges
    #    toward the neighbouring corners. It steps 2 cells at a time and stops at
    #    the edge's midpoint. Emit it outward in waves: every trail at
    #    distance 2, then every trail at distance 4, ...
    #    Each trail is (corner, unit direction, edge length).
    trails = []
    for (r, c) in corners:
        for (nr, nc) in corners:
            if (nr, nc) == (r, c):
                continue
            if nr == r or nc == c:  # a box-edge neighbour
                L = abs(nr - r) + abs(nc - c)
                dr = (nr > r) - (nr < r)
                dc = (nc > c) - (nc < c)
                trails.append((r, c, dr, dc, L))

    maxL = max(t[4] for t in trails)
    d = 2
    while 2 * d <= maxL:
        wave = set()
        for (r, c, dr, dc, L) in trails:
            if 2 * d <= L:  # still within this corner's half of the edge
                wave.add((r + dr * d, c + dc * d))
        if wave:
            ops.append(5)
            sels.append(sel_of(sorted(wave)))
        d += 2

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f35d900a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f35d900a"
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
                                f"for task f35d900a"
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
                    f"Failed to build a complete episode for task f35d900a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f35d900a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
