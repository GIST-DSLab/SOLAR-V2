"""
ARC Task: 810b9b61 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on object shape (hollow rectangular frame), not on object colors.
    # 3 is reserved as the fill color, so bgc must never be 3.
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = [c for c in range(10) if c != 3 and c != bgc]

    lo_h = min(10, max_h)
    lo_w = min(10, max_w)
    h = random.randint(lo_h, max_h)
    w = random.randint(lo_w, max_w)

    ncols = random.randint(1, min(6, len(cols)))
    ccols = random.sample(cols, ncols)

    nobjs = random.randint(3, max(3, (h * w) // 10))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def neighbors(r, c):
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    occupied = set()   # cells belonging to any placed object
    succ = 0
    tr = 0
    maxtr = 5 * nobjs

    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        if oh > h or ow > w:
            continue
        r0 = random.randint(0, h - oh)
        c0 = random.randint(0, w - ow)

        # full rectangular frame (outline of the bbox)
        frame = set()
        for r in range(r0, r0 + oh):
            for c in range(c0, c0 + ow):
                if r in (r0, r0 + oh - 1) or c in (c0, c0 + ow - 1):
                    frame.add((r, c))

        # choose full frame or a strict partial subset of the frame
        full = random.choice([True, False])
        if full:
            inobj = set(frame)
        else:
            fl = list(frame)
            k = random.randint(1, len(fl) - 1)
            inobj = set(random.sample(fl, k))

        # keep objects separated by >=1 cell so components never merge
        buffer = set()
        for (r, c) in inobj:
            buffer.add((r, c))
            for nb in neighbors(r, c):
                buffer.add(nb)
        if occupied & buffer:
            continue

        col = random.choice(ccols)
        is_frame = (inobj == frame) and min(oh, ow) > 2
        ocol = 3 if is_frame else col

        for (r, c) in inobj:
            gi[r][c] = col
            go[r][c] = ocol
        occupied |= inobj
        succ += 1

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = dominant color (objects are sparse frames on a solid canvas)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    ops, sels = [], []

    # Rule (measured from I): an object is recolored to 3 iff it is a hollow
    # rectangular frame == outline of its bounding box AND min(shape) > 2.
    # Iterate whole objects (connected same-color components), one op per frame.
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            color = I[r, c]
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == color:
                        seen[ny, nx] = True
                        dq.append((ny, nx))

            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            bh = rmax - rmin + 1
            bw = cmax - cmin + 1
            if min(bh, bw) <= 2:
                continue

            perim = set()
            for y in range(rmin, rmax + 1):
                for x in range(cmin, cmax + 1):
                    if y in (rmin, rmax) or x in (cmin, cmax):
                        perim.add((y, x))

            if set(comp) == perim:
                # hollow frame: one FloodFill3 seeded at a corner (always a frame cell)
                ops.append(13)
                sels.append([rmin, cmin, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
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
                                f"for task 810b9b61"
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
                    f"Failed to build a complete episode for task 810b9b61 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"810b9b61-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
