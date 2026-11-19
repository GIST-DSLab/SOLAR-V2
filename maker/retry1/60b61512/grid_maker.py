"""
ARC Task: 60b61512 (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 7]
    bgc = random.choice(cols)
    return {"bgc": bgc}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    import sys
    sys.path.insert(0, '/hdd_data/yunho/solar-traj/re-arc')
    from dsl import (interval, remove, canvas, asindices, totuple, sfilter, mapply,
                     neighbors, shape, shift, backdrop, fill, outbox)
    from utils import unifint
    cols = remove(7, interval(0, 10, 1))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(list(remcols), numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        indsss = asindices(canvas(-1, (oh, ow)))
        chch = random.choice(totuple(indsss))
        obj = {chch}
        indsss = remove(chch, indsss)
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = random.choice((numcd, oh * ow - numcd))
        numc = min(max(2, numc), oh * ow - 1)
        for k in range(numc):
            obj.add(random.choice(totuple(indsss & mapply(neighbors, obj))))
            indsss = indsss - obj
        oh, ow = shape(obj)
        obj = shift(obj, (loci, locj))
        bd = backdrop(obj)
        col = random.choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, obj)
            go = fill(go, 7, bd)
            go = fill(go, col, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}

def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter, deque
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Background: the canvas colour of THIS input (the generator paints it first and
    # object backdrops are disjoint, so it dominates). Not pooled across examples:
    # different episodes/examples may use a different canvas colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Objects (same colour, 8-connected) in I and their bounding rectangles.
    seen = np.zeros((h, w), dtype=bool)
    boxes = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = I[r, c]
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
            c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
            boxes.append((r0, r1, c0, c1))

    ops, sels = [], []
    # 1. Lay a 7 base over every object's bounding rectangle at once
    #    (skipped if every rectangle is already solid: base+redraw would be a cycle).
    union = set()
    for (r0, r1, c0, c1) in boxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                union.add((r, c))
    holes = [p for p in union if I[p] == bgc]
    if holes:
        ops.append(7)
        sels.append(sel_of(sorted(union)))
        # 2. Redraw the objects one colour at a time: all I-cells of that colour.
        colours = sorted({int(v) for v in I.flatten().tolist() if v != bgc and v != 7})
        for col in colours:
            cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] == col]
            ops.append(col)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 60b61512"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 60b61512"
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
                                f"for task 60b61512"
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
                    f"Failed to build a complete episode for task 60b61512 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"60b61512-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
