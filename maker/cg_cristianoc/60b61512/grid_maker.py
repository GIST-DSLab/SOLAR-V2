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
from collections import deque


def sample_colors(num_examples=None) -> dict:
    # Same colour pools as the generator. 7 is reserved because it is the fill colour.
    cols = [c for c in range(10) if c != 7]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 8)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    def neighbors4(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    cols = [c for c in range(10) if c != 7]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        indsss = {(i, j) for i in range(oh) for j in range(ow)}
        chch = random.choice(sorted(indsss))
        obj = {chch}
        indsss = indsss - {chch}
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = random.choice((numcd, oh * ow - numcd))
        numc = min(max(2, numc), oh * ow - 1)
        for _ in range(numc):
            nb = set()
            for p in obj:
                nb |= neighbors4(p)
            cand = sorted(indsss & nb)
            if not cand:
                break
            obj.add(random.choice(cand))
            indsss = indsss - obj
        obj = {(r + loci, c + locj) for r, c in obj}
        r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
        c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)
        bd = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        col = random.choice(ccols)
        if bd.issubset(indss):
            for (r, c) in bd:
                go[r][c] = 7
            for (r, c) in obj:
                gi[r][c] = col
                go[r][c] = col
            succ += 1
            outbox = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)} - bd
            indss = (indss - bd) - outbox
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # The golf entry reads the task the same way the maker does. For every object it
    # takes the object's bounding box and turns the background cells inside that box
    # into 7. The golf version just hardcodes 3x3 frames. The generator keeps each
    # object's backdrop (and the ring around it) apart from every other object. So inside
    # one object's box, "box minus object" and "background cells in the box" are the same
    # cells. Both readings agree, so the unit is the same: one Color7 per object, on the
    # background cells of its bounding box. All of this is read from I.
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # The generator only places objects where loci < h-oh and locj < w-ow, so the last
    # row and last column are always background. That makes the bottom-right cell a
    # reliable background colour, even when objects cover most of the grid.
    bgc = int(I[h - 1, w - 1])

    # Same-coloured, diagonally connected objects (objects(I, T, T, T)).
    seen = np.zeros((h, w), dtype=bool)
    objects = []
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
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            objects.append(cells)

    ops, sels = [], []
    for cells in objects:
        r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
        c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
        # Background cells inside this object's bounding box (its frame's empty spots).
        holes = [(r, c)
                 for r in range(r0, r1 + 1)
                 for c in range(c0, c1 + 1)
                 if I[r, c] == bgc]
        if not holes:
            continue  # the object is a solid rectangle, so a Color7 here would change nothing
        ops.append(7)
        sels.append(sel_of(holes))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole-grid bbox for Submit
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
