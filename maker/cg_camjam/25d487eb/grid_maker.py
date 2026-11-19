"""
ARC Task: 25d487eb (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background colour plays a role in the rule; the arrow-head colours
    # (bulk / marker) are read off each object, so they are sampled per instance.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if bgc is None:
        bgc = random.choice(range(10))
    cap = max(10, min(max_h, max_w))  # grid gets rotated, so both dims share one cap
    h = unifint(diff_lb, diff_ub, (10, cap))
    w = unifint(diff_lb, diff_ub, (10, cap))
    remcols = [c for c in range(10) if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (2, 8))
    ccols = random.sample(remcols, ncols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))
    succ, tr, maxtr = 0, 0, 10 * nobjs
    inds = {(i, j) for i in range(h) for j in range(w)}
    rots = [lambda g: g, lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    while tr < maxtr and succ < nobjs:
        if len(inds) == 0:
            break
        tr += 1
        dim = random.randint(1, 3)
        obj = {(i, j) for i in range(dim + 1) for j in range(dim + 1) if i <= j}
        obj = obj | {(i, dim - j + dim) for (i, j) in obj}
        mp = (0, dim)
        tric, linc = random.sample(ccols, 2)
        loci, locj = random.choice(sorted(inds))
        icells = {(loci + i, locj + j): (linc if (i, j) == mp else tric) for (i, j) in obj}
        ocells = dict(icells)
        for r in range(loci, h):
            if (r, locj + dim) not in icells:
                ocells[(r, locj + dim)] = linc
        if set(ocells).issubset(inds):
            succ += 1
            for (r, c), v in icells.items():
                gi[r, c] = v
            for (r, c), v in ocells.items():
                go[r, c] = v
        rotf = random.choice(rots)
        gi = np.ascontiguousarray(rotf(gi))
        go = np.ascontiguousarray(rotf(go))
        h, w = gi.shape
        nonbg = {(r, c) for r in range(h) for c in range(w) if go[r, c] != bgc}
        blocked = set()
        for (r, c) in nonbg:
            blocked.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                blocked.add((r + dr, c + dc))
        inds = {(r, c) for r in range(h) for c in range(w)} - blocked
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): every object is an arrow-head whose
    single odd-coloured marker sits mid flat side; the arrow points away from the marker's
    one outside neighbour.  The background run that starts just past the apex is painted
    in the marker colour to the edge of the grid -- one Color op per object.
    The golf entry processes the grid one pointing direction at a time (rotate, fill all
    rows that match, rotate again), so rays are emitted grouped by direction:
    down, right, up, left.  Each ray is the contiguous background run past the apex.
    O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    seen = np.zeros((hi, wi), dtype=bool)
    objects = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            comp, dq = [], deque([(r, c)])
            seen[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for dy, dx in DIRS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                        seen[ny, nx] = True
                        dq.append((ny, nx))
            objects.append(comp)

    rays = {d: [] for d in ((1, 0), (0, 1), (-1, 0), (0, -1))}
    for comp in objects:
        cells = set(comp)
        counts = Counter(int(I[y, x]) for y, x in comp)
        if len(counts) < 2:
            continue
        marker = min(counts, key=lambda k: counts[k])
        mp = next((y, x) for y, x in comp if int(I[y, x]) == marker)
        outside = [d for d in DIRS if (mp[0] + d[0], mp[1] + d[1]) not in cells]
        if len(outside) != 1:
            continue
        dy, dx = -outside[0][0], -outside[0][1]
        # walk through the object's own spine to the apex
        y, x = mp[0] + dy, mp[1] + dx
        while 0 <= y < hi and 0 <= x < wi and (y, x) in cells:
            y += dy
            x += dx
        # contiguous background run beyond the apex
        run = []
        while 0 <= y < hi and 0 <= x < wi and I[y, x] == bgc:
            run.append((y, x))
            y += dy
            x += dx
        if run:
            rays[(dy, dx)].append((marker, run))

    ops, sels = [], []
    for d in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        for marker, run in rays[d]:
            ops.append(int(marker))
            sels.append(sel_of(run))

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
                        f"num_examples+1 ({num_examples + 1}) for task 25d487eb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 25d487eb"
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
                                f"for task 25d487eb"
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
                    f"Failed to build a complete episode for task 25d487eb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"25d487eb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
