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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(2, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if bgc is None:
        bgc = random.randint(0, 9)
    if ccols is None:
        rem = [c for c in range(10) if c != bgc]
        ccols = random.sample(rem, random.randint(2, 8))

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    inds = {(i, j) for i in range(h) for j in range(w)}
    rots = [lambda g: g, lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    while tr < maxtr and succ < nobjs:
        if len(inds) == 0:
            break
        tr += 1
        dim = random.randint(1, 3)
        obj = {(i, j) for i in range(dim + 1) for j in range(dim + 1) if i <= j}
        mir = {(i, dim - j) for (i, j) in obj}
        obj = obj | {(i, j + dim) for (i, j) in mir}
        mp = (0, dim)
        tric, linc = random.sample(ccols, 2)
        loci, locj = random.choice(sorted(inds))
        iplcd = {}
        for (i, j) in obj:
            iplcd[(i + loci, j + locj)] = linc if (i, j) == mp else tric
        oplcd = dict(iplcd)
        for i in range(loci, h):
            p = (i, locj + dim)
            if p not in oplcd:
                oplcd[p] = linc
        if set(oplcd.keys()).issubset(inds):
            succ += 1
            for (i, j), v in iplcd.items():
                gi[i, j] = v
            for (i, j), v in oplcd.items():
                go[i, j] = v
        rf = random.choice(rots)
        gi = np.ascontiguousarray(rf(gi))
        go = np.ascontiguousarray(rf(go))
        h, w = gi.shape
        ofc = {(i, j) for i in range(h) for j in range(w) if go[i, j] == bgc}
        bad = set()
        for i in range(h):
            for j in range(w):
                if go[i, j] != bgc:
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        bad.add((i + di, j + dj))
        inds = ofc - bad
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): every object is an arrow-head made of a bulk colour plus a single
    marker cell in the middle of its flat side. The arrow points away from the marker's one
    orthogonal neighbour that is outside the object. A ray of the marker colour runs from
    the marker in that direction to the grid edge, painting only background cells.

    Unit of work: one COLOUR at a time. All ray cells that get the same marker colour,
    across all arrows, are selected together and painted with one Color op.
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
            comp = []
            dq = deque([(r, c)])
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

    rays_by_colour = {}
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
        y, x = mp[0] + dy, mp[1] + dx
        while 0 <= y < hi and 0 <= x < wi:
            if (y, x) not in cells and I[y, x] == bgc:
                rays_by_colour.setdefault(marker, set()).add((y, x))
            y += dy
            x += dx

    ops, sels = [], []
    for colour in sorted(rays_by_colour):
        cells = sorted(rays_by_colour[colour])
        if not cells:
            continue
        ops.append(int(colour))
        sels.append(sel_of(cells))   # every ray of this marker colour, painted together

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
