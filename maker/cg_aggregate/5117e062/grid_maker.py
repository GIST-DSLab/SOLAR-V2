"""
ARC Task: 5117e062 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background matters to the rule: the target is "the object with more than
    # one colour", and its marker/body colours are just roles, so they stay random.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _neighbors(loc):
    r, c = loc
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def _dneighbors(loc):
    r, c = loc
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    go = None
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        bx = {(i, j) for i in range(oh) for j in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx.discard(sp)
        obj = {sp}
        for _ in range(nc - 1):
            nb = set()
            for o in obj:
                nb |= _dneighbors(o)
            cands = sorted((bx - obj) & nb)
            obj.add(random.choice(cands))
        is_target = False
        if not done:
            idx = random.choice(sorted(obj))
            coll = random.choice(remcols)
            coll2 = random.choice([c for c in remcols if c != coll])
            cobj = {(coll, idx)} | {(coll2, ij) for ij in obj if ij != idx}
            is_target = True
        else:
            cc = random.choice(remcols)
            cobj = {(cc, ij) for ij in obj}
        tr += 1
        locopts = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not locopts:
            continue
        loc = random.choice(sorted(locopts))
        plcd = {(c, (i + loc[0], j + loc[1])) for c, (i, j) in cobj}
        plcdi = {ij for _, ij in plcd}
        if plcdi.issubset(inds):
            for c, (i, j) in plcd:
                gi[i][j] = c
            succ += 1
            nb = set()
            for ij in plcdi:
                nb |= _neighbors(ij)
            inds = (inds - plcdi) - nb
            if is_target:
                done = True
                ri = [i for i, _ in obj]
                ci = [j for _, j in obj]
                r0, c0 = min(ri), min(ci)
                oh2 = max(ri) - r0 + 1
                ow2 = max(ci) - c0 + 1
                go = [[bgc] * ow2 for _ in range(oh2)]
                for (i, j) in obj:
                    go[i - r0][j - c0] = coll2
    if go is None:
        raise ValueError("target object not placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour (verifier uses mostcolor(I))
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected multicolour components (generator keeps objects apart incl. diagonals)
    visited = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] != bgc:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cells)

    # the target is the one object carrying a marker of a second colour
    target = None
    for cells in comps:
        if len({int(I[y, x]) for (y, x) in cells}) > 1:
            target = cells
            break
    if target is None:
        target = max(comps, key=len)

    ys = [y for y, _ in target]
    xs = [x for _, x in target]
    r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    body = Counter(int(I[y, x]) for (y, x) in target).most_common(1)[0][0]
    objset = set(target)

    ops, sels = [], []

    # 1) cut out the target's frame: crop to its bounding rectangle
    #    (full rectangle intended, background corners included)
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    # Simulate the cropped grid (transparent copy: zeros dropped -> 0)
    G = I[r0:r1 + 1, c0:c1 + 1].copy()

    # 2) clear any piece of a neighbouring object that pokes into the frame
    intruders = [(r - r0, c - c0) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if (r, c) not in objset and I[r, c] != bgc]
    intruders = [(r, c) for (r, c) in intruders if G[r, c] != bgc]
    if intruders:
        ops.append(int(bgc))
        sels.append(sel_of(intruders))
        for (r, c) in intruders:
            G[r, c] = bgc

    # 3) the marker takes on the object's own colour
    marker = [(y - r0, x - c0) for (y, x) in target if I[y, x] != body]
    marker = [(r, c) for (r, c) in marker if G[r, c] != body]
    if marker:
        ops.append(int(body))
        sels.append(sel_of(marker))

    ops.append(34)
    sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5117e062"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5117e062"
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
                                f"for task 5117e062"
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
                    f"Failed to build a complete episode for task 5117e062 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5117e062-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
