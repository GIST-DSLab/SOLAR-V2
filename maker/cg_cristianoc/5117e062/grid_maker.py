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
    # Only the background is an episode-level colour role; the rule (keep the
    # multi-coloured object, unify it to its majority colour) is colour-agnostic.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _nbrs8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield (r + dr, c + dc)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    h = min(max(unifint(10, 30), 10), max_h)
    w = min(max(unifint(10, 30), 10), max_w)
    nobjs = max(2, unifint(2, max(2, (h * w) // 15)))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    tr, maxtr, succ, done = 0, 4 * nobjs, 0, False
    go = None
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        bx = {(r, c) for r in range(oh) for c in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx.discard(sp)
        obj = {sp}
        for _ in range(nc - 1):
            cand = set()
            for (r, c) in obj:
                cand |= set(_nbrs8(r, c))
            cand = sorted((bx - obj) & cand)
            obj.add(random.choice(cand))
        if not done:
            done = True
            idx = random.choice(sorted(obj))
            coll = random.choice(remcols)
            coll2 = random.choice([c for c in remcols if c != coll])
            cobj = {(coll2, p) for p in obj if p != idx} | {(coll, idx)}
            mr = min(r for r, c in obj)
            mc = min(c for r, c in obj)
            hh = max(r for r, c in obj) - mr + 1
            ww = max(c for r, c in obj) - mc + 1
            go = [[bgc] * ww for _ in range(hh)]
            for (r, c) in obj:
                go[r - mr][c - mc] = coll2
            first = True
        else:
            col = random.choice(remcols)
            cobj = {(col, p) for p in obj}
            first = False
        tr += 1
        if h - oh < 0 or w - ow < 0:
            if first:
                done = False
            continue
        locopts = [(r, c) for (r, c) in inds if r <= h - oh and c <= w - ow]
        if not locopts:
            if first:
                done = False
            continue
        lr, lc = random.choice(sorted(locopts))
        plcd = {(v, (r + lr, c + lc)) for v, (r, c) in cobj}
        plcdi = {p for _, p in plcd}
        if plcdi.issubset(inds):
            for v, (r, c) in plcd:
                gi[r][c] = v
            succ += 1
            nb = set()
            for (r, c) in plcdi:
                nb |= set(_nbrs8(r, c))
            inds = (inds - plcdi) - nb
        elif first:
            done = False
    if go is None or succ < 2:
        raise ValueError("generation failed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: what every generator canvas is painted with (dominant colour).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected objects (the generator keeps objects 8-separated).
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
                for ny, nx in _nbrs8(y, x):
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] != bgc:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)

    # The object carrying a marker: the one with more than one colour.
    target = None
    for cells in comps:
        if len({int(I[y, x]) for (y, x) in cells}) > 1:
            target = cells
            break
    if target is None:
        target = max(comps, key=len)

    r0 = min(y for y, x in target); r1 = max(y for y, x in target)
    c0 = min(x for y, x in target); c1 = max(x for y, x in target)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    objset = set(target)
    maj = Counter(int(I[y, x]) for (y, x) in target).most_common(1)[0][0]
    marker = [(y, x) for (y, x) in target if I[y, x] != maj]

    ops, sels = [], []

    # 1) Cut the window around the marked object out of the scene
    #    (full bbox rectangle, background corners included).
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])

    # 2) Anything inside that window that is not part of the object (a foreign
    #    object that happened to sit in a concave corner) becomes background.
    foreign = [(y - r0, x - c0) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
               if (y, x) not in objset and I[y, x] != bgc]
    if foreign:
        ops.append(int(bgc)); sels.append(sel_of(foreign))
    # Background cells that were 0-valued are already 0 after the crop; if bgc
    # is 0 they're correct. If bgc != 0, background cells were nonzero and copied.

    # 3) The marker cell takes on the object's own (majority) colour.
    if marker:
        ops.append(int(maj)); sels.append(sel_of([(y - r0, x - c0) for (y, x) in marker]))

    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
