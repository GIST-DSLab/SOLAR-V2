"""
ARC Task: 72ca375d (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    return a + round(random.uniform(lb, ub) * (b - a)) if b > a else a


def _nb(cell):
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def _mnb(cells):
    s = set()
    for x in cells:
        s |= _nb(x)
    return s


def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return {(r - mr, c - mc) for r, c in cells}


def _vm(cells):
    mc = min(c for _, c in cells)
    xc = max(c for _, c in cells)
    return {(r, mc + xc - c) for r, c in cells}


def _comps(cells):
    cells = set(cells)
    out = []
    while cells:
        s = cells.pop()
        comp = {s}
        st = [s]
        while st:
            x = st.pop()
            for n in _nb(x):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; the symmetric object's
    # colour and distractor colours are irrelevant to the rule.
    return {"bgc": random.randrange(10)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = _unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    sh = _unifint(diff_lb, diff_ub, (2, 8))
    sw = _unifint(diff_lb, diff_ub, (1, 4))
    bnds = {(i, j) for i in range(sh) for j in range(sw)}
    sp = (random.randint(0, sh - 1), sw - 1)
    src = {sp}
    bnds.discard(sp)
    ncd = _unifint(diff_lb, diff_ub, (0, (sh * sw) // 2))
    n1 = random.choice((ncd, sh * sw - ncd))
    n2 = _unifint(diff_lb, diff_ub, (1, sh * sw))
    nc = min(max(1, (n1 + n2) // 2), sh * sw, (h * w) // 2 - 1)
    for _ in range(nc - 1):
        cand = sorted((bnds - src) & _mnb(src))
        if not cand:
            break
        src.add(random.choice(cand))
    src = _norm(src)
    wd = max(c for _, c in src) + 1
    src = src | {(r, c + wd) for r, c in _vm(src)}
    oh = max(r for r, _ in src) + 1
    ow = max(c for _, c in src) + 1
    remcols = [c for c in cols if c != bgc]
    trgc = random.choice(remcols)
    go = np.full((oh, ow), bgc, dtype=int)
    for r, c in src:
        go[r, c] = trgc
    if oh > h or ow > w:
        raise ValueError("object too big")
    li = random.randint(0, h - oh)
    lj = random.randint(0, w - ow)
    gi = np.full((h, w), bgc, dtype=int)
    shftd = {(r + li, c + lj) for r, c in src}
    for r, c in shftd:
        gi[r, c] = trgc
    inds = {(i, j) for i in range(h) for j in range(w)}
    inds = (inds - shftd) - _mnb(shftd)
    tr = 0
    succ = 0
    remc = {(i, j) for i in range(8) for j in range(8)} - src
    while succ < nobjs and tr <= 4 * nobjs and inds:
        for _att in range(200):
            new = set(src)
            npert = 64 - _unifint(diff_lb, diff_ub, (1, 63))
            for _ in range(npert):
                isadd = random.choice((True, False))
                if isadd and len(new) < 64:
                    cd = sorted((remc - new) & _mnb(new))
                    if not cd:
                        break
                    new.add(random.choice(cd))
                if not isadd and len(new) > 2:
                    new.discard(random.choice(sorted(new)))
            new = _norm(max(_comps(new), key=len))
            if new != _vm(new):
                break
        else:
            tr += 1
            continue
        col = random.choice(remcols)
        lc = [p for p in inds if all((r + p[0], c + p[1]) in inds for r, c in new)]
        if not lc:
            tr += 1
            continue
        p = random.choice(lc)
        new = {(r + p[0], c + p[1]) for r, c in new}
        for r, c in new:
            gi[r, c] = col
        succ += 1
        inds = (inds - new) - _mnb(new)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Background: fixed per episode, dominant across all example inputs plus I.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # Find the single left-right symmetric (8-connected, univalued) object in I.
    target = None
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col}
        for comp in _comps(cells):
            if comp == _vm(comp):
                target = comp
                break
        if target:
            break

    r0 = min(r for r, _ in target); r1 = max(r for r, _ in target)
    c0 = min(c for _, c in target); c1 = max(c for _, c in target)

    ops, sels = [], []
    # Crop the canvas to the symmetric object's full bounding rectangle
    # (the intended region IS that whole rectangle, background included).
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])

    # Any other object's cells that fell inside that rectangle are not part of
    # the symmetric object: paint them background (visible from I alone).
    foreign = [(r - r0, c - c0) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
               if (r, c) not in target and I[r, c] != bgc and I[r, c] != 0]
    if foreign:
        ops.append(bgc); sels.append(sel_of(foreign))

    ops.append(34); sels.append([0, 0, r1 - r0, c1 - c0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
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
                                f"for task 72ca375d"
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
                    f"Failed to build a complete episode for task 72ca375d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72ca375d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
