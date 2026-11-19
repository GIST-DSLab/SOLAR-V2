"""
ARC Task: 045e512c (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = max(lo_i, int(round(hi)))
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, objc)]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "objc": objc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, ccols=None) -> dict:
    cols = list(range(10))
    if bgc is None or objc is None:
        bgc, objc = random.sample(cols, 2)
    if not ccols:
        remcols = [c for c in cols if c not in (bgc, objc)]
        ccols = random.sample(remcols, random.randint(1, 8))
    max_h = max(11, min(30, max_h))
    max_w = max(11, min(30, max_w))
    for _attempt in range(200):
        h = _unifint(diff_lb, diff_ub, (11, max_h))
        w = _unifint(diff_lb, diff_ub, (11, max_w))
        # build a connected (8-conn) shape touching all 4 sides of its oh x ow box
        while True:
            oh = _unifint(diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
            ow = _unifint(diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            c1 = (random.randint(0, oh - 1), 0)
            c2 = (0, random.randint(0, ow - 1))
            c3 = (oh - 1, random.randint(0, ow - 1))
            c4 = (random.randint(0, oh - 1), ow - 1)
            obj = {c1, c2, c3, c4}
            remc = sorted(bounds - obj)
            ncells = _unifint(diff_lb, diff_ub, (0, len(remc)))
            for loc in random.sample(remc, ncells):
                obj.add(loc)
            # 8-connectivity check
            start = next(iter(obj))
            seen = {start}
            stack = [start]
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        n = (y + dy, x + dx)
                        if n in obj and n not in seen:
                            seen.add(n)
                            stack.append(n)
            if len(seen) == len(obj):
                break
        if h - 2 * oh - 1 < oh + 1 or w - 2 * ow - 1 < ow + 1:
            continue
        loci = random.randint(oh + 1, h - 2 * oh - 1)
        locj = random.randint(ow + 1, w - 2 * ow - 1)
        gi = np.full((h, w), bgc, dtype=int)
        go = np.full((h, w), bgc, dtype=int)
        sobj = [(i + loci, j + locj) for i, j in obj]
        for i, j in sobj:
            gi[i, j] = objc
            go[i, j] = objc
        options = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        ndirs = _unifint(diff_lb, diff_ub, (1, 8))
        dirs = random.sample(options, ndirs)
        dcols = [random.choice(ccols) for _ in range(ndirs)]
        hb = {loci - 2, loci + oh + 1}
        vb = {locj - 2, locj + ow + 1}
        sr, sc = oh + 1, ow + 1
        for (dr, dc), col in zip(dirs, dcols):
            ind = [(i + dr * sr, j + dc * sc) for i, j in sobj]
            ind = [(i, j) for i, j in ind if i in hb or j in vb]
            if not ind:
                continue
            nindsd = _unifint(diff_lb, diff_ub, (0, len(ind) - 1))
            ninds = len(ind) - nindsd
            ind = random.sample(ind, ninds)
            if 0 < len(ind) < len(sobj):
                for i, j in ind:
                    if 0 <= i < h and 0 <= j < w:
                        gi[i, j] = col
                for k in range(1, 10):
                    for i, j in sobj:
                        y, x = i + k * dr * sr, j + k * dc * sc
                        if 0 <= y < h and 0 <= x < w:
                            go[y, x] = col
        if np.array_equal(gi, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    """Reading (shared by the maker and the golf entry, which carve it the same way):
       the master shape is a stamp; each indicator one step away (step = shape size + 1
       gap) is a partial stamp whose colour names a live direction. Complete that stamp
       and keep stamping the master shape outward along the direction, one full
       stamp (= one Color op on the master footprint shifted k steps) per copy.
       Everything is read from I: the shape, the step, the live directions and colours.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the canvas colour every example is drawn on
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for ei, _eo in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        cand = cnt.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected same-colour components
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and I[ny, nx] == col:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
            comps.append(comp)

    ops, sels = [], []
    if comps:
        master = max(comps, key=len)
        rs = [p[0] for p in master]
        cs = [p[1] for p in master]
        oh = max(rs) - min(rs) + 1
        ow = max(cs) - min(cs) + 1
        sr, sc = oh + 1, ow + 1
        K = max(h // oh, w // ow) + 1

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                # indicator = partial stamp one step away; its colour marks the ray
                probe = [(y + dr * sr, x + dc * sc) for y, x in master]
                vals = [int(I[y, x]) for y, x in probe
                        if 0 <= y < h and 0 <= x < w and I[y, x] != bgc]
                if not vals:
                    continue
                col = Counter(vals).most_common(1)[0][0]
                # stamp the full master footprint at every step outward
                for k in range(1, K + 1):
                    cells = [(y + k * dr * sr, x + k * dc * sc) for y, x in master]
                    cells = [(y, x) for y, x in cells if 0 <= y < h and 0 <= x < w]
                    if not cells:
                        break
                    ops.append(int(col))
                    sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 045e512c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 045e512c"
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
                                f"for task 045e512c"
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
                    f"Failed to build a complete episode for task 045e512c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"045e512c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
