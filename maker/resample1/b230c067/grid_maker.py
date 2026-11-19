"""
ARC Task: b230c067 (RE-ARC) — LLM-generated grid_maker
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
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, int(round(a + (b - a) * d))), b)


def _neighbors8(ij):
    i, j = ij
    return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)}


def _neighbors_set(cells):
    out = set()
    for c in cells:
        out |= _neighbors8(c)
    return out


def _normalize(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _hw(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    while True:
        h = _unifint(diff_lb, diff_ub, (10, max_h))
        w = _unifint(diff_lb, diff_ub, (10, max_w))
        oh = _unifint(diff_lb, diff_ub, (2, h // 3 - 1))
        ow = _unifint(diff_lb, diff_ub, (2, w // 3 - 1))
        numcd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = random.choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        sp = random.choice(sorted(bounds))
        shp = {sp}
        for _ in range(numca):
            cands = sorted((bounds - shp) & _neighbors_set(shp))
            if not cands:
                break
            shp.add(random.choice(cands))
        shpa = set(_normalize(shp))
        shpb = set(shpa)
        mxnch = oh * ow - len(shpa)
        if mxnch < 1:
            continue
        nchinv = _unifint(diff_lb, diff_ub, (1, mxnch))
        nch = mxnch - nchinv
        nch = min(max(1, nch), mxnch)
        for _ in range(nch):
            cands = sorted((bounds - shpb) & _neighbors_set(shpb))
            if not cands:
                break
            shpb.add(random.choice(cands))
        shpb = set(_normalize(shpb))
        if shpa == shpb:
            continue
        if random.choice((True, False)):
            shpa, shpb = shpb, shpa
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        ha, wa = _hw(shpa)
        acands = [(i, j) for i in range(h) for j in range(w) if i <= h - ha and j <= w - wa]
        ai, aj = random.choice(acands)
        aplcd = {(r + ai, c + aj) for r, c in shpa}
        for r, c in aplcd:
            gi[r][c] = fgc
            go[r][c] = 2
        inds = {(i, j) for i in range(h) for j in range(w)}
        inds = (inds - aplcd) - _neighbors_set(aplcd)
        hb, wb = _hw(shpb)
        inds = {(i, j) for (i, j) in inds if i <= h - hb and j <= w - wb}
        succ, tr = 0, 0
        while succ < 2 and tr <= 10:
            if not inds:
                break
            li, lj = random.choice(sorted(inds))
            plcbd = {(r + li, c + lj) for r, c in shpb}
            if plcbd.issubset(inds):
                for r, c in plcbd:
                    gi[r][c] = fgc
                    go[r][c] = 1
                succ += 1
                inds = (inds - plcbd) - _neighbors_set(plcbd)
            tr += 1
        if succ == 2:
            break
    return {"input": gi, "output": go}


def _objects8(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            objs.append(sorted(cells))
    return objs


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: generator paints the canvas with bgc; objects are small -> majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    objs = _objects8(I, bgc)
    shapes = [_normalize(o) for o in objs]
    counts = Counter(shapes)
    # odd one out: object whose shape occurs least often
    odd_idx = min(range(len(objs)), key=lambda k: counts[shapes[k]])
    ops, sels = [], []
    ops.append(2); sels.append(sel_of(objs[odd_idx]))
    for k, o in enumerate(objs):
        if k == odd_idx:
            continue
        ops.append(1); sels.append(sel_of(o))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b230c067"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b230c067"
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
                                f"for task b230c067"
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
                    f"Failed to build a complete episode for task b230c067 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b230c067-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
