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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, int(round(a + (b - a) * d))), b)


def _nbrs8(ij):
    i, j = ij
    return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)}


def _mapply_nbrs(cells):
    out = set()
    for ij in cells:
        out |= _nbrs8(ij)
    return out


def _normalize(cells):
    mr = min(i for i, _ in cells)
    mc = min(j for _, j in cells)
    return {(i - mr, j - mc) for i, j in cells}


def _height(cells):
    return max(i for i, _ in cells) - min(i for i, _ in cells) + 1


def _width(cells):
    return max(j for _, j in cells) - min(j for _, j in cells) + 1


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c not in (1, 2)]
        bgc, fgc = random.sample(cols, 2)
    while True:
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        oh = _unifint(diff_lb, diff_ub, (2, max(2, h // 3 - 1)))
        ow = _unifint(diff_lb, diff_ub, (2, max(2, w // 3 - 1)))
        numcd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = random.choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        sp = random.choice(sorted(bounds))
        shp = {sp}
        ok = True
        for _ in range(numca):
            cands = sorted((bounds - shp) & _mapply_nbrs(shp))
            if not cands:
                ok = False
                break
            shp.add(random.choice(cands))
        if not ok:
            continue
        shpa = _normalize(shp)
        shpb = set(_normalize(shp))
        mxnch = oh * ow - len(shpa)
        if mxnch < 1:
            continue
        nchinv = _unifint(diff_lb, diff_ub, (1, mxnch))
        nch = min(max(1, mxnch - nchinv), mxnch)
        for _ in range(nch):
            cands = sorted((bounds - shpb) & _mapply_nbrs(shpb))
            if not cands:
                ok = False
                break
            shpb.add(random.choice(cands))
        if not ok:
            continue
        if random.choice((True, False)):
            shpa, shpb = shpb, shpa
        gi = np.full((h, w), bgc, dtype=int)
        go = np.full((h, w), bgc, dtype=int)
        inds = {(i, j) for i in range(h) for j in range(w)}
        ha, wa = _height(shpa), _width(shpa)
        acands = sorted(ij for ij in inds if ij[0] <= h - ha and ij[1] <= w - wa)
        if not acands:
            continue
        ai, aj = random.choice(acands)
        aplcd = {(i + ai, j + aj) for i, j in shpa}
        for i, j in aplcd:
            gi[i, j] = fgc
            go[i, j] = 2
        inds = (inds - aplcd) - _mapply_nbrs(aplcd)
        hb, wb = _height(shpb), _width(shpb)
        inds = {ij for ij in inds if ij[0] <= h - hb and ij[1] <= w - wb}
        succ, tr = 0, 0
        while succ < 2 and tr <= 10:
            if not inds:
                break
            li, lj = random.choice(sorted(inds))
            plcbd = {(i + li, j + lj) for i, j in shpb}
            if plcbd.issubset(inds):
                for i, j in plcbd:
                    gi[i, j] = fgc
                    go[i, j] = 1
                succ += 1
                inds = (inds - plcbd) - _mapply_nbrs(plcbd)
            tr += 1
        if succ == 2:
            break
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background is the canvas colour; objects are small
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected components of foreground
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            comps.append(cells)

    def key(cells):
        mr = min(y for y, _ in cells)
        mc = min(x for _, x in cells)
        return tuple(sorted((y - mr, x - mc) for y, x in cells))

    groups = {}
    for cells in comps:
        groups.setdefault(key(cells), []).append(cells)
    rare = min(groups, key=lambda k: len(groups[k]))

    ops, sels = [], []
    # 1) every foreground cell (all non-background cells) -> colour 1
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    ops.append(1); sels.append(sel_of(fg))
    # 2) the odd-shaped object(s) -> colour 2
    for cells in groups[rare]:
        ops.append(2); sels.append(sel_of(cells))

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
