"""
ARC Task: b775ac94 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is an episode-level role; marker/blob colours are free per object.
    return {"bgc": random.choice(list(range(10)))}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def _neighbors8(r, c):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in cols if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(2, 5)
        ow = random.randint(2, 5)
        c1, c2, c3, c4 = random.sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        for _ in range(ncells):
            nb = set()
            for (a, b) in obj:
                nb.update(_neighbors8(a, b))
            cand = sorted((bounds - obj) & nb)
            if not cand:
                break
            obj.add(random.choice(cand))
        gLR = np.full((oh, ow), bgc, dtype=int)
        for (a, b) in obj:
            gLR[a, b] = c1

        def rep(g, a, b):
            g = g.copy()
            g[g == a] = b
            return g

        gLL = rep(np.fliplr(gLR), c1, c2)
        gUR = rep(np.flipud(gLR), c1, c3)
        gUL = rep(np.fliplr(np.flipud(gLR)), c1, c4)
        g = np.vstack([np.hstack([gUL, gUR]), np.hstack([gLL, gLR])])
        g2 = np.full((oh * 2, ow * 2), bgc, dtype=int)
        for (a, b) in obj:
            g2[a + oh, b + ow] = c1
        nkeep = _unifint(diff_lb, diff_ub, (1, 3))
        keepcols = random.sample((c2, c3, c4), nkeep)
        for cc in (c2, c3, c4):
            if cc not in keepcols:
                g = rep(g, cc, bgc)
            else:
                ofsi = -1 if cc in (c3, c4) else 0
                ofsj = -1 if cc in (c2, c4) else 0
                g2[oh + ofsi, ow + ofsj] = cc
        k = random.choice((0, 1, 2, 3))   # DSL rot90 = clockwise
        g = np.rot90(g, -k)
        g2 = np.rot90(g2, -k)
        objo = [(int(g[i, j]), i, j) for i in range(g.shape[0]) for j in range(g.shape[1]) if g[i, j] != bgc]
        obji = [(int(g2[i, j]), i, j) for i in range(g2.shape[0]) for j in range(g2.shape[1]) if g2[i, j] != bgc]
        r0 = min(t[1] for t in objo)
        cc0 = min(t[2] for t in objo)
        objo = [(v, i - r0, j - cc0) for v, i, j in objo]
        obji = [(v, i - r0, j - cc0) for v, i, j in obji]
        ohh = max(t[1] for t in objo) + 1
        oww = max(t[2] for t in objo) + 1
        cands = [(i, j) for (i, j) in inds if i <= h - ohh and j <= w - oww]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        plcdoi = {(i + li, j + lj) for _, i, j in objo}
        if plcdoi.issubset(inds):
            succ += 1
            nb = set()
            for (a, b) in plcdoi:
                nb.update(_neighbors8(a, b))
            inds = (inds - plcdoi) - nb
            for v, i, j in obji:
                gi[i + li, j + lj] = v
            for v, i, j in objo:
                go[i + li, j + lj] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Each object = a multi-cell blob S (one colour) plus 1-3 single-pixel markers
    at one corner of S's bbox. Each marker spawns S mirrored across the bbox edge(s)
    it sits beyond, in the marker's colour. Marker colours within an object are all
    distinct, so each mirrored copy needs its own Color op: one Color per copy
    (selection = the cells the mirrored copy lands on, computed from I alone).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the generator paints a canvas and places sparse objects
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    seen = set()
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or (sr, sc) in seen:
                continue
            comp = []
            dq = deque([(sr, sc)])
            seen.add((sr, sc))
            while dq:
                pr, pc = dq.popleft()
                comp.append((pr, pc))
                for qr, qc in _neighbors8(pr, pc):
                    if 0 <= qr < hi and 0 <= qc < wi and (qr, qc) not in seen and I[qr, qc] != bgc:
                        seen.add((qr, qc))
                        dq.append((qr, qc))
            groups = {}
            for p in comp:
                groups.setdefault(int(I[p]), []).append(p)
            shape_col, S = max(groups.items(), key=lambda kv: (len(kv[1]), -kv[0]))
            if len(S) < 2:
                continue
            r0 = min(p[0] for p in S); r1 = max(p[0] for p in S)
            c0 = min(p[1] for p in S); c1 = max(p[1] for p in S)
            markers = []
            for mcol, cells in groups.items():
                if mcol == shape_col:
                    continue
                for (mr, mc) in cells:
                    dr = -1 if mr == r0 - 1 else (1 if mr == r1 + 1 else 0)
                    dc = -1 if mc == c0 - 1 else (1 if mc == c1 + 1 else 0)
                    if dr == 0 and dc == 0:
                        continue
                    markers.append((dr, dc, mcol))
            markers.sort()
            for dr, dc, mcol in markers:
                copy = set()
                for (a, b) in S:
                    na = a if dr == 0 else (2 * r0 - 1 - a if dr < 0 else 2 * r1 + 1 - a)
                    nb = b if dc == 0 else (2 * c0 - 1 - b if dc < 0 else 2 * c1 + 1 - b)
                    if 0 <= na < hi and 0 <= nb < wi:
                        copy.add((na, nb))
                # only emit if the stamp visibly changes something
                if any(I[p] != mcol for p in copy):
                    ops.append(int(mcol))
                    sels.append(sel_of(sorted(copy)))

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
                        f"num_examples+1 ({num_examples + 1}) for task b775ac94"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b775ac94"
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
                                f"for task b775ac94"
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
                    f"Failed to build a complete episode for task b775ac94 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b775ac94-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
