"""
ARC Task: e76a88a6 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dmyc = random.choice(rem)
    return {"bgc": bgc, "dmyc": dmyc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * d))
    lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _dnb(r, c):
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None or dmyc == bgc:
        dmyc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = _unifint(diff_lb, diff_ub, (2, 5))
    objw = _unifint(diff_lb, diff_ub, (2, 5))
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = _unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for _ in range(nc):
        nb = set()
        for (i, j) in shp:
            nb |= _dnb(i, j)
        cand = sorted((bounds - shp) & nb)
        shp.add(random.choice(cand))
    mi = min(i for i, j in shp); mj = min(j for i, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, j in shp) + 1
    ow = max(j for i, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(i + loci, j + locj) for i, j in shp}
    numco = _unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(remcols, numco)
    while True:
        shppc = {ij: random.choice(colll) for ij in shpp}
        if len(set(shppc.values())) > 1:
            break
    shppcn = {(i - loci, j - locj): v for (i, j), v in shppc.items()}
    gi = [[bgc] * w for _ in range(h)]
    for (i, j), v in shppc.items():
        gi[i][j] = v
    go = [row[:] for row in gi]
    ub = max(1, int((h * w) / (oh * ow)) // 2)
    numlocs = _unifint(diff_lb, diff_ub, (1, ub))
    cnt = fails = 0
    maxfails = 5 * numlocs
    allidx = {(i, j) for i in range(h) for j in range(w)}
    nbs = set()
    for (i, j) in shpp:
        nbs |= _dnb(i, j)
    idns = (allidx - shpp) - nbs
    idns = {(i, j) for (i, j) in idns if i <= h - oh and j <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        li, lj = random.choice(sorted(idns))
        plcd = {(i + li, j + lj): v for (i, j), v in shppcn.items()}
        plcdi = set(plcd)
        if plcdi.issubset(idns):
            for (i, j), v in plcd.items():
                go[i][j] = v
                gi[i][j] = dmyc
            cnt += 1
            nb2 = set()
            for (i, j) in plcdi:
                nb2 |= _dnb(i, j)
            idns = (idns - plcdi) - nb2
        else:
            fails += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the canvas colour, shared by every input of the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            cnt.update(ei.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 4-connected non-background objects of I
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] != bgc and not seen[r0, c0]:
                stack = [(r0, c0)]; seen[r0, c0] = True; cells = []
                while stack:
                    rr, cc = stack.pop(); cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True; stack.append((nr, nc))
                comps.append(sorted(cells))

    def ncol(cs):
        return len({int(I[r, c]) for r, c in cs})

    if not comps or max(ncol(cs) for cs in comps) < 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # the multi-coloured object is the template; every mono-coloured object is a
    # placeholder silhouette of it.  Reading: each placeholder cell takes the
    # template colour found at the same offset from the object's upper-left corner.
    tmpl = max(comps, key=ncol)
    tr = min(r for r, c in tmpl); tc = min(c for r, c in tmpl)
    pattern = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}

    holders = [cs for cs in comps if cs is not tmpl and ncol(cs) == 1]
    holders.sort(key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))

    for cs in holders:
        r0 = min(r for r, c in cs); c0 = min(c for r, c in cs)
        # recolour this placeholder layer by layer: one Color op per template colour,
        # selecting the placeholder cells whose offset holds that colour in the template
        layers = {}
        for (r, c) in cs:
            col = pattern.get((r - r0, c - c0))
            if col is None or col == int(I[r, c]):
                continue
            layers.setdefault(col, []).append((r, c))
        for col in sorted(layers):
            ops.append(int(col)); sels.append(sel_of(sorted(layers[col])))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
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
                                f"for task e76a88a6"
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
                    f"Failed to build a complete episode for task e76a88a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e76a88a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
