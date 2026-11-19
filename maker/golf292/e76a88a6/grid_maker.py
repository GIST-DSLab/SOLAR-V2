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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dmyc = random.choice(rem)
    return {"bgc": bgc, "dmyc": dmyc}


def _uni(diff_lb, diff_ub, lo, hi):
    hi = max(lo, hi)
    d = random.uniform(diff_lb, diff_ub)
    return int(round(lo + d * (hi - lo)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None or dmyc == bgc:
        dmyc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    h = _uni(diff_lb, diff_ub, 8, min(30, max_h))
    w = _uni(diff_lb, diff_ub, 8, min(30, max_w))
    objh = _uni(diff_lb, diff_ub, 2, 5)
    objw = _uni(diff_lb, diff_ub, 2, 5)
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = _uni(diff_lb, diff_ub, 2, len(bounds) - 2)
    for _ in range(nc):
        nb = set()
        for (i, j) in shp:
            nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
        shp.add(random.choice(sorted((bounds - shp) & nb)))
    mi = min(i for i, j in shp); mj = min(j for i, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, j in shp) + 1
    ow = max(j for i, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    numco = _uni(diff_lb, diff_ub, 2, 8)
    colll = random.sample(remcols, numco)
    while True:
        pat = {ij: random.choice(colll) for ij in shp}
        if len(set(pat.values())) > 1:
            break
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), c in pat.items():
        gi[loci + i, locj + j] = c
    go = gi.copy()

    def dn(cells):
        out = set()
        for (i, j) in cells:
            out |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
        return out

    shpp = {(loci + i, locj + j) for i, j in shp}
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = _uni(diff_lb, diff_ub, 1, ub)
    allidx = {(i, j) for i in range(h) for j in range(w)}
    idns = (allidx - shpp) - dn(shpp)
    idns = {ij for ij in idns if ij[0] <= h - oh and ij[1] <= w - ow}
    cnt = fails = 0
    maxfails = 5 * numlocs
    while cnt < numlocs and fails < maxfails and idns:
        loc = random.choice(sorted(idns))
        plcd = {(loc[0] + i, loc[1] + j) for i, j in shp}
        if plcd <= idns:
            for (i, j) in shp:
                go[loc[0] + i, loc[1] + j] = pat[(i, j)]
                gi[loc[0] + i, loc[1] + j] = dmyc
            cnt += 1
            idns = (idns - plcd) - dn(plcd)
        else:
            fails += 1
    if cnt == 0:
        raise ValueError("no copies placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the colour every episode input is painted on
    pool = I.flatten().tolist()
    for ex in (examples or []):
        pool += np.asarray(ex[0], dtype=int).flatten().tolist()
    bgc = Counter(pool).most_common(1)[0][0]

    # 4-connected non-background objects (multicolour allowed)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] != bgc and not seen[r0, c0]:
                stack = [(r0, c0)]; seen[r0, c0] = True; cells = []
                while stack:
                    r, c = stack.pop(); cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True; stack.append((nr, nc))
                comps.append(sorted(cells))

    def ncol(cs):
        return len({int(I[r, c]) for r, c in cs})

    full = [0, 0, h - 1, w - 1]
    if not comps:
        return [34], [full]
    tmpl = max(comps, key=ncol)
    dests = [cs for cs in comps if cs is not tmpl and ncol(cs) == 1]
    if ncol(tmpl) < 2 or not dests:
        return [34], [full]

    # copy exactly the template object's cells (CopyI honours the mask, so the
    # background inside its bounding box is not taken along)
    ops.append(28); sels.append(sel_of(tmpl))
    # stamp it on each dummy object's upper-left corner
    dests.sort(key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))
    for cs in dests:
        r0 = min(r for r, c in cs); c0 = min(c for r, c in cs)
        ops.append(30); sels.append(sel_of([(r0, c0)]))

    # Paste cannot write colour 0: template cells of colour 0 are still the
    # dummy colour; recolour every remaining dummy-coloured cell to 0.
    if any(int(I[r, c]) == 0 for r, c in tmpl):
        dmy = {int(I[cs[0][0], cs[0][1]]) for cs in dests}
        grid = I.copy()
        tr = min(r for r, c in tmpl); tc = min(c for r, c in tmpl)
        for cs in dests:
            r0 = min(r for r, c in cs); c0 = min(c for r, c in cs)
            for r, c in tmpl:
                v = int(I[r, c])
                if v:
                    rr, cc = r - tr + r0, c - tc + c0
                    if 0 <= rr < h and 0 <= cc < w:
                        grid[rr, cc] = v
        left = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) in dmy]
        if left:
            ops.append(0); sels.append(sel_of(left))

    ops.append(34); sels.append(full)
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
