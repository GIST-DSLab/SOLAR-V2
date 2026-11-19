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
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + round((b - a) * lb)
    hi = a + round((b - a) * ub)
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _dn(ij):
    i, j = ij
    return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None or dmyc == bgc:
        dmyc = random.choice([c for c in cols if c != bgc])
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = _unifint(diff_lb, diff_ub, (2, 5))
    objw = _unifint(diff_lb, diff_ub, (2, 5))
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = _unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for _ in range(nc):
        nb = set()
        for ij in shp:
            nb |= _dn(ij)
        shp.add(random.choice(sorted((bounds - shp) & nb)))
    mi = min(i for i, j in shp); mj = min(j for i, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    oh = max(i for i, j in shp) + 1
    ow = max(j for i, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(i + loci, j + locj) for i, j in shp}
    numco = _unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(remcols, numco)
    while True:
        colmap = {ij: random.choice(colll) for ij in shp}
        if len(set(colmap.values())) > 1:
            break
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), c in colmap.items():
        gi[i + loci, j + locj] = c
    go = gi.copy()
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = _unifint(diff_lb, diff_ub, (1, ub))
    cnt = fails = 0
    maxfails = 5 * numlocs
    nb = set()
    for ij in shpp:
        nb |= _dn(ij)
    idns = {(i, j) for i in range(h) for j in range(w)} - shpp - nb
    idns = {(i, j) for i, j in idns if i <= h - oh and j <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        loc = random.choice(sorted(idns))
        plcdi = {(i + loc[0], j + loc[1]) for i, j in shp}
        if plcdi <= idns:
            for (i, j), c in colmap.items():
                go[i + loc[0], j + loc[1]] = c
                gi[i + loc[0], j + loc[1]] = dmyc
            cnt += 1
            nb = set()
            for ij in plcdi:
                nb |= _dn(ij)
            idns = idns - plcdi - nb
        else:
            fails += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]
    # background: the canvas colour the generator paints first (majority colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 4-connected non-background components (multi-coloured allowed)
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
    if not comps:
        return [34], [full]
    ncol = lambda cs: len({int(I[r, c]) for r, c in cs})
    tmpl = max(comps, key=ncol)                  # the multi-coloured template
    dests = [cs for cs in comps if cs is not tmpl]  # mono-coloured placeholders
    if ncol(tmpl) < 2 or not dests:
        return [34], [full]
    tr = min(r for r, c in tmpl); tc = min(c for r, c in tmpl)
    rel = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}
    zero_rel = sorted(p for p, col in rel.items() if col == 0)

    # Reading: the template block is copied onto every placeholder block.
    # CopyI with the template's exact cells: the clipboard holds the template object
    # only (no surrounding background), framed by the template's bbox.
    ops.append(28); sels.append(sel_of(tmpl))
    grid = I.copy()
    dests.sort(key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))
    for cs in dests:
        dr0 = min(r for r, c in cs); dc0 = min(c for r, c in cs)
        # stamp the template onto this placeholder (same shape, same corner)
        ops.append(30); sels.append(sel_of([(dr0, dc0)]))
        for (a, b), col in rel.items():
            if col != 0 and 0 <= dr0 + a < h and 0 <= dc0 + b < w:
                grid[dr0 + a, dc0 + b] = col
        # template cells of colour 0 are transparent to Paste: paint them on this copy
        zc = [(dr0 + a, dc0 + b) for a, b in zero_rel
              if 0 <= dr0 + a < h and 0 <= dc0 + b < w and grid[dr0 + a, dc0 + b] != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))
            for r, c in zc:
                grid[r, c] = 0
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
