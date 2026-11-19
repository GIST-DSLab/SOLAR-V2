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


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dmyc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dmyc is None:
        dmyc = choice(remove(bgc, cols))

    hub = max(8, min(30, max_h))
    wub = max(8, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (8, hub))
    w = unifint(diff_lb, diff_ub, (8, wub))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)

    remcols = remove(bgc, cols)
    remcols = remove(dmyc, remcols)

    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)
    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the colour the generator fills the canvas with (majority colour here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- find the 4-connected non-background components of I -------------
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] != bgc and not seen[r0, c0]:
                stack = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(sorted(cells))

    if not comps:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # the multi-coloured component is the template; the mono-coloured ones are the copies
    tmpl = max(comps, key=lambda cs: len({int(I[r, c]) for r, c in cs}))
    if len({int(I[r, c]) for r, c in tmpl}) < 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    dests = [cs for cs in comps if cs is not tmpl]
    if not dests:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    tr = min(r for r, c in tmpl); br = max(r for r, c in tmpl)
    tc = min(c for r, c in tmpl); bc = max(c for r, c in tmpl)
    oh, ow = br - tr + 1, bc - tc + 1
    pattern = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}

    # ---- grab the template once ------------------------------------------
    # bbox selection is intended here: CopyI takes the whole template rectangle
    ops.append(28); sels.append([tr, tc, oh - 1, ow - 1])
    clip = I[tr:tr + oh, tc:tc + ow].copy()
    clip_nz = clip != 0

    sim = I.copy()
    dests.sort(key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))
    dest_origins = []

    def repair(cells_origin):
        """paint this copy's cells that do not yet hold their template colour"""
        dr0, dc0 = cells_origin
        buckets = {}
        for (ddr, ddc), col in pattern.items():
            rr, cc = dr0 + ddr, dc0 + ddc
            if 0 <= rr < h and 0 <= cc < w and int(sim[rr, cc]) != col:
                buckets.setdefault(col, []).append((rr, cc))
        for col in sorted(buckets):
            cells = sorted(buckets[col])
            ops.append(int(col)); sels.append(sel_of(cells))
            for rr, cc in cells:
                sim[rr, cc] = col

    for cs in dests:
        dr0 = min(r for r, c in cs)
        dc0 = min(c for r, c in cs)
        dest_origins.append((dr0, dc0))
        # stamp the template at this copy's upper-left corner
        ops.append(30); sels.append(sel_of([(dr0, dc0)]))
        rh = min(oh, h - dr0)
        rw = min(ow, w - dc0)
        region = sim[dr0:dr0 + rh, dc0:dc0 + rw]
        m = clip_nz[:rh, :rw]
        region[m] = clip[:rh, :rw][m]
        # Paste is transparent for colour 0, so any 0-cells of the template
        # (and anything else still off) are painted explicitly for this copy
        repair((dr0, dc0))

    # a later stamp may have covered a cell of an earlier copy (or of the
    # template) that sat inside its bounding box - restore those objects
    for origin in dest_origins:
        repair(origin)
    buckets = {}
    for r, c in tmpl:
        if int(sim[r, c]) != int(I[r, c]):
            buckets.setdefault(int(I[r, c]), []).append((r, c))
    for col in sorted(buckets):
        cells = sorted(buckets[col])
        ops.append(int(col)); sels.append(sel_of(cells))
        for rr, cc in cells:
            sim[rr, cc] = col

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
