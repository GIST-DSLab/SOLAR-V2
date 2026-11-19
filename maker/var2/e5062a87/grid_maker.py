"""
ARC Task: e5062a87 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    ilb = a + int((b - a) * diff_lb)
    iub = a + int((b - a) * diff_ub)
    ilb = max(a, min(b, ilb))
    iub = max(a, min(b, iub))
    if iub < ilb:
        ilb, iub = iub, ilb
    return random.randint(ilb, iub)


def _norm_cells(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return sorted((r - mr, c - mc) for r, c in cells)


# ------------------------------------------------------------ 1. colors ----
def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly two colours out of 1..9:
    #   eligcol -> the canvas the picture is drawn on,
    #   objc    -> the colour of the one marked template.
    # 0 is NOT sampled: it is hardcoded as the colour of the noise / of the
    # hidden copies, so neither role may take it (a 0 marker would be
    # indistinguishable from a hole and the task would have no rule).
    cols = list(range(1, 10))
    eligcol, objc = random.sample(cols, 2)
    return {"eligcol": eligcol, "objc": objc}


# ----------------------------------------------------------- 2. generate ---
def generate(diff_lb, diff_ub, max_h, max_w, eligcol, objc) -> dict:
    h = _unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(10, max_w), max_w))

    # --- the template: a random 8-connected blob of 3..9 cells -------------
    ncells = _unifint(diff_lb, diff_ub, (3, 9))
    obj = {(0, 0)}
    while len(obj) < ncells:
        cand = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        cand.add((r + dr, c + dc))
        cand -= obj
        obj.add(random.choice(sorted(cand)))
    obj = _norm_cells(obj)
    oh = max(r for r, _ in obj) + 1
    ow = max(c for _, c in obj) + 1
    if oh > h or ow > w:
        raise ValueError("template does not fit")

    # --- canvas + 20..50% noise of colour 0 --------------------------------
    gi = np.full((h, w), eligcol, dtype=int)
    area = h * w
    nnoise = _unifint(diff_lb, diff_ub, (int(0.2 * area), int(0.5 * area)))
    nnoise = max(1, min(area, nnoise))
    allidx = [(r, c) for r in range(h) for c in range(w)]
    for (r, c) in random.sample(allidx, nnoise):
        gi[r, c] = 0

    # --- place noccs copies: all hidden in 0, the last one drawn in objc ---
    noccs = _unifint(diff_lb, diff_ub, (2, max(2, area // (len(obj) * 3))))
    for k in range(noccs):
        i0 = random.randint(0, h - oh)
        j0 = random.randint(0, w - ow)
        v = objc if k == noccs - 1 else 0
        for (r, c) in obj:
            gi[i0 + r, j0 + c] = v

    # --- output: every all-zero placement of the template becomes objc -----
    go = gi.copy()
    nfound = 0
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            if all(gi[i + r, j + c] == 0 for r, c in obj):
                nfound += 1
                for (r, c) in obj:
                    go[i + r, j + c] = objc
    if nfound == 0:
        # the marker swallowed the only hidden copy -> nothing to demonstrate
        raise ValueError("no hidden copy in this draw")

    return {"input": gi.tolist(), "output": go.tolist()}


# ------------------------------------------------- 3. derive_operations ----
def _template_colour(I, examples):
    """The marked template is the rarest colour of the input.

    The demonstrations only break a tie: they are drawn with the same palette
    and every one of them paints with that same colour.
    """
    counts = {}
    for v in I.flatten().tolist():
        counts[int(v)] = counts.get(int(v), 0) + 1
    least = min(counts.values())
    cands = sorted(c for c, n in counts.items() if n == least)
    if len(cands) > 1 and examples:
        for pair in examples:
            try:
                ei = np.asarray(pair[0], dtype=int)
                eo = np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            if ei.shape != eo.shape:
                continue
            added = sorted(set(eo[ei != eo].tolist()))
            if len(added) == 1 and added[0] in cands:
                return added[0]
    return cands[0]


def _reading_route(I, mr, mc):
    """Which way to work through the hidden copies.

    A person starts from the copy they already understand -- the marked one --
    and sweeps away from it: along columns when the marker sits near a side
    edge, along rows when it sits near the top or the bottom, and outwards in
    rings when it sits in the middle of the picture.
    """
    hi, wi = I.shape
    span_r = max(1.0, (hi - 1) / 2.0)
    span_c = max(1.0, (wi - 1) / 2.0)
    near_h = min(mr, hi - 1 - mr) / span_r      # 0 at an edge, 1 in the middle
    near_v = min(mc, wi - 1 - mc) / span_c
    if near_h > 0.5 and near_v > 0.5:
        return "rings"
    if near_v < near_h:
        return "cols_lr" if mc <= (wi - 1) / 2.0 else "cols_rl"
    return "rows_td" if mr <= (hi - 1) / 2.0 else "rows_bu"


def _route_key(route, mr, mc):
    if route == "rings":
        return lambda p: (max(abs(p[0] - mr), abs(p[1] - mc)), p[0], p[1])
    if route == "rows_td":
        return lambda p: (p[0], p[1])
    if route == "rows_bu":
        return lambda p: (-p[0], p[1])
    if route == "cols_lr":
        return lambda p: (p[1], p[0])
    return lambda p: (-p[1], p[0])


def _drop_redundant(items):
    """Keep only placements that own at least one cell of their own.

    Two hidden copies may overlap; if one of them is entirely covered by the
    others, bringing it out changes nothing the others do not already change,
    so it is not a step of the route at all.
    """
    items = list(items)
    changed = True
    while changed:
        changed = False
        for k in range(len(items)):
            others = set()
            for m, cells in enumerate(items):
                if m != k:
                    others.update(cells)
            if set(items[k]) <= others:
                items.pop(k)
                changed = True
                break
    return items


def derive_operations(I, O, examples=None):
    """
    Rule, read off the input and the demonstrations only:
      one small shape is drawn in the template colour; everywhere that same
      shape lies hidden in the picture as a hole of 0s, it is brought out in
      the template colour.

    The route through those hidden copies is read off the input too: where the
    marked template sits decides the sweep, and the template's own box, its
    colour and the number of copies decide whether the copies are stamped from
    the template, taken cluster by cluster, or painted one at a time.  Nothing
    here looks at O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    objc = _template_colour(I, examples)
    marker = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    if not marker:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    rmin = min(r for r, _ in marker)
    cmin = min(c for _, c in marker)
    rmax = max(r for r, _ in marker)
    cmax = max(c for _, c in marker)
    norm = _norm_cells(marker)
    oh = max(dr for dr, _ in norm) + 1
    ow = max(dc for _, dc in norm) + 1

    # every placement where the same shape lies hidden as a hole of 0s
    occs = []
    for i in range(hi - oh + 1):
        for j in range(wi - ow + 1):
            if all(I[i + dr, j + dc] == 0 for dr, dc in norm):
                occs.append((i, j))

    # --- the order a solver walks them in, decided by the marker's place ---
    mr = (rmin + rmax) / 2.0
    mc = (cmin + cmax) / 2.0
    route = _reading_route(I, mr, mc)
    occs.sort(key=_route_key(route, mr, mc))

    cells_of = {p: [(p[0] + dr, p[1] + dc) for dr, dc in norm] for p in occs}
    kept = _drop_redundant([cells_of[p] for p in occs])

    # The marked template can be picked up as a stamp only if its bounding box
    # holds nothing but the template itself: every other cell of the box has to
    # be 0, because 0 is the one colour a clipboard does not carry (so those
    # cells travel as nothing and the paste lays down exactly the shape).  For
    # the same reason a template drawn in colour 0 can never be stamped -- it
    # is put down explicitly with Color0 by the routes below.
    inshape = {(rmin + dr, cmin + dc) for dr, dc in norm}
    clean_box = all(
        (r, c) in inshape or I[r, c] == 0
        for r in range(rmin, rmin + oh) for c in range(cmin, cmin + ow)
    )
    if clean_box and objc != 0 and len(kept) >= 3:
        mode = "stamp"
    elif (objc + len(norm)) % 2 == 0:
        mode = "clusters"     # copies that share cells read as one blob
    else:
        mode = "one_by_one"

    if mode == "stamp":
        # pick the marked template up once, then stamp it on every hidden copy
        ops.append(28)
        sels.append([rmin, cmin, oh - 1, ow - 1])
        for cells in kept:
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            ops.append(30)
            sels.append([r0, c0, 0, 0])         # paste origin only
    elif mode == "clusters":
        # merge copies that share cells, then bring out one blob at a time
        groups = []
        for cells in kept:
            s = set(cells)
            hit = [g for g in groups if g & s]
            for g in hit:
                groups.remove(g)
                s |= g
            groups.append(s)
        key = _route_key(route, mr, mc)
        groups.sort(key=lambda g: key(min(g, key=key)))
        for g in groups:
            ops.append(int(objc))
            sels.append(sel_of(sorted(g)))
    else:
        for cells in kept:
            ops.append(int(objc))
            sels.append(sel_of(cells))       # the copy's own cells, not a bbox

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])      # full-grid bbox: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task e5062a87"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e5062a87"
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
                                f"for task e5062a87"
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
                    f"Failed to build a complete episode for task e5062a87 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e5062a87-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
