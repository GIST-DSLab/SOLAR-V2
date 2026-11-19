"""
ARC Task: 57aa92db (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. colours
#    The rule is: every object carries one marker cell/block of `fixc`; the one
#    complete object (marker + `mainc` body) is the template, every other object
#    is a partially drawn, upscaled copy of it in its own colour.  bgc / fixc /
#    mainc are real roles and are fixed for the whole episode; the per-occurrence
#    colour is just an identity ("this object's colour") and stays random.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fixc, mainc = random.sample(cols, 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC 57aa92db, colours injected, 30 -> max_h/max_w)
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fixc: int, mainc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    oh = random.randint(2, 5)
    ow = random.randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {random.choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(random.choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = random.choice(totuple(obj))
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(obj)))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        # mild bias towards genuinely upscaled copies (large factors often fail
        # the fitting test below, so factor 1 still shows up regularly)
        fac = random.choice((1, 2, 2, 3, 3, 4))
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = random.choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = random.choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {random.choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(random.choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
#
# Rule (read off I alone): one colour (`fixc`) marks every object.  The object
# whose marker is a single cell and which is largest is the TEMPLATE - the whole
# shape.  Every other object is that shape upscaled by `fac` (= the side of its
# fixc marker block), anchored so its marker block sits on the marker's place in
# the shape, and only partially drawn.  Each such object must be completed in
# its own colour.
#
# Account used here, grouped by what the rule is about:
#   * factor 1 -> the object is the template shape itself: one Color op per
#     object completes that object.
#   * factor >= 2 -> the object is literally a repeat of one fac x fac tile at
#     the template's cell positions: fill one tile, CopyO it, and Paste it at
#     every tile position that is not finished yet.  The clipboard is reused
#     across objects whose (factor, colour) already match.
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- background: the colour all demonstration grids are drawn on ---
    grids = [I]
    if examples:
        for pair in examples:
            try:
                a, _b = pair
            except Exception:
                continue
            grids.append(np.asarray(a, dtype=int))
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]

    # --- objects: 8-connected non-background components ---
    comps = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cur = []
                while stack:
                    rr, cc = stack.pop()
                    cur.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                comps.append(cur)

    ops, sels = [], []
    if not comps:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # --- marker colour: the colour carried by every object ---
    pres = Counter()
    for comp in comps:
        for col in set(int(I[r, c]) for r, c in comp):
            pres[col] += 1
    fixc = max(pres.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def marker_of(comp):
        cells = [(r, c) for (r, c) in comp if I[r, c] == fixc]
        if not cells:
            return None, None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        fac = max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        return (min(rs), min(cs)), fac

    # --- template: marker of size 1, largest such object ---
    tmpl_idx, tmpl_size = None, -1
    info = []
    for i, comp in enumerate(comps):
        mul, fac = marker_of(comp)
        info.append((mul, fac))
        if fac == 1 and len(comp) > tmpl_size:
            tmpl_size, tmpl_idx = len(comp), i
    if tmpl_idx is None:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    tcomp = comps[tmpl_idx]
    tr0 = min(r for r, _ in tcomp)
    tc0 = min(c for _, c in tcomp)
    tcells = set((r - tr0, c - tc0) for (r, c) in tcomp)
    mul, _ = info[tmpl_idx]
    fixp = (mul[0] - tr0, mul[1] - tc0)

    # template cells ordered outward from the marker
    order = []
    seenc = {fixp}
    dq = deque([fixp])
    while dq:
        cell = dq.popleft()
        if cell != fixp:
            order.append(cell)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = (cell[0] + dr, cell[1] + dc)
                if nb in tcells and nb not in seenc:
                    seenc.add(nb)
                    dq.append(nb)
    for cell in sorted(tcells):
        if cell != fixp and cell not in seenc:
            order.append(cell)

    # --- the partial copies, in raster order of where they sit ---
    occs = []
    for i, comp in enumerate(comps):
        if i == tmpl_idx:
            continue
        mul, fac = info[i]
        if mul is None:
            continue
        others = [int(I[r, c]) for (r, c) in comp if I[r, c] != fixc]
        if not others:
            continue
        newc = Counter(others).most_common(1)[0][0]
        ul = (mul[0] - fac * fixp[0], mul[1] - fac * fixp[1])
        occs.append((ul, fac, newc))
    occs.sort()

    G = I.copy()
    clip = None

    for (ul, fac, newc) in occs:
        blocks = []
        for t in order:
            r0 = ul[0] + fac * t[0]
            c0 = ul[1] + fac * t[1]
            cells = [(r0 + i, c0 + j) for i in range(fac) for j in range(fac)
                     if 0 <= r0 + i < h and 0 <= c0 + j < w]
            if cells:
                blocks.append(cells)
        incomplete = [b for b in blocks if any(G[r, c] != newc for (r, c) in b)]
        if not incomplete:
            continue

        # factor 1: the copy is the template shape itself -> complete the object
        if fac == 1 or newc == 0:
            missing = [(r, c) for b in incomplete for (r, c) in b if G[r, c] != newc]
            ops.append(int(newc))
            sels.append(sel_of(missing))
            for (r, c) in missing:
                G[r, c] = newc
            continue

        # factor >= 2: the copy is one fac x fac tile repeated at the
        # template's cell positions
        if clip == (fac, int(newc)):
            targets = incomplete
        else:
            complete = [b for b in blocks if all(G[r, c] == newc for (r, c) in b)]
            src = complete[0] if complete else incomplete[0]
            targets = [b for b in incomplete if b is not src]
            if not targets:
                missing = [(r, c) for (r, c) in src if G[r, c] != newc]
                ops.append(int(newc))
                sels.append(sel_of(missing))
                for (r, c) in missing:
                    G[r, c] = newc
                continue
            missing = [(r, c) for (r, c) in src if G[r, c] != newc]
            if missing:
                ops.append(int(newc))
                sels.append(sel_of(missing))
                for (r, c) in missing:
                    G[r, c] = newc
            ops.append(29)                 # CopyO: the finished tile
            sels.append(sel_of(src))       # exactly the fac x fac tile
            clip = (fac, int(newc))

        for b in targets:
            ops.append(30)                 # Paste the tile at this tile slot
            sels.append(sel_of([b[0]]))
            for (r, c) in b:
                G[r, c] = newc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
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
                                f"for task 57aa92db"
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
                    f"Failed to build a complete episode for task 57aa92db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"57aa92db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
