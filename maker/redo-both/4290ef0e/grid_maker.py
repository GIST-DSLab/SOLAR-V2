"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """bgc plus the ordered pool the nested rings take their colours from.

    The rule (nest the rings by size) does not depend on which colour a ring
    has, but the pool is fixed for the whole episode so every instance shows the
    same colour scheme.  0 is pushed to the end of the pool and therefore never
    becomes a ring colour: ARCLE's object ops treat 0 as transparent, so a
    0-coloured ring could not be grabbed and slid.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    pool = [c for c in cols if c != bgc]
    random.shuffle(pool)
    if 0 in pool:
        pool.remove(0)
        pool.append(0)
    return {"bgc": bgc, "ccols_pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols_pool) -> dict:
    dcap = min(7, min(max_h, max_w) // 4)
    if dcap < 2:
        dcap = 2
    while True:
        d = unifint(diff_lb, diff_ub, (2, dcap))
        h, w = d, d
        fullh = unifint(diff_lb, diff_ub, (min(4 * d, max_h), max_h))
        fullw = unifint(diff_lb, diff_ub, (min(4 * d, max_w), max_w))
        ccols = list(ccols_pool[:d])
        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w - idx + 1)
            quad = fill(quad, c, (connect((idx, idx), (idx + linlen - 1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx + linlen - 1))))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if choice((True, False)):
            go = fill(go, ccols_pool[d], {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            ow = width(obj)
            dh = max(0, ow // 2 - 1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - ow and ij[1] <= fullw - ow)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) \
                          | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            if len(cands) == 0:
                fullsuc = False
                break
            maxtr = 10
            tr = 0
            succ = False
            while tr < maxtr and not succ:
                tr += 1
                loc = choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The scattered pieces are the concentric rings of one figure that was
    taken apart.  Each colour is one square ring; the biggest ring is the frame,
    and every smaller ring nests inside it, its side shrinking by 2 per step.
    So: slide each ring onto the ring slot its own size buys it, then crop the
    canvas down to the frame.

    Everything is measured on I: which pieces there are, how big each ring is
    (the side of its bounding box), where the frame sits, and hence how far each
    ring has to travel.  A ring that the border of I cut in half is completed
    from its own mirror symmetry - also read off I.  O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    ops, sels = [], []

    # ---- read the rings off the input ---------------------------------------
    def resolve(cells, s, axis, extent):
        """Where does this ring really start along `axis`?

        A ring is s x s, but the border of I may have cut a strip off one end,
        so the visible bounding box can be shorter.  A ring is mirror-symmetric
        about its own centre, so for a candidate start the cut-off cells are the
        mirror image of the visible ones; the candidate is only possible if
        every mirrored cell that lands back inside the grid is in fact visible.
        """
        vs = [p[axis] for p in cells]
        lo, hs = min(vs), max(vs)
        if hs - lo + 1 == s:
            return lo, set(cells)
        best = None
        for t in range(hs - s + 1, lo + 1):
            mid2 = 2 * t + s - 1
            full = set(cells)
            for (r, k) in cells:
                full.add((mid2 - r, k) if axis == 0 else (r, mid2 - k))
            # a ring reaches both ends of its own box, so the completed ring
            # has to span exactly s -- no more, no less
            fs = [q[axis] for q in full]
            if min(fs) != t or max(fs) != t + s - 1:
                continue
            ok = True
            for q in full:
                if 0 <= q[axis] < extent and q not in cells:
                    ok = False
                    break
            if not ok:
                continue
            score = (len(full) - len(cells), abs(t - lo))
            if best is None or score < best[0]:
                best = (score, t, full)
        if best is None:
            return lo, set(cells)
        return best[1], best[2]

    rings = []
    for c in sorted(set(I.flatten().tolist())):
        if c == bgc:
            continue
        cells = {(int(r), int(k)) for r, k in zip(*np.where(I == c))}
        rs = [p[0] for p in cells]
        ks = [p[1] for p in cells]
        side = max(max(rs) - min(rs) + 1, max(ks) - min(ks) + 1)
        tr, full_r = resolve(cells, side, 0, hi)
        tc, full_c = resolve(cells, side, 1, wi)
        rings.append({"color": int(c), "cells": set(cells), "full": full_r | full_c,
                      "tr": tr, "tc": tc, "s": side})
    rings.sort(key=lambda L: -L["s"])

    S = rings[0]["s"]                       # the frame, and the figure's side
    # the frame stays where it is, nudged just far enough to lie inside the grid
    FR = min(max(rings[0]["tr"], 0), hi - S)
    FC = min(max(rings[0]["tc"], 0), wi - S)
    for L in rings:
        off = (S - L["s"]) // 2             # how deep this ring nests
        L["dr"] = FR + off - L["tr"]
        L["dc"] = FC + off - L["tc"]
        L["dst"] = {(r + L["dr"], k + L["dc"]) for (r, k) in L["full"]}

    # ---- ops -----------------------------------------------------------------
    def slide(L, dr, dc):
        """Carry the ring dr/dc cells: one grab, then empty selections so ARCLE
        keeps it held and restores every cell it glides over."""
        src = sorted(L["cells"])
        pos = list(src)
        first = True
        for _ in range(abs(dr)):
            ops.append(21 if dr > 0 else 20)
            sels.append(sel_of(pos) if first else sel_of([]))
            first = False
            pos = [(r + (1 if dr > 0 else -1), k) for (r, k) in pos]
        for _ in range(abs(dc)):
            ops.append(22 if dc > 0 else 23)
            sels.append(sel_of(pos) if first else sel_of([]))
            first = False
            pos = [(r, k + (1 if dc > 0 else -1)) for (r, k) in pos]
        hole = sorted(set(src) - set(pos))
        if bgc != 0 and hole:               # only the footprint it left is 0 now
            ops.append(bgc)
            sels.append(sel_of(hole))
        L["cells"] = set(pos)
        L["tr"] += dr
        L["tc"] += dc
        L["dr"] -= dr
        L["dc"] -= dc

    def place(L):
        col = L["color"]
        drawn = set()
        if L["dr"] or L["dc"]:
            if col != 0:
                slide(L, L["dr"], L["dc"])
            else:
                # ARCLE's object ops see colour 0 as nothing, so this ring cannot
                # be grabbed: draw it on its slot and clear the ground it left
                ops.append(0)
                sels.append(sel_of(sorted(L["dst"])))
                drawn = set(L["dst"])
                rest = sorted(L["cells"] - L["dst"])
                if rest:
                    ops.append(bgc)
                    sels.append(sel_of(rest))
                L["cells"] = set(L["dst"])
                L["dr"] = L["dc"] = 0
        # the strip the border of I cut off this ring, put back by its symmetry
        missing = sorted(L["dst"] - L["cells"] - drawn)
        if missing:
            ops.append(col)
            sels.append(sel_of(missing))
            L["cells"] |= set(missing)

    def free_spot(L, blocked):
        """Ground where this ring can wait its turn: clear of every other ring
        and of every slot still to be filled."""
        cells = L["cells"]
        rs = [p[0] for p in cells]
        ks = [p[1] for p in cells]
        r0, k0 = min(rs), min(ks)
        h = max(rs) - r0 + 1
        w = max(ks) - k0 + 1
        best = None
        for r in range(0, hi - h + 1):
            for k in range(0, wi - w + 1):
                dr, dc = r - r0, k - k0
                if dr == 0 and dc == 0:
                    continue
                moved = {(a + dr, b + dc) for (a, b) in cells}
                if moved & blocked:
                    continue
                cost = abs(dr) + abs(dc)
                if best is None or cost < best[0]:
                    best = (cost, dr, dc)
        return None if best is None else (best[1], best[2])

    placed, waiting, shunts = [], list(rings), 2 * len(rings) + 4
    while waiting:
        # a ring may only be laid down once nothing else is still sitting on its slot
        pick = None
        for L in waiting:
            others = set()
            for M in waiting:
                if M is not L:
                    others |= M["cells"]
            if not (L["dst"] & others):
                pick = L
                break
        if pick is None and shunts > 0:
            # every remaining ring sits on another ring's slot: park one of the
            # blockers on ground no ring needs, and the knot unties
            shunts -= 1
            occupied = set()
            for M in placed:
                occupied |= M["cells"]
            shunted = False
            for L in waiting:
                if L["color"] == 0:
                    continue
                blocking = False
                for M in waiting:
                    if M is not L and (M["dst"] & L["cells"]):
                        blocking = True
                        break
                if not blocking:
                    continue
                occ = set(occupied)
                for M in waiting:
                    if M is not L:
                        occ |= M["cells"]
                        occ |= M["dst"]
                spot = free_spot(L, occ)
                if spot is not None:
                    slide(L, spot[0], spot[1])
                    shunted = True
                    break
            if shunted:
                continue
        if pick is None:
            pick = waiting[0]
        place(pick)
        waiting.remove(pick)
        placed.append(pick)

    # the figure is assembled: keep the frame, drop the rest of the sheet
    ops.append(33)
    sels.append([FR, FC, S - 1, S - 1])     # full rectangle -> bbox form
    ops.append(34)
    sels.append([0, 0, S - 1, S - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
