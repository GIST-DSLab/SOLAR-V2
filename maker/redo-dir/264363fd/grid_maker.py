"""
ARC Task: 264363fd (RE-ARC) — LLM-generated grid_maker
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

# spike direction indices: 0 = o1 (up), 1 = o2 (left), 2 = o3 (down), 3 = o4 (right)
SPIKE_SETS = [(0,), (2,), (3,), (0, 2), (1, 3), (0, 1), (1, 2, 3), (0, 1, 2, 3)]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    rem = [c for c in cols if c not in (bgc, sqc, linc)]
    cpcol = random.choice(rem)
    nbhcol = random.choice(rem)

    n_ex = num_examples if num_examples else 3
    variants = [{"spikes_idx": s} for s in SPIKE_SETS]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test glyph was shown

    return {"bgc": bgc, "sqc": sqc, "linc": linc, "cpcol": cpcol, "nbhcol": nbhcol,
            "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sqc=None, linc=None, cpcol=None, nbhcol=None,
             spikes_idx=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None or linc is None:
        bgc, sqc, linc = random.sample(cols, 3)
    if cpcol is None or nbhcol is None:
        rem = [c for c in cols if c not in (bgc, sqc, linc)]
        cpcol = random.choice(rem)
        nbhcol = random.choice(rem)

    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    olist = [o1, o2, o3, o4]
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}

    hlo, hhi = min(15, max_h), max_h
    wlo, whi = min(15, max_w), max_w

    last = None
    for _try in range(30):
        h = unifint(diff_lb, diff_ub, (hlo, hhi))
        w = unifint(diff_lb, diff_ub, (wlo, whi))
        if spikes_idx is None:
            nspikes = random.randint(1, 4)
            spikes = random.sample(olist, nspikes)
        else:
            spikes = [olist[i] for i in spikes_idx]
        lns = merge(set(spikes))
        obj = {(cpcol, cp)} | recolor(linc, lns) | recolor(nbhcol, neighs - lns)
        loci = random.randint(0, h - 5)
        locj = random.randint(0, w - 5)
        loc = (loci, locj)
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        gi = paint(gi, shift(obj, loc))
        numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 100)))
        succ = 0
        tr = 0
        maxtr = 10 * numsq
        inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(shift(obj, loc)))
        while succ < numsq and tr < maxtr:
            tr += 1
            gh = random.randint(5, h // 2 + 1)
            gw = random.randint(5, w // 2 + 1)
            cands = sfilter(inds, lambda ij: ij[0] <= h - gh and ij[1] <= w - gw)
            if len(cands) == 0:
                continue
            loc3 = random.choice(totuple(cands))
            g1 = canvas(sqc, (gh, gw))
            g2 = canvas(sqc, (gh, gw))
            ginds = asindices(g1)
            gindsfull = asindices(g1)
            bck = shift(ginds, loc3)
            if bck.issubset(inds):
                noccs = unifint(diff_lb, diff_ub, (1, max(1, (gh * gw) // 25)))
                succ2 = 0
                tr2 = 0
                maxtr2 = 5 * noccs
                while succ2 < noccs and tr2 < maxtr2:
                    tr2 += 1
                    cands2 = sfilter(ginds, lambda ij: ij[0] <= gh - 5 and ij[1] <= gw - 5)
                    if len(cands2) == 0:
                        break
                    loc2 = random.choice(totuple(cands2))
                    lns2 = merge(frozenset({shoot(add(cp, add(loc2, mpr[spike])), mpr[spike])
                                            for spike in spikes}))
                    lns2 = lns2 & gindsfull
                    plcd2 = shift(obj, loc2)
                    plcd2i = toindices(plcd2)
                    if plcd2i.issubset(ginds) and lns2.issubset(ginds | ofcolor(g2, linc)) \
                            and len(lns2 - plcd2i) > 0:
                        succ2 += 1
                        ginds = ((ginds - plcd2i) - mapply(neighbors, plcd2i)) - lns2
                        g1 = fill(g1, cpcol, {add(cp, loc2)})
                        g2 = paint(g2, plcd2)
                        g2 = fill(g2, linc, lns2)
                if succ2 > 0:
                    succ += 1
                    inds = (inds - bck) - outbox(bck)
                    objfull1 = shift(asobject(g1), loc3)
                    objfull2 = shift(asobject(g2), loc3)
                    gi = paint(gi, objfull1)
                    go = paint(go, objfull2)
        last = {'input': gi, 'output': go}
        if succ > 0:
            return last
    return last


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def comps_of(mask):
        seen = np.zeros_like(mask, dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cur = []
                    while stack:
                        rr, cc = stack.pop()
                        cur.append((rr, cc))
                        for dr, dc in DIRS:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    out.append(cur)
        return out

    # ---- background: colour of the uniform 4-connected region with the largest bbox ----
    best = None
    for col in sorted(set(I.flatten().tolist())):
        for comp in comps_of(I == col):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if best is None or area > best[0]:
                best = (area, col)
    bgc = int(best[1])

    # ---- the key glyph (smallest non-background component) and the square panels ----
    parts = comps_of(I != bgc)
    parts.sort(key=lambda p: (len(p), min(p)))
    key = parts[0]
    panels = sorted(parts[1:], key=lambda p: min(p))
    kset = set(key)

    # centre of the key = the cell that owns all 8 of its Moore neighbours
    bestc = None
    for (r, c) in key:
        cnt = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr, dc) != (0, 0) and (r + dr, c + dc) in kset)
        if bestc is None or cnt > bestc[0]:
            bestc = (cnt, (r, c))
    cr, cc = bestc[1]

    spikes = [d for d in DIRS if (cr + 2 * d[0], cc + 2 * d[1]) in kset]
    linc = int(I[cr + 2 * spikes[0][0], cc + 2 * spikes[0][1]]) if spikes else None

    # offsets of the 3x3 key core whose colour is 0 (Paste cannot carry those cells)
    zero_off = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                if (i, j) != (0, 0) and int(I[cr + i, cc + j]) == 0]

    # ---- collect the marker dots panel by panel ----
    targets = []   # (panel bbox, dot)
    for comp in panels:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sqc = Counter(int(I[r, c]) for (r, c) in comp).most_common(1)[0][0]
        dots = sorted([(r, c) for (r, c) in comp if int(I[r, c]) != sqc])
        for d in dots:
            targets.append(((r0, c0, r1, c1), d))

    ops, sels = [], []

    if targets:
        # grab the key's 3x3 core from the input — full rectangle, exactly the cells wanted
        ops.append(28); sels.append([cr - 1, cc - 1, 2, 2])

        for (r0, c0, r1, c1), (dr_, dc_) in targets:
            # stamp the key core so its centre lands on the marker dot
            ops.append(30); sels.append([dr_ - 1, dc_ - 1, 0, 0])
            # cells the clipboard could not carry (colour 0 is transparent to Paste)
            zc = [(dr_ + i, dc_ + j) for (i, j) in zero_off]
            if zc:
                ops.append(0); sels.append(sel_of(zc))
            # shoot each spike of the glyph outwards to the panel border
            for d in spikes:
                ray, k = [], 2
                while True:
                    rr, ccx = dr_ + d[0] * k, dc_ + d[1] * k
                    if not (r0 <= rr <= r1 and c0 <= ccx <= c1):
                        break
                    ray.append((rr, ccx))
                    k += 1
                if ray:
                    ops.append(linc); sels.append(sel_of(ray))

    # the key has been stamped everywhere — remove it from the background
    ops.append(bgc); sels.append(sel_of(sorted(key)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 264363fd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 264363fd"
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
                                f"for task 264363fd"
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
                    f"Failed to build a complete episode for task 264363fd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"264363fd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
