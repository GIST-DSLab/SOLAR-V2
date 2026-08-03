"""
ARC Task: 39e1d7f9 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc) -> dict:
    cols = interval(0, 10, 1)
    h_ub = max(5, min(10, (max_h + 1) // 2))
    w_ub = max(5, min(10, (max_w + 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, h_ub))
    w = unifint(diff_lb, diff_ub, (5, w_ub))
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    if h == 5:
        loci = choice((1, h - 2))
    if w == 5:
        locj = choice((1, w - 2))
    npix = unifint(diff_lb, diff_ub, (1, 8))
    ncols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * len(pixs) + 1)))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(diff_lb, diff_ub, (1, (max_h - h + 1) // h))
    wfac = unifint(diff_lb, diff_ub, (1, (max_w - w + 1) // w))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            c = gi[a][b]
            gi2 = fill(gi2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            c = go[a][b]
            go2 = fill(go2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    gi, go = gi2, go2
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- recover the lattice: first uniform row/col is the first separator line ---
    hfac = H
    for r in range(H):
        if len(set(I[r].tolist())) == 1:
            hfac = r
            break
    wfac = W
    for c in range(W):
        if len(set(I[:, c].tolist())) == 1:
            wfac = c
            break
    hp, wp = hfac + 1, wfac + 1
    h = (H + 1) // hp
    w = (W + 1) // wp

    Ci = np.array([[I[a * hp, b * wp] for b in range(w)] for a in range(h)], dtype=int)
    Co = np.array([[O[a * hp, b * wp] for b in range(w)] for a in range(h)], dtype=int)
    bgc = Counter(Ci.flatten().tolist()).most_common(1)[0][0]

    def pattern_at(G, r, c):
        P = {}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and G[rr, cc] != bgc:
                    P[(dr, dc)] = int(G[rr, cc])
        return P

    def block(a, b):
        return [(a * hp + i, b * wp + j) for i in range(hfac) for j in range(wfac)]

    # --- identify the marker color and the blob pattern stamped onto every marker ---
    best = None
    for cand in sorted(set(Ci.flatten().tolist()) - {bgc}):
        S = [(r, c) for r in range(h) for c in range(w) if Ci[r, c] == cand]
        s0 = max(S, key=lambda s: len(pattern_at(Ci, s[0], s[1])))
        P = pattern_at(Co, s0[0], s0[1])
        if P.get((0, 0)) != cand:
            continue
        T = Ci.copy()
        ok = True
        for (r, c) in S:
            for (dr, dc), v in P.items():
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                T[rr, cc] = v
            if not ok:
                break
        if ok and np.array_equal(T, Co):
            best = (S, P, s0)
            break

    if best is None:
        # fallback: paint every differing lattice cell directly
        for a in range(h):
            for b in range(w):
                if Ci[a, b] != Co[a, b]:
                    ops.append(int(Co[a, b]))
                    sels.append(sel_of(block(a, b)))
        ops.append(34)
        sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    S, P, s0 = best
    r0 = min(dr for dr, _ in P)
    r1 = max(dr for dr, _ in P)
    c0 = min(dc for _, dc in P)
    c1 = max(dc for _, dc in P)

    dests = [s for s in S if pattern_at(Ci, s[0], s[1]) != P]

    if dests:
        # copy the template blob's lattice-aligned bounding box from the input
        src_r, src_c = (s0[0] + r0), (s0[1] + c0)
        ops.append(28)
        sels.append([src_r * hp, src_c * wp,
                     (r1 - r0) * hp + hfac - 1, (c1 - c0) * wp + wfac - 1])

        cur = Ci.copy()
        for s in dests:
            ops.append(30)
            sels.append([(s[0] + r0) * hp, (s[1] + c0) * wp, 0, 0])
            for dr in range(r0, r1 + 1):
                for dc in range(c0, c1 + 1):
                    v = int(Ci[s0[0] + dr, s0[1] + dc])
                    if v != 0:  # Paste is transparent on 0
                        cur[s[0] + dr, s[1] + dc] = v

        # repair only cells the transparent paste could not deliver
        for s in dests:
            for dr in range(r0, r1 + 1):
                for dc in range(c0, c1 + 1):
                    a, b = s[0] + dr, s[1] + dc
                    if cur[a, b] != Co[a, b]:
                        ops.append(int(Co[a, b]))
                        sels.append(sel_of(block(a, b)))
                        cur[a, b] = Co[a, b]

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
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
                                f"for task 39e1d7f9"
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
                    f"Failed to build a complete episode for task 39e1d7f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39e1d7f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
