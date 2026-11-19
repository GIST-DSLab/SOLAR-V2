"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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
    bgc, ca, cb = random.sample(range(10), 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or ca is None or cb is None:
        bgc, ca, cb = random.sample(cols, 3)
    himax = max(2, min(8, max_h // 3, max_w // 3))
    while True:
        hi = unifint(diff_lb, diff_ub, (2, himax))
        h = unifint(diff_lb, diff_ub, (3 * hi, max_h))
        w = unifint(diff_lb, diff_ub, (3 * hi, max_w))
        c = canvas(-1, (hi, hi))
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
        go = fill(c, ca, A)
        go = fill(go, cb, B)
        fullocs = totuple(asindices(gi))
        A = normalize(A)
        B = normalize(B)
        ha, wa = shape(A)
        hb, wb = shape(B)
        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locai = randint(0, h - ha)
                locaj = randint(0, w - wa)
                plcda = shift(A, (locai, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(remlocs2)
                plcdb = shift(B, ch)
                if set(plcdb).issubset(set(remlocs)):
                    minisuc = True
                    break
        if minisuc:
            break
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the two single-coloured pieces (4-connected, distinct colours)
    pieces = []
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
        pieces.append((col, cells))
    assert len(pieces) == 2
    total = sum(len(p[1]) for p in pieces)
    s = int(round(total ** 0.5))

    def norm(cells):
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        return [(r - r0, c - c0) for r, c in cells], (r0, c0)

    (ca, A), (cb, B) = pieces
    nA, posA = norm(A)
    nB, posB = norm(B)
    haA = max(r for r, _ in nA) + 1; waA = max(c for _, c in nA) + 1
    haB = max(r for r, _ in nB) + 1; waB = max(c for _, c in nB) + 1

    # all placements of the two pieces that tile an s x s square
    full = {(r, c) for r in range(s) for c in range(s)}
    fits = []
    for oar in range(s - haA + 1):
        for oac in range(s - waA + 1):
            sa = {(r + oar, c + oac) for r, c in nA}
            for obr in range(s - haB + 1):
                for obc in range(s - waB + 1):
                    sb = {(r + obr, c + obc) for r, c in nB}
                    if not (sa & sb) and (sa | sb) == full:
                        fits.append(((oar, oac), (obr, obc)))
    assert fits
    # several tilings only if the pieces are ambiguous; take the one the rule yields
    def render(f):
        g = np.zeros((s, s), dtype=int)
        for r, c in nA: g[r + f[0][0], c + f[0][1]] = ca
        for r, c in nB: g[r + f[1][0], c + f[1][1]] = cb
        return g
    fit = next((f for f in fits if O.shape == (s, s) and np.array_equal(render(f), O)), fits[0])
    (oar, oac), (obr, obc) = fit

    movable = {"A": ca != 0, "B": cb != 0}   # ARCLE cannot grab colour-0 cells
    info = {"A": (A, posA, (oar, oac)), "B": (B, posB, (obr, obc))}

    def plan_for(R, C):
        dest = {}
        for k in "AB":
            cells, pos, off = info[k]
            d = (R + off[0] - pos[0], C + off[1] - pos[1])
            if d != (0, 0) and not movable[k]:
                return None
            dest[k] = d
        fin = {k: {(r + dest[k][0], c + dest[k][1]) for r, c in info[k][0]} for k in "AB"}
        src = {k: set(info[k][0]) for k in "AB"}
        cost = sum(abs(dest[k][0]) + abs(dest[k][1]) for k in "AB")
        moving = [k for k in "AB" if dest[k] != (0, 0)]
        for order in (moving, moving[::-1]):
            ok = True
            for i, k in enumerate(order):
                # k's final position must not cover pieces that still have to be grabbed
                later = [o for o in order[i + 1:]]
                if any(fin[k] & src[o] for o in later):
                    ok = False
            if ok:
                return cost, list(order), dest
        return None

    best = None
    for R in range(hi - s + 1):
        for C in range(wi - s + 1):
            p = plan_for(R, C)
            if p is not None and (best is None or p[0] < best[0]):
                best = (p[0], R, C, p[1], p[2])

    ops, sels = [], []

    def slide(cells, dr, dc):
        # one grab, then empty selections continue the same object
        steps = [20 if dr < 0 else 21] * abs(dr) + [23 if dc < 0 else 22] * abs(dc)
        for i, op in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(cells) if i == 0 else sel_of([]))

    if best is not None:
        _, R, C, order, dest = best
        for k in order:
            slide(info[k][0], *dest[k])
        # crop the canvas to the assembled square (full rectangle intended)
        ops.append(33); sels.append([R, C, s - 1, s - 1])
    else:
        # a colour-0 piece cannot be grabbed and its square would leave the grid:
        # slide the coloured piece into a square slot, crop, then draw the 0-piece.
        zk = "A" if not movable["A"] else "B"
        mk = "B" if zk == "A" else "A"
        cells, pos, off = info[mk]
        cand = []
        for R in range(hi - s + 1):
            for C in range(wi - s + 1):
                d = (R + off[0] - pos[0], C + off[1] - pos[1])
                cand.append((abs(d[0]) + abs(d[1]), R, C, d))
        _, R, C, d = min(cand)
        slide(cells, *d)
        ops.append(33); sels.append([R, C, s - 1, s - 1])
        zcells, zpos, zoff = info[zk]
        nz = nA if zk == "A" else nB
        ops.append(0); sels.append(sel_of([(r + zoff[0], c + zoff[1]) for r, c in nz]))

    ops.append(34); sels.append([0, 0, s - 1, s - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
