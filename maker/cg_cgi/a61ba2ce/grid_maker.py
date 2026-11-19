"""
ARC Task: a61ba2ce (RE-ARC) — LLM-generated grid_maker
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
    # 0 reads as "nothing" to ARCLE object ops, so keep 0 for the background only
    pcols = random.sample([c for c in range(1, 10) if c != bgc], 4)
    return {"bgc": bgc, "c1": pcols[0], "c2": pcols[1], "c3": pcols[2], "c4": pcols[3]}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, c3=None, c4=None) -> dict:
    if bgc is None or None in (c1, c2, c3, c4):
        bgc = random.choice(range(10))
        c1, c2, c3, c4 = random.sample([c for c in range(1, 10) if c != bgc], 4)
    hub = max(4, min(15, max_h // 2))
    wub = max(4, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (4, hub))
    w = unifint(diff_lb, diff_ub, (4, wub))
    lociL = randint(2, h - 2)
    lociR = randint(2, h - 2)
    locjT = randint(2, w - 2)
    locjB = randint(2, w - 2)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(diff_lb, diff_ub, (2 * h, max(2 * h, min(30, max_h))))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max(2 * w, min(30, max_w))))
    gi = canvas(bgc, (fullh, fullw))
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    while True:
        inds = asindices(gi)
        locs = []
        for o, c in zip(objs, ocols):
            cands = sfilter(inds, lambda ij: shift(o, ij).issubset(inds))
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            locs.append(loc)
            inds = inds - shift(o, loc)
        if len(locs) == 4:
            break
    for o, c, l in zip(objs, ocols, locs):
        gi = fill(gi, c, shift(o, l))
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    ops, sels = [], []

    # background: the canvas colour; four thin L pieces sit on it
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    cellmap = {}
    for r in range(fh):
        for c in range(fw):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))

    # each colour is one L-shaped corner piece; its missing bbox corner names it
    pieces = {}
    for col, cells in cellmap.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        s = set(cells)
        if (r1, c1) not in s:
            kind = 'UL'
        elif (r1, c0) not in s:
            kind = 'UR'
        elif (r0, c1) not in s:
            kind = 'LL'
        else:
            kind = 'LR'
        pieces[kind] = {'color': col, 'cells': s, 'r0': r0, 'c0': c0,
                        'h': r1 - r0 + 1, 'w': c1 - c0 + 1}

    KINDS = ['UL', 'UR', 'LL', 'LR']
    if not all(k in pieces for k in KINDS):
        ops.append(34)
        sels.append([0, 0, fh - 1, fw - 1])
        return ops, sels

    # frame size: left side = UL height + LL height, top side = UL width + UR width
    h = pieces['UL']['h'] + pieces['LL']['h']
    w = pieces['UL']['w'] + pieces['UR']['w']

    def dest_origins(R, C):
        return {
            'UL': (R, C),
            'UR': (R, C + w - pieces['UR']['w']),
            'LL': (R + h - pieces['LL']['h'], C),
            'LR': (R + h - pieces['LR']['h'], C + w - pieces['LR']['w']),
        }

    def cost_of(R, C):
        d = dest_origins(R, C)
        return sum(abs(d[k][0] - pieces[k]['r0']) + abs(d[k][1] - pieces[k]['c0'])
                   for k in KINDS)

    def dst_cells(d, k):
        dr = d[k][0] - pieces[k]['r0']
        dc = d[k][1] - pieces[k]['c0']
        return {(r + dr, c + dc) for (r, c) in pieces[k]['cells']}

    def feasible_order(R, C):
        d = dest_origins(R, C)
        srcs = {k: pieces[k]['cells'] for k in KINDS}
        dsts = {k: dst_cells(d, k) for k in KINDS}
        rem = list(KINDS)
        order = []
        while rem:
            pick = None
            for i in rem:
                if all(not (dsts[i] & srcs[j]) for j in rem if j != i):
                    pick = i
                    break
            if pick is None:
                return None
            order.append(pick)
            rem.remove(pick)
        return order

    # nearest placement of the h x w reassembly window that can be filled
    # without a piece landing on one that has not moved yet
    cands = sorted((cost_of(R, C), R, C)
                   for R in range(fh - h + 1) for C in range(fw - w + 1))
    R = C = 0
    order = None
    for _cost, rr, cc in cands:
        o = feasible_order(rr, cc)
        if o is not None:
            R, C, order = rr, cc, o
            break

    if order is None:
        # no collision-free ordering: lift every piece off, then lay it down
        R, C = cands[0][1:]
        dorg = dest_origins(R, C)
        for k in KINDS:
            ops.append(bgc)
            sels.append(sel_of(sorted(pieces[k]['cells'])))
        for k in KINDS:
            ops.append(int(pieces[k]['color']))
            sels.append(sel_of(sorted(dst_cells(dorg, k))))
        ops.append(33)
        sels.append([R, C, h - 1, w - 1])
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    dorg = dest_origins(R, C)
    # only the frame's interior survives the crop as background
    interior = {(r, c) for r in range(R + 1, R + h - 1) for c in range(C + 1, C + w - 1)}

    for k in order:
        p = pieces[k]
        src = sorted(p['cells'])
        dr = dorg[k][0] - p['r0']
        dc = dorg[k][1] - p['c0']
        if dr == 0 and dc == 0:
            continue
        color = int(p['color'])

        if color == 0:
            # a 0-coloured piece is invisible to ARCLE's Move (0 = "nothing"),
            # so lay it down at its corner directly and clear where it was
            dst = sorted(dst_cells(dorg, k))
            ops.append(0)
            sels.append(sel_of(dst))
            hole = sorted((set(src) - set(dst)) & interior)
            if hole and bgc != 0:
                ops.append(bgc)
                sels.append(sel_of(hole))
            continue

        # slide the piece: grab once, then continue with empty selections
        cur = list(src)
        grabbed = False
        for vr, vc, n, op in ((-1 if dr < 0 else 1, 0, abs(dr), 20 if dr < 0 else 21),
                              (0, -1 if dc < 0 else 1, abs(dc), 23 if dc < 0 else 22)):
            for _ in range(n):
                ops.append(op)
                sels.append(sel_of([]) if grabbed else sel_of(cur))
                grabbed = True
                cur = [(r + vr, c + vc) for r, c in cur]

        # vacated footprint reads 0; repair it only inside the frame interior
        hole = sorted((set(src) - set(cur)) & interior)
        if bgc != 0 and hole:
            ops.append(bgc)
            sels.append(sel_of(hole))

    # the closed frame is exactly this full rectangle -> bbox crop
    ops.append(33)
    sels.append([R, C, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
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
                                f"for task a61ba2ce"
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
                    f"Failed to build a complete episode for task a61ba2ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a61ba2ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
