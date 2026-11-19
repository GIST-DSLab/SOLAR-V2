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

# The golf entry and the maker's derive_operations read this task the same way.
# Each colour is one L-shaped piece. A piece's kind (UL/UR/LL/LR) is set by the
# corner of its bounding box that it leaves empty. The output is the frame you
# get by putting each piece in the matching corner of an
# (h_UL + h_LL) x (w_UL + w_UR) window.
# The units, the order and the fixed quantities all match, so the maker's route
# is kept: slide each whole piece with Move into the reassembly window, then crop.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; the rule never depends on
    # which colour each corner piece has.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    cols = [c for c in range(10) if c != bgc]
    hmax = max(4, min(15, max_h // 2))
    wmax = max(4, min(15, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (4, hmax))
    w = _unifint(diff_lb, diff_ub, (4, wmax))
    lociL = random.randint(2, h - 2)
    lociR = random.randint(2, h - 2)
    locjT = random.randint(2, w - 2)
    locjB = random.randint(2, w - 2)
    c1, c2, c3, c4 = random.sample(cols, 4)

    def vline(c, r0, r1):
        return {(r, c) for r in range(min(r0, r1), max(r0, r1) + 1)}

    def hline(r, c0, c1_):
        return {(r, c) for c in range(min(c0, c1_), max(c0, c1_) + 1)}

    ulco = vline(0, 0, lociL - 1) | hline(0, 0, locjT - 1)
    urco = hline(0, w - 1, locjT) | vline(w - 1, 0, lociR - 1)
    llco = vline(0, h - 1, lociL) | hline(h - 1, 0, locjB - 1)
    lrco = hline(h - 1, w - 1, locjB) | vline(w - 1, h - 1, lociR)

    go = np.full((h, w), bgc, dtype=int)
    for o, c in zip((ulco, urco, llco, lrco), (c1, c2, c3, c4)):
        for r, cc in o:
            go[r, cc] = c

    fullh = _unifint(diff_lb, diff_ub, (2 * h, max(2 * h, min(30, max_h))))
    fullw = _unifint(diff_lb, diff_ub, (2 * w, max(2 * w, min(30, max_w))))
    gi = np.full((fullh, fullw), bgc, dtype=int)
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    allinds = {(r, c) for r in range(fullh) for c in range(fullw)}
    for _ in range(200):
        inds = set(allinds)
        locs = []
        for o in objs:
            cands = [(i, j) for (i, j) in inds
                     if all((r + i, c + j) in inds for r, c in o)]
            if not cands:
                break
            loc = random.choice(cands)
            locs.append(loc)
            inds -= {(r + loc[0], c + loc[1]) for r, c in o}
        if len(locs) == 4:
            break
    else:
        raise RuntimeError("placement failed")
    for o, c, (i, j) in zip(objs, ocols, locs):
        for r, cc in o:
            gi[r + i, cc + j] = c
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    ops, sels = [], []

    # background = the canvas colour, which covers almost the whole input
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    cellmap = {}
    for r in range(fh):
        for c in range(fw):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))

    # one piece per colour; its kind is the bbox corner it leaves empty
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

    mask = (I != bgc).astype(int)
    pref = np.zeros((fh + 1, fw + 1), dtype=int)
    pref[1:, 1:] = mask.cumsum(0).cumsum(1)

    def rect_sum(R, C):
        return int(pref[R + h, C + w] - pref[R, C + w] - pref[R + h, C] + pref[R, C])

    def feasible_order(R, C):
        # order the moves so no piece lands on a piece that has not moved yet
        d = dest_origins(R, C)
        srcs = {k: pieces[k]['cells'] for k in KINDS}
        dsts = {}
        for k in KINDS:
            dr = d[k][0] - pieces[k]['r0']
            dc = d[k][1] - pieces[k]['c0']
            dsts[k] = {(r + dr, c + dc) for (r, c) in pieces[k]['cells']}
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

    cands = sorted(((cost_of(R, C), R, C)
                    for R in range(fh - h + 1) for C in range(fw - w + 1)))
    R = C = 0
    order = None
    # a piece coloured 0 is read as "nothing" by Move, so first try building
    # the frame around where that piece already lies
    zero = [k for k in KINDS if int(pieces[k]['color']) == 0]
    if zero:
        k0 = zero[0]
        ar = pieces[k0]['r0'] - (h - pieces[k0]['h'] if k0 in ('LL', 'LR') else 0)
        ac = pieces[k0]['c0'] - (w - pieces[k0]['w'] if k0 in ('UR', 'LR') else 0)
        if 0 <= ar <= fh - h and 0 <= ac <= fw - w:
            cands = [(-1, ar, ac)] + cands
    for _cost, rr, cc in cands:
        if rect_sum(rr, cc) == 0:
            R, C, order = rr, cc, list(KINDS)
            break
        o = feasible_order(rr, cc)
        if o is not None:
            R, C, order = rr, cc, o
            break

    dorg = dest_origins(R, C)

    if order is None:
        # last resort (no collision-free order exists): repaint the pieces
        R, C = min(cands)[1:] if cands[0][0] >= 0 else cands[1][1:]
        dorg = dest_origins(R, C)
        for k in KINDS:
            ops.append(bgc)
            sels.append(sel_of(sorted(pieces[k]['cells'])))
        for k in KINDS:
            dr = dorg[k][0] - pieces[k]['r0']
            dc = dorg[k][1] - pieces[k]['c0']
            ops.append(int(pieces[k]['color']))
            sels.append(sel_of(sorted((r + dr, c + dc) for (r, c) in pieces[k]['cells'])))
        ops.append(33)
        sels.append([R, C, h - 1, w - 1])
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # only the frame interior survives the crop as background
    interior = {(r, c) for r in range(R + 1, R + h - 1) for c in range(C + 1, C + w - 1)}

    used = set(int(v) for v in I.flatten().tolist())
    temp = next((t for t in range(1, 10) if t not in used), 1)

    for k in order:
        p = pieces[k]
        src = sorted(p['cells'])
        dr = dorg[k][0] - p['r0']
        dc = dorg[k][1] - p['c0']
        if dr == 0 and dc == 0:
            continue
        color = int(p['color'])
        recolored = (color == 0)
        if recolored:
            ops.append(temp)
            sels.append(sel_of(src))

        cur = list(src)
        grabbed = False
        for vr, vc, n, op in ((-1 if dr < 0 else 1, 0, abs(dr), 20 if dr < 0 else 21),
                              (0, -1 if dc < 0 else 1, abs(dc), 23 if dc < 0 else 22)):
            for _ in range(n):
                ops.append(op)
                sels.append(sel_of([]) if grabbed else sel_of(cur))
                grabbed = True
                cur = [(r + vr, c + vc) for r, c in cur]

        if recolored:
            ops.append(0)
            sels.append(sel_of(sorted(cur)))

        hole = sorted((set(src) - set(cur)) & interior)
        if bgc != 0 and hole:
            ops.append(bgc)
            sels.append(sel_of(hole))

    # the closed frame is exactly this full rectangle -> bbox selection is right
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
