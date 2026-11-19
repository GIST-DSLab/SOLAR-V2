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


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, c1, c2, c3, c4 = random.sample(cols, 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


# ----------------------------------------------------------------------------
# 2. generator  (four L-shaped corner pieces scattered on a big canvas)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
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
    fullh = unifint(diff_lb, diff_ub, (min(2 * h, max_h), max_h))
    fullw = unifint(diff_lb, diff_ub, (min(2 * w, max_w), max_w))
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


# ----------------------------------------------------------------------------
# 3. operations
#
# Rule: the four L-shaped pieces are the corners of one rectangular frame.
#       Slide them together so the frame closes, then crop to the frame.
#
# The ROUTE is re-derived from the input each time (never from O):
#   * where the frame is assembled  — on top of one of the pieces (that piece
#     never moves) when that is not much more walking than the cheapest spot,
#     otherwise one of the equally-cheap meeting points,
#   * which corner is carried first and whether we go clockwise or anti-,
#   * per piece, whether it travels vertically first or horizontally first,
#   * whether the vacated footprints are tidied up as we go or all at the end.
# Every one of those knobs reads position/colour/size out of I, so one instance
# always yields one route, and different instances yield different routes.
# ----------------------------------------------------------------------------
def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    ops, sels = [], []

    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # --- the four corner pieces --------------------------------------------
    cellmap = {}
    for r in range(fh):
        for c in range(fw):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))

    pieces = {}
    for col, cells in cellmap.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        s = set(cells)
        # an L misses exactly the bbox corner diagonally opposite its own elbow
        if (r1, c1) not in s:
            kind = 'UL'
        elif (r1, c0) not in s:
            kind = 'UR'
        elif (r0, c1) not in s:
            kind = 'LL'
        else:
            kind = 'LR'
        pieces[kind] = {'color': int(col), 'cells': s, 'r0': r0, 'c0': c0,
                        'h': r1 - r0 + 1, 'w': c1 - c0 + 1}

    KINDS = ['UL', 'UR', 'LL', 'LR']
    CYCLE = ['UL', 'UR', 'LR', 'LL']          # visual clockwise tour
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

    def deltas(R, C):
        d = dest_origins(R, C)
        return {k: (d[k][0] - pieces[k]['r0'], d[k][1] - pieces[k]['c0']) for k in KINDS}

    def dest_cells(R, C):
        dl = deltas(R, C)
        return {k: {(r + dl[k][0], c + dl[k][1]) for (r, c) in pieces[k]['cells']}
                for k in KINDS}

    def cost_of(R, C):
        dl = deltas(R, C)
        return sum(abs(dl[k][0]) + abs(dl[k][1]) for k in KINDS)

    # --- input-derived personality of this solve ---------------------------
    colsum = sum(pieces[k]['color'] for k in KINDS)
    rowsum = sum(pieces[k]['r0'] for k in KINDS)
    colpos = sum(pieces[k]['c0'] for k in KINDS)

    anchor_start = colsum % 4
    clockwise = ((pieces['UL']['color'] + pieces['LR']['color'] + rowsum) % 2 == 0)
    order_start = (pieces['UL']['color'] + colpos) % 4
    tidy_as_you_go = ((colsum + h + w) % 2 == 0)
    vfirst = {k: ((pieces[k]['color'] + pieces[k]['r0'] + pieces[k]['c0']) % 2 == 0)
              for k in KINDS}

    step_dir = 1 if clockwise else -1
    pref_order = [CYCLE[(order_start + i * step_dir) % 4] for i in range(4)]

    def plan(R, C):
        """greedy order following this solver's preferred tour; None if a piece
        would always have to land on another piece that has not moved yet"""
        dc_ = dest_cells(R, C)
        rem = list(pref_order)
        order = []
        while rem:
            pick = None
            for k in rem:
                if all(not (dc_[k] & pieces[j]['cells']) for j in rem if j != k):
                    pick = k
                    break
            if pick is None:
                return None
            order.append(pick)
            rem.remove(pick)
        return order

    windows = [(R, C) for R in range(fh - h + 1) for C in range(fw - w + 1)]
    costs = {p: cost_of(*p) for p in windows}
    best = min(costs.values())

    def anchor_window(k):
        if k == 'UL':
            return (pieces['UL']['r0'], pieces['UL']['c0'])
        if k == 'UR':
            return (pieces['UR']['r0'], pieces['UR']['c0'] - (w - pieces['UR']['w']))
        if k == 'LL':
            return (pieces['LL']['r0'] - (h - pieces['LL']['h']), pieces['LL']['c0'])
        return (pieces['LR']['r0'] - (h - pieces['LR']['h']),
                pieces['LR']['c0'] - (w - pieces['LR']['w']))

    chosen = None
    # (a) assemble right where one piece already sits, if that is not much more
    #     walking than the cheapest meeting point
    budget = max(best + 8, (3 * best) // 2)
    for i in range(4):
        k = CYCLE[(anchor_start + i) % 4]
        R, C = anchor_window(k)
        if 0 <= R <= fh - h and 0 <= C <= fw - w and costs[(R, C)] <= budget:
            o = plan(R, C)
            if o is not None:
                chosen = (R, C, o)
                break
    # (b) otherwise one of the equally-cheap meeting points
    if chosen is None:
        opt = [p for p in windows if costs[p] == best]
        idx = (rowsum + colpos) % len(opt)
        for t in range(len(opt)):
            R, C = opt[(idx + t) % len(opt)]
            o = plan(R, C)
            if o is not None:
                chosen = (R, C, o)
                break
    # (c) otherwise the cheapest window that admits any collision-free order
    if chosen is None:
        for p in sorted(windows, key=lambda q: (costs[q], q)):
            o = plan(*p)
            if o is not None:
                chosen = (p[0], p[1], o)
                break

    if chosen is None:
        # last resort: no collision-free ordering exists anywhere
        R, C = min(windows, key=lambda q: (costs[q], q))
        dl = deltas(R, C)
        for k in KINDS:
            ops.append(bgc)
            sels.append(sel_of(sorted(pieces[k]['cells'])))
        for k in KINDS:
            dr, dc = dl[k]
            ops.append(int(pieces[k]['color']))
            sels.append(sel_of(sorted((r + dr, c + dc) for (r, c) in pieces[k]['cells'])))
        ops.append(33)
        sels.append([R, C, h - 1, w - 1])
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    R, C, order = chosen
    dl = deltas(R, C)
    dcells = dest_cells(R, C)
    window = {(r, c) for r in range(R, R + h) for c in range(C, C + w)}
    placed = set().union(*[dcells[k] for k in KINDS])

    used = set(int(v) for v in I.flatten().tolist())
    temp = next((t for t in range(1, 10) if t not in used), 1)

    def slide(src, dr, dc, v_first):
        cur = list(src)
        state = {'grabbed': False}

        def step(op, vr, vc):
            ops.append(op)
            sels.append(sel_of([]) if state['grabbed'] else sel_of(list(cur)))
            state['grabbed'] = True
            for i in range(len(cur)):
                r, c = cur[i]
                cur[i] = (r + vr, c + vc)

        def vert():
            for _ in range(abs(dr)):
                step(20 if dr < 0 else 21, -1 if dr < 0 else 1, 0)

        def horiz():
            for _ in range(abs(dc)):
                step(23 if dc < 0 else 22, 0, -1 if dc < 0 else 1)

        if v_first:
            vert()
            horiz()
        else:
            horiz()
            vert()
        return cur

    carried = []
    for k in order:
        dr, dc = dl[k]
        if dr == 0 and dc == 0:
            continue                      # this corner is already where it belongs
        p = pieces[k]
        src = sorted(p['cells'])
        color = int(p['color'])

        # ARCLE's object ops ignore 0-valued cells: lift such a piece to a spare
        # colour so it can be grabbed at all, and put it back down afterwards
        lifted = (color == 0)
        if lifted:
            ops.append(temp)
            sels.append(sel_of(src))

        cur = slide(src, dr, dc, vfirst[k])

        if lifted:
            ops.append(0)
            sels.append(sel_of(sorted(cur)))

        carried.append(set(src))

        if tidy_as_you_go and bgc != 0:
            hole = sorted((set(src) - set(cur)) & window)
            if hole:
                ops.append(bgc)
                sels.append(sel_of(hole))

    if (not tidy_as_you_go) and bgc != 0 and carried:
        hole = sorted((set().union(*carried) - placed) & window)
        if hole:
            ops.append(bgc)
            sels.append(sel_of(hole))

    # the closed frame is exactly this full rectangle -> a bbox selection is right
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
