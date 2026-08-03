"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """Episode-level colors: background + ordered foreground palette.

    palette[0] is the 'frame' colour (the 4 corner cells) — fixed across the whole
    episode so the rule stays readable.  Colour 0 is kept out of the foreground
    palette: ARCLE's object ops (Move) only grab NONZERO cells, so a 0-coloured
    object could never be translated.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc and c != 0]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = None, palette=None) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    if palette is None:
        palette = [c for c in range(10) if c != bgc and c != 0]
        random.shuffle(palette)

    gh_ub = max(3, min(10, max_h // 2))
    gw_ub = max(3, min(10, max_w // 2))
    gh = unifint(diff_lb, diff_ub, (3, gh_ub))
    gw = unifint(diff_lb, diff_ub, (3, gw_ub))
    h = unifint(diff_lb, diff_ub, (min(gh * 2, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(gw * 2, max_w), max_w))

    ncols = unifint(diff_lb, diff_ub, (1, min(9, len(palette))))
    ccols = list(palette[:ncols])

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]

    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]

    for c in ccols:
        if len(remring) < 3:
            break
        nc = unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))
        obj = set(sample(totuple(remring), nc))
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                objn = normalize(obj)
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Every colour-object slides into the frame (the gh x gw rectangle marked by the
    4 corner cells), then the canvas is cropped down to that frame."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def cells_of(G, col):
        return [(int(r), int(c)) for r, c in zip(*np.where(G == col))]

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    icols = [int(v) for v in np.unique(I) if int(v) != bgc]
    iobj = {c: cells_of(I, c) for c in icols}

    # ---- locate the frame: exactly 4 cells sitting on the corners of an ho x wo box
    R0 = C0 = None
    for col in icols:
        cells = iobj[col]
        if len(cells) != 4:
            continue
        r0, c0, hh, ww = bbox(cells)
        if hh != ho or ww != wo:
            continue
        if set(cells) == {(r0, c0), (r0, c0 + ww - 1), (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
            R0, C0 = r0, c0
            break
    if R0 is None:
        for col in icols:
            cells = iobj[col]
            r0, c0, hh, ww = bbox(cells)
            if len(cells) == 4 and set(cells) == {(r0, c0), (r0, c0 + ww - 1),
                                                  (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
                R0, C0 = r0, c0
                break
    if R0 is None:
        R0, C0 = 0, 0
    R0 = max(0, min(R0, hi - ho))
    C0 = max(0, min(C0, wi - wo))

    def in_win(p):
        return R0 <= p[0] < R0 + ho and C0 <= p[1] < C0 + wo

    # ---- where each colour has to end up (absolute coords inside the frame)
    dest = {}
    for col in icols:
        ocells = cells_of(O, col)
        if not ocells:
            continue
        orr = min(r for r, _ in ocells)
        occ = min(c for _, c in ocells)
        dest[col] = (R0 + orr, C0 + occ)

    grid = I.copy()
    cur = {c: list(v) for c, v in iobj.items()}

    # ---- colours that do not survive into the frame: clear the ones lying inside it
    for col in icols:
        if col in dest:
            continue
        rem = [p for p in cur[col] if in_win(p)]
        if rem:
            ops.append(int(bgc))
            sels.append(sel_of(sorted(rem)))
            for (r, c) in rem:
                grid[r, c] = bgc
            cur[col] = [p for p in cur[col] if not in_win(p)]

    def delta(col):
        r0 = min(r for r, _ in cur[col])
        c0 = min(c for _, c in cur[col])
        return dest[col][0] - r0, dest[col][1] - c0

    def do_move(col, dr, dc):
        if dr == 0 and dc == 0:
            return
        src = list(cur[col])
        tgt = [(r + dr, c + dc) for r, c in src]
        if col == 0:
            # ARCLE's object ops only grab nonzero cells -> a 0-coloured object
            # cannot be translated; paint it at its destination instead.
            ops.append(0)
            sels.append(sel_of(sorted(tgt)))
        else:
            seq = [21 if dr > 0 else 20] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
            for k, op in enumerate(seq):
                ops.append(op)
                # first step GRABS the object; later steps keep the same grab (empty sel)
                sels.append(sel_of(sorted(src)) if k == 0 else sel_of([]))
        for (r, c) in src:
            grid[r, c] = 0
        for (r, c) in tgt:
            grid[r, c] = col
        cur[col] = tgt
        # only the vacated footprint reads 0 (ARCLE restored the path); repair the part
        # of it that lies inside the frame -- the rest is discarded by the final crop.
        hole = sorted(p for p in (set(src) - set(tgt)) if in_win(p))
        if bgc != 0 and hole:
            ops.append(int(bgc))
            sels.append(sel_of(hole))
            for (r, c) in hole:
                grid[r, c] = bgc

    def blocked(col):
        dr, dc = delta(col)
        src = set(cur[col])
        return any(grid[r + dr, c + dc] != bgc and (r + dr, c + dc) not in src for r, c in src)

    pending = [c for c in icols if c in dest and cur.get(c)]
    guard = 0
    while pending and guard < 200:
        guard += 1
        progressed = False
        for col in list(pending):
            dr, dc = delta(col)
            if dr == 0 and dc == 0:
                pending.remove(col)
                progressed = True
                continue
            if not blocked(col):
                do_move(col, dr, dc)
                pending.remove(col)
                progressed = True
        if progressed or not pending:
            continue
        # rare deadlock: park a blocking object on free background outside the frame
        tcol = pending[0]
        dr, dc = delta(tcol)
        src = set(cur[tcol])
        blockers = []
        for (r, c) in src:
            p = (r + dr, c + dc)
            v = int(grid[p[0], p[1]])
            if v != bgc and p not in src and v in pending and v not in blockers:
                blockers.append(v)
        parked = False
        for b in blockers:
            bc = cur[b]
            br, bcc, bh, bw = bbox(bc)
            norm = [(r - br, c - bcc) for r, c in bc]
            for rr in range(hi - bh + 1):
                for cc in range(wi - bw + 1):
                    if (rr, cc) == (br, bcc):
                        continue
                    cand = [(rr + a, cc + bb) for a, bb in norm]
                    if any(in_win(p) for p in cand):
                        continue
                    if any(grid[p[0], p[1]] != bgc for p in cand):
                        continue
                    do_move(b, rr - br, cc - bcc)
                    parked = True
                    break
                if parked:
                    break
            if parked:
                break
        if not parked:
            do_move(tcol, dr, dc)
            pending.remove(tcol)

    # ---- crop down to the frame (full rectangle: whole window, background included)
    ops.append(33)
    sels.append([R0, C0, ho - 1, wo - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
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
                                f"for task c8cbb738"
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
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
