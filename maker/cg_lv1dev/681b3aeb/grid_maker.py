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
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    # piece colours: anything but the background; 0 is excluded because ARCLE
    # cannot carry a colour-0 object with Move
    pcols = [x for x in cols if x != bgc and x != 0]
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, min(8, max_h // 3)))
        wi = unifint(diff_lb, diff_ub, (2, min(8, max_w // 3)))
        h = unifint(diff_lb, diff_ub, (3 * hi, max_h))
        w = unifint(diff_lb, diff_ub, (3 * wi, max_w))
        c = canvas(-1, (hi, hi))
        if ca is not None and cb is not None:
            cA, cB = ca, cb
        else:
            cA, cB = sample(pcols, 2)
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
        go = fill(c, cA, A)
        go = fill(go, cB, B)
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
                # fixed: original shifted by (locaj, locaj), which could push
                # piece A partly off the grid
                plcda = shift(A, (locai, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(remlocs2)
                plcdb = shift(B, (ch))
                if set(plcdb).issubset(set(remlocs2)):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, cA, plcda)
    gi = fill(gi, cB, plcdb)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    """Two pieces lie on a background.  Read as in the golf entry: the answer is
    one square frame; one piece sits in it, and every OTHER cell of the frame
    belongs to the second piece.  So: find the frame (side s) and how both
    pieces sit in it, carry the pieces into one s x s window of the grid with
    Move, paint any frame cells the second piece could not bring along (a piece
    that the grid edge cut off), and crop the window.  O is never inspected."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- background: the colour every input of the episode is painted with
    grids = [I]
    if examples:
        for pair in examples:
            try:
                grids.append(np.asarray(pair[0], dtype=int))
            except Exception:
                pass
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]
    if not bool((I == bgc).any()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- the two pieces: all cells of each non-background colour
    pcols = sorted(set(I.flatten().tolist()) - {bgc})
    if len(pcols) != 2:
        return [34], [[0, 0, h - 1, w - 1]]
    piece = {}
    for col in pcols:
        cells = sorted((int(r), int(c)) for r, c in zip(*np.nonzero(I == col)))
        ul = (min(r for r, _ in cells), min(c for _, c in cells))
        norm = {(r - ul[0], c - ul[1]) for r, c in cells}
        dims = (max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1)
        piece[col] = (cells, ul, norm, dims)

    # ---- the frame: an s x s square = piece Y + (everything else) = piece X.
    # Pass 1: X is fully visible (the rest of the frame IS X).
    # Pass 2: X was cut by the grid edge; the visible part of X must be exactly
    #         what remains on the grid of the full frame-complement placed at a
    #         diagonal origin (how the original generator dropped it).
    def tilings(clipped):
        out = []
        smin = max(max(piece[c][3]) for c in pcols)
        for s in range(smin, max(h, w) + 1):
            for colY in pcols:
                colX = [c for c in pcols if c != colY][0]
                _, ulY, nY, (hY, wY) = piece[colY]
                cX, ulX, nX, (hX, wX) = piece[colX]
                if hY > s or wY > s or hX > s or wX > s:
                    continue
                for pr in range(s - hY + 1):
                    for pc in range(s - wY + 1):
                        Ysq = {(r + pr, c + pc) for r, c in nY}
                        F = {(r, c) for r in range(s) for c in range(s)} - Ysq
                        if not clipped:
                            if len(F) != len(nX):
                                continue
                            fr = min(r for r, _ in F)
                            fc = min(c for _, c in F)
                            if {(r - fr, c - fc) for r, c in F} == nX:
                                out.append((s, colY, colX, (pr, pc), (fr, fc), F))
                            continue
                        if len(F) <= len(nX):
                            continue
                        fr0 = min(r for r, _ in F)
                        fc0 = min(c for _, c in F)
                        for orr in range(s - hX + 1):
                            for oc in range(s - wX + 1):
                                if not all((r + orr, c + oc) in F for r, c in nX):
                                    continue
                                g = (ulX[0] - orr, ulX[1] - oc)
                                if g[0] + fr0 != g[1] + fc0:
                                    continue
                                vis = {(r + g[0], c + g[1]) for r, c in F
                                       if 0 <= r + g[0] < h and 0 <= c + g[1] < w}
                                if vis == set(cX):
                                    out.append((s, colY, colX, (pr, pc), (orr, oc), F))
            if out:
                return out
        return out

    cands = tilings(False) or tilings(True)
    if not cands:
        return [34], [[0, 0, h - 1, w - 1]]

    # ---- where to assemble: prefer leaving a piece in place, moving few cells,
    # and an order in which no piece lands on the other's not-yet-moved cells
    best = None
    for (s, colY, colX, py, ox, F) in cands:
        cY, ulY, nY, _ = piece[colY]
        cX, ulX, nX, _ = piece[colX]
        for R0 in range(0, h - s + 1):
            for C0 in range(0, w - s + 1):
                dY = (R0 + py[0] - ulY[0], C0 + py[1] - ulY[1])
                dX = (R0 + ox[0] - ulX[0], C0 + ox[1] - ulX[1])
                info = {colY: (cY, dY, None), colX: (cX, dX, F)}
                movers = [c for c in (colY, colX) if info[c][1] != (0, 0)]
                dst = {c: {(r + info[c][1][0], cc + info[c][1][1]) for r, cc in info[c][0]}
                       for c in (colY, colX)}
                src = {c: set(info[c][0]) for c in (colY, colX)}
                if len(movers) == 2:
                    a, b = movers
                    if not (dst[a] & src[b]):
                        order = [a, b]
                    elif not (dst[b] & src[a]):
                        order = [b, a]
                    else:
                        continue
                else:
                    order = movers
                cost = (1000 * sum(1 for m in movers if m == 0) + 100 * len(movers)
                        + sum(abs(info[m][1][0]) + abs(info[m][1][1]) for m in movers))
                if best is None or cost < best[0]:
                    best = (cost, R0, C0, s, order, info, colX, F)
    if best is None:
        return [34], [[0, 0, h - 1, w - 1]]
    _, R0, C0, s, order, info, colX, F = best

    ops, sels = [], []
    grid = I.copy()
    for col in order:
        cells, d, _ = info[col]
        if col != 0:
            steps = [(20 if d[0] < 0 else 21, (-1 if d[0] < 0 else 1, 0))] * abs(d[0])
            steps += [(23 if d[1] < 0 else 22, (0, -1 if d[1] < 0 else 1))] * abs(d[1])
            cur = list(cells)
            for k, (op, (sr, sc)) in enumerate(steps):
                ops.append(op)
                # first step grabs the piece; later steps continue the same grab
                sels.append(sel_of(cur) if k == 0 else sel_of([]))
                cur = [(r + sr, c + sc) for r, c in cur]
            for r, c in cells:
                grid[r, c] = 0
            for r, c in cur:
                grid[r, c] = col
            # the vacated footprint (left at 0) lies outside the window or is
            # covered by a piece, so the crop below makes it irrelevant
        else:
            # a colour-0 piece cannot be carried by Move: paint it at its slot
            full = F if col == colX else None
            if full is None:
                dst = [(r + d[0], c + d[1]) for r, c in cells]
            else:
                dst = [(r + R0, c + C0) for r, c in F]
            dst = sorted(p for p in dst if grid[p] != 0)
            if dst:
                ops.append(0)
                sels.append(sel_of(dst))
                for p in dst:
                    grid[p] = 0

    # ---- frame cells still holding neither piece colour: the part of piece X
    # the grid edge had cut away.  Paint them X's colour.
    if colX != 0:
        rest = sorted((r, c) for r in range(R0, R0 + s) for c in range(C0, C0 + s)
                      if grid[r, c] not in pcols)
        if rest:
            ops.append(colX)
            sels.append(sel_of(rest))
            for p in rest:
                grid[p] = colX

    # ---- keep only the assembled square (selection = that full rectangle)
    ops.append(33)
    sels.append([R0, C0, s - 1, s - 1])
    ops.append(34)
    sels.append([0, 0, s - 1, s - 1])
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
