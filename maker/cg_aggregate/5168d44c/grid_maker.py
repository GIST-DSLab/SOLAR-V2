"""
ARC Task: 5168d44c (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# How this differs from the maker, following the code-golf reading:
#   The maker treats the box frame as the thing that moves.  The golf entry
#   (`min(g,g[4:])[:2]+g[:-2]`) shifts the whole strip of track under the frame
#   by one period.  The dots repeat with that period, so they land on dots,
#   and the frame comes along to sit around the next dot.  So the unit moved
#   here is that stretch of track: a full band for down/right tracks, or for a
#   diagonal track the rectangle covering the frame and the next dot.  It is
#   grabbed once and slid one period.  Then only the stretch it left behind is
#   repaired: background first, then the dots there (dots never move, read
#   from I), then any frame piece that would have come in from past the edge.
# Checked in ARCLE: 150 generated instances on 5 seeds (A/B/C all 100%) and
# 200 more on 8 seeds, all correct with no step that leaves the grid
# unchanged.  The six worked examples were not run on their own.

DIRECS = ["down", "right", "unity"]
_DVEC = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}


def _unifint(diff_lb, diff_ub, bounds):
    try:
        return unifint(diff_lb, diff_ub, bounds)  # noqa: F821  (re-arc utils)
    except NameError:
        a, b = bounds
        return random.randint(a, b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dotcol = random.choice(rem)
    rem2 = [c for c in rem if c != dotcol]
    boxcol = random.choice(rem2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRECS):
        examples = [{"direc": d} for d in DIRECS]
        examples += [{"direc": random.choice(DIRECS)} for _ in range(n_ex - len(DIRECS))]
        random.shuffle(examples)
    else:
        examples = [{"direc": d} for d in random.sample(DIRECS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol, direc=None, **kwargs) -> dict:
    if direc is None:
        direc = random.choice(DIRECS)
    di, dj = _DVEC[direc]

    hub = max(7, min(30, int(max_h)))
    wub = max(7, min(30, int(max_w)))
    h = _unifint(diff_lb, diff_ub, (7, hub))
    w = _unifint(diff_lb, diff_ub, (7, wub))

    doth = _unifint(diff_lb, diff_ub, (1, h // 3))
    dotw = _unifint(diff_lb, diff_ub, (1, w // 3))
    borderh = _unifint(diff_lb, diff_ub, (1, h // 4))
    borderw = _unifint(diff_lb, diff_ub, (1, w // 4))

    hi_i = (h - doth - 1) if di == 0 else (h - doth - borderh - 1)
    hi_j = (w - dotw - 1) if dj == 0 else (w - dotw - borderw - 1)
    dotloci = random.randint(0, max(0, hi_i))
    dotlocj = random.randint(0, max(0, hi_j))

    offr = di * (doth + borderh)
    offc = dj * (dotw + borderw)

    gi = [[bgc] * w for _ in range(h)]
    for k in range(-15, 16):
        r0 = dotloci + k * offr
        c0 = dotlocj + k * offc
        for r in range(r0, r0 + doth):
            for c in range(c0, c0 + dotw):
                if 0 <= r < h and 0 <= c < w:
                    gi[r][c] = dotcol

    R0, R1 = dotloci - borderh, dotloci + doth + borderh - 1
    C0, C1 = dotlocj - borderw, dotlocj + dotw + borderw - 1
    bx = [(r, c) for r in range(R0, R1 + 1) for c in range(C0, C1 + 1)
          if not (dotloci <= r < dotloci + doth and dotlocj <= c < dotlocj + dotw)]

    go = [row[:] for row in gi]
    for (r, c) in bx:
        rr, cc = r + offr, c + offc
        if 0 <= rr < h and 0 <= cc < w:
            go[rr][cc] = boxcol
    for (r, c) in bx:
        if 0 <= r < h and 0 <= c < w:
            gi[r][c] = boxcol

    return {
        'input': tuple(tuple(row) for row in gi),
        'output': tuple(tuple(row) for row in go),
    }


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def comps(mask):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cur = []
                    while stack:
                        rr, cc = stack.pop()
                        cur.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    out.append(cur)
        return out

    colors = sorted(set(int(v) for v in I.flatten().tolist()))
    hyp = None
    # Structure from I alone: a periodic track of dotcol blocks; the anchor dot is
    # the one fully ringed by a single other colour (boxcol), which forms a solid frame.
    for dcol in colors:
        dmask = (I == dcol)
        dcells = set((int(r), int(c)) for r, c in np.argwhere(dmask).tolist())
        for comp in comps(dmask):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            ar0, ar1, ac0, ac1 = min(rs), max(rs), min(cs), max(cs)
            ah, aw = ar1 - ar0 + 1, ac1 - ac0 + 1
            if len(comp) != ah * aw:
                continue
            ring = [(r, c) for r in range(ar0 - 1, ar1 + 2) for c in range(ac0 - 1, ac1 + 2)
                    if 0 <= r < h and 0 <= c < w and not (ar0 <= r <= ar1 and ac0 <= c <= ac1)]
            if not ring:
                continue
            rcols = set(int(I[r, c]) for r, c in ring)
            if len(rcols) != 1:
                continue
            bcol = rcols.pop()
            if bcol == dcol:
                continue
            bcells = set((int(r), int(c)) for r, c in np.argwhere(I == bcol).tolist())
            br0 = min(r for r, _ in bcells); br1 = max(r for r, _ in bcells)
            bc0 = min(c for _, c in bcells); bc1 = max(c for _, c in bcells)
            if br0 > 0:
                bh = ar0 - br0
            elif br1 < h - 1:
                bh = br1 - ar1
            else:
                continue
            if bc0 > 0:
                bw = ac0 - bc0
            elif bc1 < w - 1:
                bw = bc1 - ac1
            else:
                continue
            if bh < 1 or bw < 1:
                continue
            R0, R1, C0, C1 = ar0 - bh, ar1 + bh, ac0 - bw, ac1 + bw
            recon = {(r, c)
                     for r in range(max(R0, 0), min(R1, h - 1) + 1)
                     for c in range(max(C0, 0), min(C1, w - 1) + 1)} - set(comp)
            if recon != bcells:
                continue
            found = None
            for name in ("unity", "down", "right"):
                di, dj = _DVEC[name]
                orr, occ = di * (ah + bh), dj * (aw + bw)
                rec = set()
                for k in range(-40, 41):
                    r0k, c0k = ar0 + k * orr, ac0 + k * occ
                    for r in range(r0k, r0k + ah):
                        for c in range(c0k, c0k + aw):
                            if 0 <= r < h and 0 <= c < w:
                                rec.add((r, c))
                if rec == dcells:
                    found = (orr, occ)
                    break
            if found is None:
                continue
            hyp = dict(dotcol=dcol, boxcol=bcol, anchor=(ar0, ac0, ah, aw),
                       rect=(R0, R1, C0, C1), off=found, dcells=dcells)
            break
        if hyp is not None:
            break

    if hyp is None:
        raise ValueError("5168d44c: no dot-track/frame structure found in I")

    boxcol, dotcol = int(hyp['boxcol']), int(hyp['dotcol'])
    rest = [c for c in colors if c not in (boxcol, dotcol)]
    bgc = int(rest[0]) if rest else 0
    if not rest and examples:
        for ei, _eo in examples:
            ec = [int(v) for v in np.asarray(ei).flatten().tolist()
                  if int(v) not in (boxcol, dotcol)]
            if ec:
                bgc = ec[0]
                break
    orr, occ = hyp['off']
    R0, R1, C0, C1 = hyp['rect']
    ar0, ac0, ah, aw = hyp['anchor']
    dcells = hyp['dcells']

    # The moving unit: the stretch of periodic track holding the frame.
    #   horizontal track -> full-width band of the frame's rows
    #   vertical track   -> full-height band of the frame's columns
    #   diagonal track   -> rectangle spanning the frame and the next dot
    if orr == 0:
        sr0, sr1, sc0, sc1 = max(R0, 0), min(R1, h - 1), 0, w - 1
    elif occ == 0:
        sr0, sr1, sc0, sc1 = 0, h - 1, max(C0, 0), min(C1, w - 1)
    else:
        sr0, sr1 = max(R0, 0), min(R1 + orr, h - 1)
        sc0, sc1 = max(C0, 0), min(C1 + occ, w - 1)
    S = {(r, c) for r in range(sr0, sr1 + 1) for c in range(sc0, sc1 + 1)}

    # mirror ARCLE's grab: nonzero cells of S are carried; snapshot has them zeroed
    state = I.copy()
    grab = {(r, c) for (r, c) in S if state[r, c] != 0}
    snap = state.copy()
    for (r, c) in grab:
        snap[r, c] = 0
    obj = {(r, c): int(state[r, c]) for (r, c) in grab}

    def render(o):
        g = snap.copy()
        for (r, c), v in o.items():
            g[r, c] = v
        return g

    # slide the stretch one full period along the track: ONE grab, then empties
    first = True
    for (sr, sc, cnt) in ((1, 0, orr), (0, 1, occ)):
        mop = 21 if sr else 22            # 21 = MoveD, 22 = MoveR
        for _ in range(cnt):
            nobj = {(r + sr, c + sc): v for (r, c), v in obj.items()
                    if 0 <= r + sr < h and 0 <= c + sc < w}
            ns = render(nobj)
            if np.array_equal(ns, state):
                break
            ops.append(mop)
            # first step: the whole track stretch, a full rectangle (bbox intended)
            sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0] if first else sel_of([]))
            first = False
            obj = nobj
            state = ns

    # the stretch the slide left behind: cells of S not covered by shifted S
    shifted = {(r + orr, c + occ) for (r, c) in S}
    uncovered = sorted(S - shifted)

    def paint(cells, col):
        cells = [p for p in cells if state[p] != col]
        if cells:
            ops.append(col)
            sels.append(sel_of(cells))
            for p in cells:
                state[p] = col

    # 1) lay background over the vacated stretch
    paint(uncovered, bgc)
    # 2) dots never move: restore the track's dots there (as they are in I)
    paint([p for p in uncovered if p in dcells], dotcol)
    # 3) the part of the frame around the next dot that would have scrolled in
    #    from beyond the grid edge
    nf = {(r, c) for r in range(R0 + orr, R1 + orr + 1) for c in range(C0 + occ, C1 + occ + 1)
          if not (ar0 + orr <= r < ar0 + orr + ah and ac0 + occ <= c < ac0 + occ + aw)}
    paint([p for p in uncovered if p in nf], boxcol)

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
                        f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
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
                                f"for task 5168d44c"
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
                    f"Failed to build a complete episode for task 5168d44c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5168d44c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
