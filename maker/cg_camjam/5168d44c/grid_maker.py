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

# How the golf entry reads the task, compared with the maker:
#   The maker treats the FRAME as the unit. It grabs only the frame and slides it
#   one period along the dot line.
#   The golf entry treats the WHOLE CANVAS as the unit. Leave out the frame and the
#   picture repeats along the dot line with period (dot + border). Scrolling
#   everything by exactly one period puts every dot on the next dot and carries the
#   frame onto the next dot. Nothing else changes.
#   The only new content is the band that scrolls in at the leading edge. It is
#   refilled from structure read off I: background first, then the lattice dots
#   inside the band, then any frame cells that were off-canvas in I.
#   Where the two readings agree, nothing changes: the dot lattice, the frame and
#   the period are found from I the same way the maker finds them.

_DVEC = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}
VARIANTS = [{"direc": "down"}, {"direc": "right"}, {"direc": "unity"}]


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotcol = random.choice([c for c in cols if c != bgc])
    boxcol = random.choice([c for c in cols if c not in (bgc, dotcol)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol, direc=None) -> dict:
    if direc is None:
        direc = random.choice(VARIANTS)["direc"]
    di, dj = _DVEC[direc]
    h = _unif(diff_lb, diff_ub, (7, max_h))
    w = _unif(diff_lb, diff_ub, (7, max_w))
    doth = _unif(diff_lb, diff_ub, (1, h // 3))
    dotw = _unif(diff_lb, diff_ub, (1, w // 3))
    borderh = _unif(diff_lb, diff_ub, (1, h // 4))
    borderw = _unif(diff_lb, diff_ub, (1, w // 4))
    dotloci = random.randint(0, h - doth - 1 if direc == "right" else h - doth - borderh - 1)
    dotlocj = random.randint(0, w - dotw - 1 if direc == "down" else w - dotw - borderw - 1)
    orr, occ = di * (doth + borderh), dj * (dotw + borderw)
    gi = [[bgc] * w for _ in range(h)]
    for k in range(-15, 16):
        for r in range(dotloci + k * orr, dotloci + k * orr + doth):
            for c in range(dotlocj + k * occ, dotlocj + k * occ + dotw):
                if 0 <= r < h and 0 <= c < w:
                    gi[r][c] = dotcol
    bx = [(r, c)
          for r in range(dotloci - borderh, dotloci + doth + borderh)
          for c in range(dotlocj - borderw, dotlocj + dotw + borderw)
          if not (dotloci <= r < dotloci + doth and dotlocj <= c < dotlocj + dotw)]
    go = [row[:] for row in gi]
    for r, c in bx:
        if 0 <= r + orr < h and 0 <= c + occ < w:
            go[r + orr][c + occ] = boxcol
        if 0 <= r < h and 0 <= c < w:
            gi[r][c] = boxcol
    return {"input": tuple(map(tuple, gi)), "output": tuple(map(tuple, go))}


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

    # Find the structure from I alone: the dot lattice, the anchor dot that the
    # frame surrounds, the frame size, and the period of the lattice.
    colors = sorted(set(int(v) for v in I.flatten().tolist()))
    hyp = None
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
                    for r in range(ar0 + k * orr, ar0 + k * orr + ah):
                        for c in range(ac0 + k * occ, ac0 + k * occ + aw):
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
        raise ValueError("no dot-lattice/frame structure found in I")

    boxcol, dotcol = int(hyp['boxcol']), int(hyp['dotcol'])
    rest = [c for c in colors if c not in (boxcol, dotcol)]
    bgc = int(rest[0]) if rest else 0
    orr, occ = hyp['off']
    R0, R1, C0, C1 = hyp['rect']
    ar0, ac0, ah, aw = hyp['anchor']

    state = I.copy()

    def emit(op, cells):
        # Paint a named region whole; skip the op only if it would change nothing now.
        nonlocal state
        cells = sorted(cells)
        if not cells:
            return
        ns = state.copy()
        for r, c in cells:
            ns[r, c] = op
        if np.array_equal(ns, state):
            return
        ops.append(op)
        sels.append(sel_of(cells))
        state = ns

    # 1. Scroll the WHOLE canvas one period along the dot line. The first Move
    #    grabs the full grid: the selection is the full rectangle, background
    #    included, on purpose. Later Moves use empty selections so the same grab
    #    keeps sliding. ARCLE clips whatever leaves the grid; 0-valued cells
    #    land as 0 because the snapshot is all-zero.
    def shifted(dr, dc):
        g = np.zeros((h, w), dtype=int)
        g[dr:, dc:] = I[:h - dr, :w - dc]
        return g

    first = True
    pos = (0, 0)
    for (sr, sc, cnt, mop) in ((1, 0, orr, 21), (0, 1, occ, 22)):   # 21=MoveD, 22=MoveR
        for _ in range(cnt):
            npos = (pos[0] + sr, pos[1] + sc)
            ns = shifted(*npos)
            if np.array_equal(ns, state):
                break
            ops.append(mop)
            sels.append([0, 0, h - 1, w - 1] if first else sel_of([]))
            first = False
            pos, state = npos, ns

    # 2. Refill the band that scrolled in at the leading edge (rows < orr and/or
    #    cols < occ). Lay the background first, then draw the lattice dots
    #    inside the band, then the part of the next frame that was off-canvas in I.
    band = {(r, c) for r in range(h) for c in range(w) if r < orr or c < occ}
    emit(bgc, band)
    emit(dotcol, band & hyp['dcells'])
    frame_next = {(r, c)
                  for r in range(max(R0 + orr, 0), min(R1 + orr, h - 1) + 1)
                  for c in range(max(C0 + occ, 0), min(C1 + occ, w - 1) + 1)}
    frame_next -= {(r, c) for r in range(ar0 + orr, ar0 + orr + ah)
                   for c in range(ac0 + occ, ac0 + occ + aw)}
    emit(boxcol, band & frame_next)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # full-grid bbox: submit
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
