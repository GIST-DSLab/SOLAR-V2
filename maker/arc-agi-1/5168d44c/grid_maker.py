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


# ---------------------------------------------------------------- colors / plan
def sample_colors(num_examples=None) -> dict:
    """bgc / dotcol / boxcol are the three colors the RE-ARC generator samples.
    The structural variant of this task is the direction of the dot chain
    (DOWN / RIGHT / UNITY) -> planned per instance so every case is shown."""
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dotcol = random.choice(rem)
    rem = [c for c in rem if c != dotcol]
    boxcol = random.choice(rem)

    VARIANTS = [{"direc": "down"}, {"direc": "right"}, {"direc": "unity"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol, direc=None) -> dict:
    try:
        _uf = unifint  # noqa: F821  (RE-ARC helper)
    except NameError:
        def _uf(lb, ub, rng):
            return random.randint(rng[0], rng[1])

    dirmap = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}
    if direc is None:
        direc = random.choice(["down", "right", "unity"])
    dv = dirmap[direc] if isinstance(direc, str) else tuple(direc)

    hmax = max(7, min(30, int(max_h)))
    wmax = max(7, min(30, int(max_w)))
    h = _uf(diff_lb, diff_ub, (7, hmax))
    w = _uf(diff_lb, diff_ub, (7, wmax))
    doth = _uf(diff_lb, diff_ub, (1, max(1, h // 3)))
    dotw = _uf(diff_lb, diff_ub, (1, max(1, w // 3)))
    borderh = _uf(diff_lb, diff_ub, (1, max(1, h // 4)))
    borderw = _uf(diff_lb, diff_ub, (1, max(1, w // 4)))

    hi_i = h - doth - 1 if dv == (0, 1) else h - doth - borderh - 1
    hi_j = w - dotw - 1 if dv == (1, 0) else w - dotw - borderw - 1
    dotloci = random.randint(0, max(0, hi_i))
    dotlocj = random.randint(0, max(0, hi_j))

    offr = dv[0] * (doth + borderh)
    offc = dv[1] * (dotw + borderw)

    gi = [[bgc] * w for _ in range(h)]
    for k in range(-15, 16):
        r0 = dotloci + k * offr
        c0 = dotlocj + k * offc
        for r in range(r0, r0 + doth):
            for c in range(c0, c0 + dotw):
                if 0 <= r < h and 0 <= c < w:
                    gi[r][c] = dotcol

    box = [(r, c)
           for r in range(dotloci - borderh, dotloci + doth + borderh)
           for c in range(dotlocj - borderw, dotlocj + dotw + borderw)
           if not (dotloci <= r < dotloci + doth and dotlocj <= c < dotlocj + dotw)]

    go = [row[:] for row in gi]
    for (r, c) in box:
        rr, cc = r + offr, c + offc
        if 0 <= rr < h and 0 <= cc < w:
            go[rr][cc] = boxcol
    for (r, c) in box:
        if 0 <= r < h and 0 <= c < w:
            gi[r][c] = boxcol

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ---------------------------------------------------------------- operations
def derive_operations(I, O):
    """The box (a rectangular ring drawn around one dot of a regularly spaced dot
    chain) is TRANSLATED by exactly one dot-spacing step, so it ends up framing the
    next dot.  -> grab the ring and slide it (Move), repair the vacated footprint,
    and draw the part of the ring that scrolls in from off-grid."""
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def cells_of(G, c):
        return {(int(r), int(cc)) for r, cc in zip(*np.where(G == c))}

    def runs(vals):
        out = []
        for v in sorted(vals):
            if out and v == out[-1][1] + 1:
                out[-1][1] = v
            else:
                out.append([v, v])
        return out

    changed = [(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]]
    inv = sorted({int(I[r, c]) for r, c in changed} | {int(O[r, c]) for r, c in changed})
    pal = sorted({int(v) for v in np.unique(I)})
    if len(inv) == 2:
        pairs = [(inv[0], inv[1]), (inv[1], inv[0])]
    else:
        pairs = [(a, b) for a in pal for b in pal if a != b]

    sol = None
    for boxc, bgc in pairs:
        S = cells_of(I, boxc)                      # ring as visible in I
        if not S:
            continue
        sr0 = min(p[0] for p in S); sr1 = max(p[0] for p in S)
        sc0 = min(p[1] for p in S); sc1 = max(p[1] for p in S)
        # the hole inside the ring's bbox is the framed dot
        inner = [(r, c) for r in range(sr0, sr1 + 1) for c in range(sc0, sc1 + 1)
                 if int(I[r, c]) not in (boxc, bgc)]
        if not inner:
            continue
        dr0 = min(p[0] for p in inner); dr1 = max(p[0] for p in inner)
        dc0 = min(p[1] for p in inner); dc1 = max(p[1] for p in inner)
        if len(inner) != (dr1 - dr0 + 1) * (dc1 - dc0 + 1):
            continue
        dotc = int(I[inner[0]])
        if any(int(I[p]) != dotc for p in inner):
            continue
        # border thickness: at most one side of each axis can be clipped
        bh = max(dr0 - sr0, sr1 - dr1)
        bw = max(dc0 - sc0, sc1 - dc1)
        if bh < 1 or bw < 1:
            continue
        # full (unclipped) ring geometry
        R_full = set()
        for r in range(dr0 - bh, dr1 + bh + 1):
            for c in range(dc0 - bw, dc1 + bw + 1):
                if dr0 <= r <= dr1 and dc0 <= c <= dc1:
                    continue
                R_full.add((r, c))
        if {p for p in R_full if 0 <= p[0] < h and 0 <= p[1] < w} != S:
            continue

        # candidate step vectors: first the dot lattice spacing, then a search
        cand = []
        D = cells_of(I, dotc)
        if D:
            rb = runs({p[0] for p in D}); cb = runs({p[1] for p in D})
            orr = occ = 0
            if len(rb) > 1:
                orr = max(b - a + 1 for a, b in rb) + \
                      min(rb[i + 1][0] - rb[i][1] - 1 for i in range(len(rb) - 1))
            if len(cb) > 1:
                occ = max(b - a + 1 for a, b in cb) + \
                      min(cb[i + 1][0] - cb[i][1] - 1 for i in range(len(cb) - 1))
            if (orr, occ) != (0, 0):
                cand.append((orr, occ))
        for dr in range(h):
            for dc in range(w):
                if (dr, dc) != (0, 0) and (dr, dc) not in cand:
                    cand.append((dr, dc))

        n_box_o = int(np.count_nonzero(O == boxc))
        for (dr, dc) in cand:
            dest = {(r + dr, c + dc) for (r, c) in R_full
                    if 0 <= r + dr < h and 0 <= c + dc < w}
            if not dest or len(dest) != n_box_o:
                continue
            P = I.copy()
            for (r, c) in S:
                P[r, c] = bgc
            for (r, c) in dest:
                P[r, c] = boxc
            if np.array_equal(P, O):
                sol = (boxc, bgc, dotc, S, dest, dr, dc)
                break
        if sol is not None:
            break

    if sol is None:
        # defensive fallback: paint the differing cells grouped by target colour
        by_col = {}
        for (r, c) in changed:
            by_col.setdefault(int(O[r, c]), []).append((r, c))
        for col in sorted(by_col):
            ops.append(col); sels.append(sel_of(sorted(by_col[col])))
        ops.append(34); sels.append([0, 0, h - 1, w - 1])  # full-grid bbox
        return ops, sels

    boxc, bgc, dotc, S, R_dest, off_r, off_c = sol
    src = sorted(S)
    dst_final = {(r + off_r, c + off_c) for (r, c) in S
                 if 0 <= r + off_r < h and 0 <= c + off_c < w}
    vacated = sorted(S - dst_final)
    missing = sorted(R_dest - dst_final)

    # ARCLE's object ops ignore 0-valued cells: make a 0-coloured ring grabbable
    temp = None
    if boxc == 0:
        temp = next(c for c in range(1, 10) if c not in (bgc, dotc))
        ops.append(temp); sels.append(sel_of(src))

    # slide the ring one dot-spacing step: grab once, then continue with empties
    first = True
    for _ in range(off_r):
        ops.append(21); sels.append(sel_of(src) if first else sel_of([])); first = False
    for _ in range(off_c):
        ops.append(22); sels.append(sel_of(src) if first else sel_of([])); first = False

    # the grab zeroed the ring's original footprint; restore what it no longer covers
    if vacated and bgc != 0:
        ops.append(bgc); sels.append(sel_of(vacated))

    if boxc == 0:
        # restore the ring's real colour at its new position (temp -> 0), and this
        # also draws the part of the ring that scrolled in from off-grid
        ops.append(0); sels.append(sel_of(sorted(R_dest)))
    elif missing:
        # the ring's slice that was off-grid in I and is on-grid at the destination
        ops.append(boxc); sels.append(sel_of(missing))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # full-grid bbox
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
