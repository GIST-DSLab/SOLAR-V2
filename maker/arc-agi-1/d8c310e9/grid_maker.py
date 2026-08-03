"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _hperiod(cells):
    """smallest T>0 with: cell at (i,j) implies same color at (i,j-T)  (DSL hperiod)."""
    if not cells:
        return 1
    ws = max(j for _, j in cells) + 1
    for T in range(1, ws):
        ok = True
        for (i, j), c in cells.items():
            if j - T >= 0 and cells.get((i, j - T)) != c:
                ok = False
                break
        if ok:
            return T
    return ws


# structural variant = which of the 4 rotations the whole scene is presented in
ROTS = [0, 1, 2, 3]          # np.rot90 k (CCW) applied to the canonical scene


# ----------------------------------------------------------- sample_colors ---
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


# ----------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    # canonical scene dims; a 90/270 rotation swaps them
    if rot in (1, 3):
        hlim, wlim = max_w, max_h
    else:
        hlim, wlim = max_h, max_w

    h = _unifint(diff_lb, diff_ub, (3, max(3, hlim)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, wlim)))
    p = _unifint(diff_lb, diff_ub, (2, max(2, (w - 1) // 3)))

    pool = [c for c in range(10) if c != bgc]
    numc = _unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(pool, numc)

    # base unit: p columns, each grown upward from the bottom row
    obj = {}
    for j in range(p):
        numcells = _unifint(diff_lb, diff_ub, (1, max(1, h - 1)))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj[(ii, j)] = random.choice(ccols)

    # doubled unit (period p), then shifted left a bit (clipped at the edge)
    full = dict(obj)
    for (i, j), c in obj.items():
        full[(i, j + p)] = c
    addonw = random.randint(0, p)
    leftshift = random.randint(0, addonw)

    P = np.full((h, w), bgc, dtype=int)
    for (i, j), c in full.items():
        jj = j - leftshift
        if 0 <= jj < w:
            P[i, jj] = c

    cells = {(i, j): int(P[i, j]) for i in range(h) for j in range(w) if P[i, j] != bgc}
    minc = min(j for _, j in cells)
    norm = {(i, j - minc): c for (i, j), c in cells.items()}
    T = _hperiod(norm)

    # output: the T-wide unit strip tiled rightward across the canvas
    G = P.copy()
    for j in range(minc, w):
        G[:, j] = P[:, minc + (j - minc) % T]

    gi = np.rot90(P, rot)
    go = np.rot90(G, rot)
    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


# --------------------------------------------------------- derive_operations ---
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # 1) find the canonical orientation the same way the rule defines it:
    #    the rotation whose FIRST ROW and LAST COLUMN are uniform background
    #    (object anchored bottom-left, repetition running rightward).
    k_sel = 0
    for k in range(4):
        Ck = np.rot90(I, k)
        if len(set(Ck[0].tolist())) == 1 and len(set(Ck[:, -1].tolist())) == 1:
            k_sel = k
            break
    C = np.rot90(I, k_sel)
    hc, wc = C.shape
    bg = int(C[0, 0])

    # canonical (i,j) -> original (r,c)
    def m(i, j):
        if k_sel == 0:
            return (i, j)
        if k_sel == 1:
            return (j, W - 1 - i)
        if k_sel == 2:
            return (H - 1 - i, W - 1 - j)
        return (H - 1 - j, i)

    # rectangle in canonical coords -> [r, c, h-1, w-1] in original coords
    # (these bboxes ARE full rectangles: whole bands of the grid, background included)
    def rect(i0, i1, j0, j1):
        pts = [m(i0, j0), m(i0, j1), m(i1, j0), m(i1, j1)]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        return [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]

    cells = {(i, j): int(C[i, j]) for i in range(hc) for j in range(wc) if C[i, j] != bg}
    if not cells:
        ops.append(34)
        sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    # 2) measure the repetition unit from the INPUT object itself
    minc = min(j for _, j in cells)
    norm = {(i, j - minc): c for (i, j), c in cells.items()}
    T = _hperiod(norm)

    # 3) tile that unit strip across the rest of the canvas
    blocks = []                                   # (dest_j0, q, needs_change)
    j = minc + T
    while j < wc:
        q = min(T, wc - j)
        need = not np.array_equal(C[:, j:j + q], C[:, minc:minc + q])
        blocks.append((j, q, need))
        j += T

    full_needed = [b for b in blocks if b[1] == T and b[2]]
    if full_needed:
        ops.append(28)                            # CopyI: the whole unit strip
        sels.append(rect(0, hc - 1, minc, minc + T - 1))
        for (dj, q, need) in blocks:
            if q == T and need:
                r = rect(0, hc - 1, dj, dj + q - 1)
                ops.append(30)                    # Paste at the block's origin
                sels.append([r[0], r[1], 0, 0])

    # trailing partial block: copy only the leading part of the unit strip
    tail = [b for b in blocks if b[1] < T and b[2]]
    for (dj, q, _need) in tail:
        ops.append(28)
        sels.append(rect(0, hc - 1, minc, minc + q - 1))
        r = rect(0, hc - 1, dj, dj + q - 1)
        ops.append(30)
        sels.append([r[0], r[1], 0, 0])

    # 4) Paste is transparent for colour 0, so any cell of the unit that is 0
    #    must be painted explicitly wherever it still holds something else.
    zero_cells = []
    for jj in range(minc + T, wc):
        src = minc + (jj - minc) % T
        for i in range(hc):
            if int(C[i, src]) == 0 and int(C[i, jj]) != 0:
                zero_cells.append(m(i, jj))
    if zero_cells:
        ops.append(0)
        sels.append(sel_of(sorted(set(zero_cells))))

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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
