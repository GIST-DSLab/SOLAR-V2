"""
ARC Task: 3631a71a (RE-ARC) — LLM-generated grid_maker
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

# ── LLM-generated: x
import random
import numpy as np
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    patchcol = random.choice(cols)
    bgc = random.choice([c for c in cols if c != patchcol])
    return {"bgc": bgc, "patchcol": patchcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, patchcol=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        return int(round(random.uniform(a + (b - a) * lb, a + (b - a) * ub)))

    cols = list(range(10))
    if patchcol is None:
        patchcol = random.choice(cols)
    if bgc is None:
        bgc = random.choice([c for c in cols if c != patchcol])
    cutoff = 2
    hmax = max(6, min(15, (min(max_h, max_w) + cutoff) // 2))
    h = unifint(diff_lb, diff_ub, (6, hmax))
    w = h
    remcols = [c for c in cols if c not in (bgc, patchcol)]
    c = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w) if i >= j]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = random.sample(inds, ncells)
    for (i, j) in cells:
        col = random.choice(ccols)
        c[i][j] = col
        c[j][i] = col
    c = [row + row[::-1] for row in c]
    c = c + c[::-1]
    go = [row[:-cutoff] for row in c[:-cutoff]]
    gi = [list(r) for r in go]
    fullh, fullw = len(gi), len(gi[0])

    def recoverable(g):
        # every orbit (under the crop-offset mirror group) keeps a non-patch cell
        R = fullh + 2
        seen = set()
        for r in range(fullh):
            for q in range(fullw):
                if (r, q) in seen:
                    continue
                s = {(r, q)}
                st = [(r, q)]
                while st:
                    a, b = st.pop()
                    for x, y in ((R - 1 - a, b), (a, R - 1 - b), (b, a)):
                        if (x, y) not in s:
                            s.add((x, y))
                            st.append((x, y))
                inside = [(a, b) for (a, b) in s if a < fullh and b < fullw]
                seen.update(inside)
                if all(g[a][b] == patchcol for (a, b) in inside):
                    return False
        return True

    ndist = unifint(diff_lb, diff_ub, (1, int((2 * h * 2 * w) ** 0.5)))
    succ, tr, maxtr = 0, 0, 10 * ndist
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = random.randint(2, h // 2 + 1)
        ow = random.randint(2, w // 2 + 1)
        loci = random.randint(0, fullh - oh)
        locj = random.randint(0, fullw - ow)
        gi2 = [list(r) for r in gi]
        for a in range(loci, loci + oh):
            for b in range(locj, locj + ow):
                gi2[a][b] = patchcol
        if any(gi2[a][b] == patchcol for a in range(cutoff) for b in range(cutoff)):
            continue
        if not recoverable(gi2):
            continue
        succ += 1
        gi = gi2
    if succ == 0:
        raise ValueError("no occlusion placed")
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading: the canvas is a mirror-symmetric picture (transpose, plus a row
    mirror and a column mirror about the centre of the block that excludes
    the 2-wide top/left frame).  One colour paints rectangular blotches over
    it.  Repair = wipe the blotches, then lay the picture's own reflections
    over itself, so every wiped cell is filled by its mirror partner.  Each
    reflection is one layered unit:
        CopyO(block)  -> Flip/Transpose(block)  -> Paste the copy back.
    The flipped picture shows through only where the copy has holes (0),
    i.e. exactly at the wiped cells; elsewhere the original sits on top and,
    the picture being symmetric, agrees with its reflection anyway.
    Reflections are applied in turn until none of them changes the grid.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- symmetry orbits (crop of the doubled canvas, offset +1) -------------
    R, Cc = hi + 2, wi + 2
    owner = -np.ones((hi, wi), dtype=int)
    orbits = []
    for r in range(hi):
        for c in range(wi):
            if owner[r, c] >= 0:
                continue
            s = {(r, c)}
            st = [(r, c)]
            while st:
                a, b = st.pop()
                cands = [(R - 1 - a, b), (a, Cc - 1 - b)]
                if R == Cc:
                    cands.append((b, a))
                for x, y in cands:
                    if 0 <= x < R and 0 <= y < Cc and (x, y) not in s:
                        s.add((x, y))
                        st.append((x, y))
            cells = [(a, b) for (a, b) in s if a < hi and b < wi]
            for a, b in cells:
                owner[a, b] = len(orbits)
            orbits.append(cells)

    def consistent(p):
        changed = 0
        for cells in orbits:
            vals = {int(I[a, b]) for a, b in cells if int(I[a, b]) != p}
            npatch = sum(1 for a, b in cells if int(I[a, b]) == p)
            if len(vals) > 1 or (not vals and npatch):
                return -1
            changed += npatch
        return changed

    # --- occluding colour: read from the demonstrations when available -------
    patchcol = None
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            s = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            cand = s if cand is None else (cand & s)
        if cand:
            cand = [p for p in sorted(cand) if consistent(p) >= 0]
            if len(cand) == 1:
                patchcol = cand[0]
    if patchcol is None:
        best, score = None, 0
        for p in sorted(set(I.flatten().tolist())):
            sc = consistent(p)
            if sc > score:
                best, score = p, sc
        patchcol = best
    if patchcol is None or not (I == patchcol).any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    g = I.copy()

    # --- 1. wipe the blotches: every cell of the occluding colour -> 0 --------
    if patchcol != 0:
        ops.append(0); sels.append(sel_of(zip(*np.nonzero(I == patchcol))))
        g[I == patchcol] = 0

    # --- 2. layer reflections until nothing changes ---------------------------
    reflections = []
    if wi > 2:   # column mirror c -> W+1-c : flip the block right of the frame
        blk = [0, 2, hi - 1, wi - 3]
        reflections.append(("col", blk, [26],
                            lambda x: np.concatenate([x[:, :2], x[:, 2:][:, ::-1]], 1)))
    if hi > 2:   # row mirror r -> H+1-r : flip the block below the frame
        blk = [2, 0, hi - 3, wi - 1]
        reflections.append(("row", blk, [27],
                            lambda x: np.concatenate([x[:2], x[2:][::-1]], 0)))
    if hi == wi:  # transpose of the whole square: rot90 CCW then flip up/down
        blk = [0, 0, hi - 1, wi - 1]
        reflections.append(("tr", blk, [24, 27], lambda x: x.T.copy()))

    def blk_slice(b):
        r, c, h, w = b
        return slice(r, r + h + 1), slice(c, c + w + 1)

    seen = {g.tobytes()}
    idle = 0
    k = 0
    while idle < len(reflections):
        name, blk, flipops, fn = reflections[k % len(reflections)]
        k += 1
        mirrored = fn(g)
        layered = np.where(g != 0, g, mirrored)
        if np.array_equal(layered, g):
            idle += 1
            continue
        idle = 0
        rs, cs = blk_slice(blk)
        ops.append(29); sels.append(blk)                 # CopyO the block as it is
        if name == "tr":
            mid = np.rot90(g, 1)
            ops.append(24); sels.append(blk)             # rotate whole square CCW
            ops.append(27); sels.append(blk)             # flip up/down -> transpose
        else:
            ops.append(flipops[0]); sels.append(blk)     # reflect the block in place
        if not np.array_equal(layered, mirrored):
            ops.append(30); sels.append([blk[0], blk[1], 0, 0])  # lay the copy back on top
            g = layered
        else:
            g = mirrored
        seen.add(g.tobytes())

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3631a71a"
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
                                f"for task 3631a71a"
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
                    f"Failed to build a complete episode for task 3631a71a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3631a71a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
