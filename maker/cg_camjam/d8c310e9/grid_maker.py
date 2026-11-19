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

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical grid


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def _period(C, bgc):
    """Shortest slide along the row axis that maps the (canonical) block onto itself."""
    cells = np.argwhere(C != bgc)
    colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
    rmin, rmax = int(cells[:, 0].min()), int(cells[:, 0].max())
    minc, maxc = int(cells[:, 1].min()), int(cells[:, 1].max())
    for cand in range(1, maxc - minc + 1):
        if all(colors.get((r, c), bgc) == colors.get((r, c - cand), bgc)
               for r in range(rmin, rmax + 1) for c in range(minc + cand, maxc + 1)):
            return cand
    return maxc - minc + 1


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, rot):
    def unif(lo, hi):
        if hi <= lo:
            return lo
        lb = lo + (hi - lo) * diff_lb
        ub = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, random.randint(int(round(lb)), max(int(round(lb)), int(round(ub))))))
    if rot is None:
        rot = random.choice(ROTS)
    if rot % 2 == 1:
        hmax, wmax = max_w, max_h
    else:
        hmax, wmax = max_h, max_w
    hmax = max(3, min(30, hmax))
    wmax = max(10, min(30, wmax))
    h = unif(3, hmax)
    w = unif(10, wmax)
    p = unif(2, (w - 1) // 3)
    remcols = [c for c in range(10) if c != bgc]
    numc = unif(1, 9)
    ccols = random.sample(remcols, numc)
    obj = {}
    for j in range(p):
        numcells = unif(1, h - 1)
        for ii in range(h - 1, h - numcells - 1, -1):
            obj[(ii, j)] = random.choice(ccols)
    fullobj = dict(obj)
    for (i, j), v in obj.items():
        fullobj[(i, j + p)] = v
    addonw = random.randint(0, p)
    for (i, j), v in obj.items():
        if j < addonw:
            fullobj[(i, j)] = v
    leftshift = random.randint(0, addonw)
    fullobj = {(i, j - leftshift): v for (i, j), v in fullobj.items()}
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), v in fullobj.items():
        if 0 <= j < w:
            gi[i, j] = v
    go = gi.copy()
    for k in range(w // (2 * p) + 2):
        for (i, j), v in fullobj.items():
            jj = j + k * 2 * p
            if 0 <= jj < w:
                go[i, jj] = v
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go], "_p": p}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, rot=None) -> dict:
    # reject instances whose marching direction is ambiguous (more than one quarter turn
    # puts a pure-background row on top and a pure-background column on the right), and
    # instances whose visible block does not reveal its true period
    d = None
    for _ in range(200):
        d = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, rot)
        A = np.array(d["input"])
        ks = [k for k in range(4)
              if len(set(np.rot90(A, k)[0].tolist())) == 1
              and len(set(np.rot90(A, k)[:, -1].tolist())) == 1]
        if len(ks) == 1:
            C = np.rot90(A, ks[0])
            if _period(C, bgc) == d["_p"]:
                d.pop("_p")
                return d
    d.pop("_p", None)
    return d


def derive_operations(I, O=None, examples=None):
    """The block is a stack of columns that repeats every `per` cells along one axis.
    Take ONE period of it (a per-long slice at the block's near end) as the stamp and
    lay copies of that slice end to end, outward from the block, until the grid ends."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    # Frame: the quarter turn that stands the block in the bottom-left corner
    # (top row and right column pure background) tells the marching direction.
    axis = {0: (0, 1), 3: (-1, 0), 2: (0, -1), 1: (1, 0)}
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [[0, 0, H - 1, W - 1]]
    C = np.rot90(A, kk)
    bgc = int(C[0, -1])
    dr, dc = axis[kk]
    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [[0, 0, H - 1, W - 1]]

    # Period: shortest slide along the march axis that maps the block onto itself.
    per = _period(C, bgc)

    # Block bbox in the given orientation; the tile is its per-long slice at the near end.
    oc = np.argwhere(A != bgc)
    r0, r1 = int(oc[:, 0].min()), int(oc[:, 0].max())
    c0, c1 = int(oc[:, 1].min()), int(oc[:, 1].max())
    if dc == 1:
        tr0, tr1, tc0, tc1 = r0, r1, c0, c0 + per - 1
    elif dc == -1:
        tr0, tr1, tc0, tc1 = r0, r1, c1 - per + 1, c1
    elif dr == 1:
        tr0, tr1, tc0, tc1 = r0, r0 + per - 1, c0, c1
    else:
        tr0, tr1, tc0, tc1 = r1 - per + 1, r1, c0, c1

    grid = A.copy()
    clip = None       # (rect, array) currently in the clipboard
    ops, sels = [], []

    def paste(g, arr, a, b):
        g = g.copy()
        ch, cw = arr.shape
        a1, b1 = min(H, a + ch), min(W, b + cw)
        sub = arr[:a1 - a, :b1 - b]
        reg = g[a:a1, b:b1]
        g[a:a1, b:b1] = np.where(sub != 0, sub, reg)
        return g

    k = 1
    while True:
        orr, occ = dr * per * k, dc * per * k
        a0, a1, b0, b1 = tr0 + orr, tr1 + orr, tc0 + occ, tc1 + occ
        if a1 < 0 or a0 >= H or b1 < 0 or b0 >= W:
            break
        # a stamp running off the near (top/left) edge must start at the edge,
        # so it carries only the part of the tile that fits; the far edge just clips
        pa, pb = max(0, a0), max(0, b0)
        sr0, sc0 = pa - orr, pb - occ
        rect = (sr0, sc0, tr1 - sr0, tc1 - sc0)
        src = A[sr0:tr1 + 1, sc0:tc1 + 1]
        after = paste(grid, src, pa, pb)
        if not np.array_equal(after, grid):
            need_copy = clip is None or (clip[0] != rect and
                                         not np.array_equal(paste(grid, clip[1], pa, pb), after))
            if need_copy:
                ops.append(28); sels.append(list(rect))   # full rectangle: one period of the block
                clip = (rect, src)
            ops.append(30); sels.append([pa, pb, 0, 0])
            grid = paste(grid, clip[1], pa, pb)
        # clipboard treats 0 as empty: where the tile is black, the stamp left bg standing
        if bgc != 0:
            sh, sw = min(src.shape[0], H - pa), min(src.shape[1], W - pb)
            black = [(pa + i, pb + j) for i in range(sh) for j in range(sw)
                     if src[i, j] == 0 and grid[pa + i, pb + j] != 0]
            if black:
                ops.append(0); sels.append(sel_of(black))
                for (r, c) in black:
                    grid[r, c] = 0
        k += 1

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
