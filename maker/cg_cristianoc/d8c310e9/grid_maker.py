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

ROTS = [0, 1, 2, 3]  # identity, rot90, rot180, rot270 (frame the generator picks)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "instance_plan": plan}


def _min_period_tiles(gi, go, bgc):
    # True when continuing the visible block with its own smallest period
    # reproduces go (the generator's addon can leave a single period visible,
    # which makes the period ambiguous; such draws are resampled).
    G = np.array(gi)
    cs = np.argwhere(G != bgc)
    if len(cs) == 0:
        return False
    r0, c0, c1 = int(cs[:, 0].min()), int(cs[:, 1].min()), int(cs[:, 1].max())
    band = G[r0:]
    wd = c1 - c0 + 1
    per = wd
    for q in range(1, wd):
        if all(np.array_equal(band[:, c], band[:, c - q]) for c in range(c0 + q, c1 + 1)):
            per = q
            break
    R = G.copy()
    for c in range(c1 + 1, G.shape[1]):
        R[r0:, c] = R[r0:, c - per]
    return np.array_equal(R, np.array(go))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if rot is None:
        rot = random.choice(ROTS)
    side = min(max_h, max_w) if rot in (1, 3) else None
    hmax = side if side else max_h
    wmax = side if side else max_w
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (3, max(3, hmax)))
        w = unifint(diff_lb, diff_ub, (10, max(10, wmax)))
        p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numc)
        obj = set()
        for j in range(p):
            numcells = unifint(diff_lb, diff_ub, (1, h - 1))
            for ii in range(h - 1, h - numcells - 1, -1):
                obj.add((choice(ccols), (ii, j)))
        gi = canvas(bgc, (h, w))
        minobj = obj | shift(obj, (0, p))
        addonw = randint(0, p)
        addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
        fullobj = minobj | addon
        leftshift = randint(0, addonw)
        fullobj = shift(fullobj, (0, -leftshift))
        gi = paint(gi, fullobj)
        go = tuple(e for e in gi)
        for j in range(w // (2 * p) + 2):
            go = paint(go, shift(fullobj, (0, j * 2 * p)))
        if _min_period_tiles(gi, go, bgc):
            break
    fn = (identity, rot90, rot180, rot270)[rot]
    return {"input": fn(gi), "output": fn(go)}


def derive_operations(I, O, examples=None):
    """Read one period of the striped block and lay it down again and again,
    one period further along each time, until the grid edge.

    Everything (frame, background, period, tile, stamp positions) is measured
    from I alone; O is never consulted.
    """
    A = np.asarray(I, dtype=int)
    H, W = A.shape

    # 1) frame: the quarter turn that stands the block in the bottom-left corner
    #    (top row and right column pure background).  Work in that frame and map
    #    every rectangle back to I's own coordinates.
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [[0, 0, H - 1, W - 1]]
    C = np.rot90(A, kk)
    idx = np.rot90(np.arange(H * W).reshape(H, W), kk)
    Hc, Wc = C.shape
    bgc = int(C[0, -1])
    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [[0, 0, H - 1, W - 1]]
    r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
    c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())
    band = C[r0:r1 + 1]

    # 2) period: the smallest column shift that maps the block onto itself
    width = c1 - c0 + 1
    per = width
    for cand in range(1, width):
        if all(np.array_equal(band[:, c], band[:, c - cand]) for c in range(c0 + cand, c1 + 1)):
            per = cand
            break

    def to_A(rr0, cc0, rr1, cc1):
        pts = [divmod(int(idx[a, b]), W) for a in (rr0, rr1) for b in (cc0, cc1)]
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        return min(ys), min(xs), max(ys), max(xs)

    # the motif: one period, the first `per` columns of the block
    t0, t1 = c0, c0 + per - 1

    grid = A.copy()
    ops, sels = [], []
    clip = None          # (A-rect of copied source)
    zero_cells = []      # cells a stamp landed on where the motif is colour 0
    k = 1
    while True:
        d0 = t0 + k * per
        if d0 >= Wc:
            break
        k += 1
        d1 = min(Wc - 1, d0 + per - 1)  # far-edge part is clipped
        if d1 <= c1:
            continue  # whole stamp already present in the block
        # destination and matching source in I's coordinates
        dy0, dx0, dy1, dx1 = to_A(r0, d0, r1, d1)
        sy0, sx0, sy1, sx1 = to_A(r0, t0, r1, t0 + (d1 - d0))
        full = to_A(r0, t0, r1, t1)
        # the motif in I's frame is `full`; when the stamp is cut short on the
        # side of I's origin, only the matching part of the motif can be copied
        oy, ox = dy0 - sy0, dx0 - sx0
        if (full[0] + oy >= 0) and (full[1] + ox >= 0):
            src = full
        else:
            src = (sy0, sx0, sy1, sx1)
        piece = A[src[0]:src[2] + 1, src[1]:src[3] + 1]
        py, px = src[0] + oy, src[1] + ox
        after = grid.copy()
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                y, x = py + i, px + j
                if 0 <= y < H and 0 <= x < W:
                    if piece[i, j] != 0:
                        after[y, x] = piece[i, j]
                    elif bgc != 0:
                        zero_cells.append((y, x))
        if not np.array_equal(after, grid):
            if clip != src:
                # full rectangle: one period of the block, background included
                ops.append(28); sels.append([src[0], src[1], src[2] - src[0], src[3] - src[1]])
                clip = src
            ops.append(30); sels.append([py, px, 0, 0])
            grid = after

    # the clipboard drops colour 0, so where the motif is black the stamps left
    # background standing: blacken those landed cells in one go
    if bgc != 0:
        zs = sorted({c for c in zero_cells if grid[c] != 0})
        if zs:
            ops.append(0); sels.append(sel_of(zs))

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
