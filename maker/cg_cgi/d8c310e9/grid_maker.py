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

ROTS = [0, 1, 2, 3]  # np.rot90 k applied to the canonical grid


def sample_colors(num_examples=None) -> dict:
    bgc = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": k} for k in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": k} for k in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + d * (b - a)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None, **kw) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # canonical grid is h x w; odd rotations swap the shape
    mh, mw = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unif(diff_lb, diff_ub, (3, min(30, mh)))
    w = _unif(diff_lb, diff_ub, (10, min(30, mw)))
    p = _unif(diff_lb, diff_ub, (2, (w - 1) // 3))
    remcols = [c for c in range(10) if c != bgc]
    numc = _unif(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numc)
    obj = {}
    for j in range(p):
        numcells = _unif(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj[(ii, j)] = random.choice(ccols)
    full = dict(obj)
    for (i, j), v in obj.items():
        full[(i, j + p)] = v
    addonw = random.randint(0, p)
    for (i, j), v in obj.items():
        if j < addonw:
            full[(i, j + 2 * p)] = v
    ls = random.randint(0, addonw)
    full = {(i, j - ls): v for (i, j), v in full.items()}
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), v in full.items():
        if 0 <= j < w:
            gi[i, j] = v
    go = gi.copy()
    for k in range(w // (2 * p) + 2):
        for (i, j), v in full.items():
            jj = j + k * 2 * p
            if 0 <= jj < w:
                go[i, jj] = v
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """Golf reading: ONE period of the striped block, taken as a full-length strip across
    the grid (whole cross extent), is the unit; it repeats along the block's axis, one
    period at a time, until it runs off the far edge. O is never consulted."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # frame: the quarter turn that stands the block in the bottom-left corner
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    Hf, Wf = C.shape
    bgc = int(C[0, -1])
    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [full]

    # period: shortest slide along the axis that drops the block onto itself
    colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
    minc, maxc = int(cells[:, 1].min()), int(cells[:, 1].max())
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        if all(colors.get((r, c - cand)) == v for (r, c), v in colors.items() if c - cand >= minc):
            per = cand
            break

    # frame -> original coordinates
    idx = np.rot90(np.arange(H * W).reshape(H, W), kk)

    def to_rect(c_lo, c_hi):  # frame columns [c_lo, c_hi], all frame rows
        vals = idx[:, c_lo:c_hi + 1].ravel()
        rs, cs = vals // W, vals % W
        return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())

    ops, sels = [], []
    grid = A.copy()
    clip_rect = None
    k = 1
    while minc + k * per < Wf:
        d0 = minc + k * per
        width = min(per, Wf - d0)  # last stamp: only the part of the strip that still fits
        sr0, sc0, sr1, sc1 = to_rect(minc, minc + width - 1)
        dr0, dc0, dr1, dc1 = to_rect(d0, d0 + width - 1)
        src = A[sr0:sr1 + 1, sc0:sc1 + 1]
        after = grid.copy()
        reg = after[dr0:dr1 + 1, dc0:dc1 + 1]
        after[dr0:dr1 + 1, dc0:dc1 + 1] = np.where(src != 0, src, reg)
        if not np.array_equal(after, grid):
            rect = [sr0, sc0, sr1 - sr0, sc1 - sc0]
            if clip_rect != rect:
                # full rectangle: one period strip spanning the whole cross extent
                ops.append(28); sels.append(rect)
                clip_rect = rect
            ops.append(30); sels.append([dr0, dc0, 0, 0])
            grid = after
        # clipboard skips colour 0: black the pasted strip's cells whose source is 0
        if bgc != 0:
            black = [(dr0 + i, dc0 + j) for i in range(src.shape[0]) for j in range(src.shape[1])
                     if src[i, j] == 0 and grid[dr0 + i, dc0 + j] != 0]
            if black:
                ops.append(0); sels.append(sel_of(black))
                for r, c in black:
                    grid[r, c] = 0
        k += 1
    ops.append(34); sels.append(full)
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
