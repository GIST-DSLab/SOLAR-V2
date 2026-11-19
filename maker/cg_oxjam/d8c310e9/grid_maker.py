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

ORIENTS = [0, 1, 2, 3]  # identity, rot90, rot180, rot270 applied after generation


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = list(ORIENTS) + [random.choice(ORIENTS) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ORIENTS, n_ex)
    plan = [{"orient": o} for o in ex]
    plan.append({"orient": random.choice(ex)})
    return {"bgc": bgc, "instance_plan": plan}


def _period_ok(gi, go, bgc):
    A = np.array(gi)
    nz = np.argwhere(A != bgc)
    if len(nz) == 0:
        return False
    minc, maxc = int(nz[:, 1].min()), int(nz[:, 1].max())
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        if all(np.array_equal(A[:, c], A[:, c - cand]) for c in range(minc + cand, maxc + 1)):
            per = cand
            break
    B = A.copy()
    for c in range(minc + per, A.shape[1]):
        B[:, c] = A[:, minc + (c - minc) % per]
    return np.array_equal(B, np.array(go))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, orient=None, **kw) -> dict:
    from dsl import (interval, remove, canvas, shift, paint,
                     identity, rot90, rot180, rot270)
    from utils import unifint, randint, choice, sample
    if orient is None:
        orient = random.choice(ORIENTS)
    for _attempt in range(100):
        cols = interval(0, 10, 1)
        # a quarter turn swaps the dimensions, so bound them accordingly
        hb, wb = (max_w, max_h) if orient in (1, 3) else (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (3, hb))
        w = unifint(diff_lb, diff_ub, (10, wb))
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
        addon = {cij for cij in obj if cij[1][1] < addonw}
        fullobj = minobj | addon
        leftshift = randint(0, addonw)
        fullobj = shift(fullobj, (0, -leftshift))
        gi = paint(gi, fullobj)
        go = tuple(e for e in gi)
        for j in range(w // (2 * p) + 2):
            go = paint(go, shift(fullobj, (0, j * 2 * p)))
        # reject instances whose visible block does not pin down the period
        # (e.g. only one period left on screen after the left shift)
        if _period_ok(gi, go, bgc):
            break
    fn = (identity, rot90, rot180, rot270)[orient]
    return {"input": fn(gi), "output": fn(go)}


def derive_operations(I, O=None, examples=None):
    """Read the striped block as ONE PERIOD repeated: take the single period-wide
    slab standing against the grid edge and lay it down again every period until
    the far edge is reached (the unit being stamped is one period, not the block)."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # frame: the quarter turn that leaves the top row and right column pure background
    axis = {0: (0, 1), 3: (-1, 0), 2: (0, -1), 1: (1, 0)}
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    bgc = int(C[0, -1])
    dr, dc = axis[kk]
    nz = np.argwhere(C != bgc)
    if len(nz) == 0:
        return [34], [full]
    minc, maxc = int(nz[:, 1].min()), int(nz[:, 1].max())

    # period: shortest column shift under which the block's columns repeat
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        if all(np.array_equal(C[:, c], C[:, c - cand]) for c in range(minc + cand, maxc + 1)):
            per = cand
            break

    # the period slab (canonical columns minc..minc+per-1, full height) in original coords
    idx = np.rot90(np.arange(H * W).reshape(H, W), kk)
    slab = idx[:, minc:minc + per].ravel()
    rr, cc = slab // W, slab % W
    t_r0, t_r1, t_c0, t_c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())

    grid = A.copy()
    ops, sels = [], []
    clip_rect = None
    k = 1
    while True:
        offr, offc = dr * per * k, dc * per * k
        d_r0, d_r1, d_c0, d_c1 = t_r0 + offr, t_r1 + offr, t_c0 + offc, t_c1 + offc
        if d_r1 < 0 or d_r0 >= H or d_c1 < 0 or d_c0 >= W:
            break
        # a slab running past the near edge carries only the part that fits;
        # one running past the far edge is clipped by the paste itself
        p_r0, p_c0 = max(0, d_r0), max(0, d_c0)
        s_r0, s_c0 = p_r0 - offr, p_c0 - offc
        src = A[s_r0:t_r1 + 1, s_c0:t_c1 + 1]
        sh, sw = min(src.shape[0], H - p_r0), min(src.shape[1], W - p_c0)
        cut = src[:sh, :sw]
        after = grid.copy()
        reg = after[p_r0:p_r0 + sh, p_c0:p_c0 + sw]
        after[p_r0:p_r0 + sh, p_c0:p_c0 + sw] = np.where(cut != 0, cut, reg)
        if not np.array_equal(after, grid):
            rect = [s_r0, s_c0, t_r1 - s_r0, t_c1 - s_c0]  # full rectangle: one period slab
            if clip_rect != rect:
                ops.append(28); sels.append(rect)
                clip_rect = rect
            ops.append(30); sels.append([p_r0, p_c0, 0, 0])
            grid = after
        # colour 0 is empty to the clipboard: black the stamp's 0-cells in one go
        black = [(p_r0 + i, p_c0 + j) for i in range(sh) for j in range(sw)
                 if cut[i, j] == 0 and grid[p_r0 + i, p_c0 + j] != 0]
        if black:
            ops.append(0); sels.append(sel_of(black))
            for (a, b) in black:
                grid[a, b] = 0
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
