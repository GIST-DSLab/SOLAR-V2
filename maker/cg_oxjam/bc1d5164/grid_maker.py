"""
ARC Task: bc1d5164 (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    bgc, objc = random.sample(range(10), 2)
    return {"bgc": bgc, "objc": objc}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a: b = a
        return random.randint(a + int(round((b - a) * lb)), a + int(round((b - a) * ub)))
    if bgc is None or objc is None:
        bgc, objc = random.sample(range(10), 2)
    hmax = max(3, min(15, (max_h + 1) // 2))
    wmax = max(2, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    fullh, fullw = 2 * h - 1, 2 * w + 1
    inds = [(i, j) for i in range(h) for j in range(w)]
    n = (h - 1) * (w - 1) - 1
    def pick(pred):
        pool = [ij for ij in inds if pred(ij)]
        return random.sample(pool, random.randint(1, max(1, n)))
    A = pick(lambda ij: ij[0] < h - 1 and ij[1] < w - 1)
    B = pick(lambda ij: ij[0] < h - 1 and ij[1] > 0)
    C = pick(lambda ij: ij[0] > 0 and ij[1] < w - 1)
    D = pick(lambda ij: ij[0] > 0 and ij[1] > 0)
    gi = [[bgc] * fullw for _ in range(fullh)]
    go = [[bgc] * w for _ in range(h)]
    for S, (dr, dc) in ((A, (0, 0)), (B, (0, fullw - w)), (C, (fullh - h, 0)), (D, (fullh - h, fullw - w))):
        for i, j in S:
            gi[i + dr][j + dc] = objc
            go[i][j] = objc
    return {"input": gi, "output": go}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    h = (fh + 1) // 2          # quadrant height; bottom band starts at row h-1
    w = (fw - 1) // 2          # quadrant width; right block starts at col w+1
    bgc = int(I[h - 1, 0])     # the row between the bands is pure background
    fg = [int(c) for c in np.unique(I) if c != bgc]
    objc = fg[0] if fg else bgc

    # Reading: fold the sheet twice. Fold 1 lays the bottom band (rows h-1..2h-2)
    # over the top band; fold 2 lays the right block (cols w+1..2w) of the
    # folded band over the left block; the left block is the answer.
    obj = I == objc
    top = obj[0:h, :]
    bot = obj[h - 1:2 * h - 1, :]
    L = slice(0, w); R = slice(w + 1, 2 * w + 1)
    band = top | bot
    need1 = bool(((band[:, L] | band[:, R]) & ~(top[:, L] | top[:, R])).any())
    cur = band if need1 else top
    need2 = bool((cur[:, R] & ~cur[:, L]).any())
    ops, sels = [], []

    union = top[:, L] | top[:, R] | bot[:, L] | bot[:, R]
    # with a non-zero background, an opaque paste would overwrite whole blocks;
    # if that alone already lands on the union, the transparency step would be idle
    if need1 and need2:
        opaque = bot[:, R]
    elif need1:
        opaque = bot[:, L]
    elif need2:
        opaque = top[:, R]
    else:
        opaque = None
    paint_route = objc == 0 or (bgc != 0 and opaque is not None and np.array_equal(opaque, union))

    if paint_route:
        # object colour 0 is invisible to the clipboard: paint the fold landings
        left = top[:, L].copy()
        if need1:
            cells = [(r, c) for r in range(h) for c in range(w) if bot[r, c] and not left[r, c]]
            if cells:
                ops.append(objc); sels.append(sel_of(cells))  # bottom-left lands on top-left
                left |= bot[:, L]
            src = band[:, R]
        else:
            src = top[:, R]
        if need2:
            cells = [(r, c) for r in range(h) for c in range(w) if src[r, c] and not left[r, c]]
            ops.append(objc); sels.append(sel_of(cells))      # folded right block lands on left
        ops.append(33); sels.append([0, 0, h - 1, w - 1])     # keep the left block
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    if bgc != 0:
        # make the background transparent so pasting overlays only the object
        ops.append(0); sels.append(sel_of([tuple(map(int, x)) for x in np.argwhere(I == bgc)]))
    if need1:
        ops.append(29 if ops else 28); sels.append([h - 1, 0, h - 1, fw - 1])  # whole bottom band
        ops.append(30); sels.append([0, 0, 0, 0])                              # onto the top band
    if need2:
        ops.append(29); sels.append([0, w + 1, h - 1, w - 1])  # right block of the folded band
        ops.append(30); sels.append([0, 0, 0, 0])              # onto the left block
    ops.append(33); sels.append([0, 0, h - 1, w - 1])          # keep the left block
    if bgc != 0:
        left = cur[:, L] | cur[:, R] if need2 else cur[:, L]
        # restore background on the canvas cells that are 0 now
        holes = [(r, c) for r in range(h) for c in range(w) if not left[r, c]]
        if holes:
            ops.append(bgc); sels.append(sel_of(holes))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
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
                                f"for task bc1d5164"
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
                    f"Failed to build a complete episode for task bc1d5164 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bc1d5164-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
