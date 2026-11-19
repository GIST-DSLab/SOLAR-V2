"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, objc=None) -> dict:
    from utils import unifint
    cols = list(range(1, 10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    remcols = [c for c in cols if c not in (bgc, linc)]
    if objc is None or objc in (bgc, linc):
        objc = random.choice(remcols)
    nc = unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start}
    bx.discard(start)
    for _ in range(nc - 1):
        cand = sorted({(r + dr, c + dc) for r, c in obj
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)} & bx)
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)
    canv = np.full((h, w), bgc, dtype=int)
    gi = canv.copy()
    for r, c in obj:
        gi[r, c] = objc
    line_v = np.full((h, 1), linc, dtype=int)
    gi1 = np.hstack([gi, line_v, canv])
    gi2 = np.hstack([canv, line_v, canv])
    gin = np.vstack([gi1, np.full((1, 2 * w + 1), linc, dtype=int), gi2])
    go = gi.copy()
    go = np.hstack([go, np.fliplr(go)])
    go = np.vstack([go, np.flipud(go)])
    go[go == objc] = linc
    scf = random.choice([lambda g: g, np.flipud, np.fliplr,
                         lambda g: np.flipud(np.fliplr(g))])
    gin = scf(gin)
    go = scf(go)
    return {"input": gin.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # the divider row/column cross exactly in the middle -> four h x w panels
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, w])

    bgc, src = None, None
    for (r0, c0) in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()           # an empty panel shows the background colour
        else:
            src = (r0, c0)             # the one panel holding the object
    r0, c0 = src
    objc = [v for v in set(I[r0:r0 + h, c0:c0 + w].flatten().tolist()) if v != bgc][0]

    ops, sels = [], []
    # tiny shadow of the working grid, only used to skip ops with no visible effect
    g = I.copy()

    def emit(op, sel):
        ops.append(op); sels.append(sel)

    # 1. every object cell takes the divider colour (one unit: all cells of objc)
    obj_cells = [(r, c) for r in range(H) for c in range(W) if I[r, c] == objc]
    emit(linc, sel_of(obj_cells))
    for r, c in obj_cells:
        g[r, c] = linc

    # 2. keep only the object panel (crop, unless it already sits at the origin),
    #    then grow the canvas to the 2h x 2w mirror mosaic
    if (r0, c0) != (0, 0):
        emit(33, [r0, c0, h - 1, w - 1])          # crop to the panel (full rectangle)
        g = g[r0:r0 + h, c0:c0 + w].copy()
    emit(33, [0, 0, 2 * h - 1, 2 * w - 1])        # grow to 2h x 2w
    ng = np.zeros((2 * h, 2 * w), dtype=int)
    hh, ww = min(2 * h, g.shape[0]), min(2 * w, g.shape[1])
    ng[:hh, :ww] = g[:hh, :ww]
    g = ng
    panel = g[:h, :w].copy()

    # 3. horizontal doubling: copy the panel beside itself, mirror the side that is
    #    NOT the panel's own side in I (full-rectangle flips)
    if not np.array_equal(g[:h, w:], panel):
        emit(29, [0, 0, h - 1, w - 1])
        emit(30, [0, w, 0, 0])
        g[:h, w:] = panel
    if not np.array_equal(panel, np.fliplr(panel)):
        fc = 0 if c0 > 0 else w                   # panel was right half -> flip the left copy
        emit(26, [0, fc, h - 1, w - 1])
        g[:h, fc:fc + w] = np.fliplr(g[:h, fc:fc + w])

    # 4. vertical doubling: copy the whole strip below, mirror one of the two strips
    strip = g[:h].copy()
    if not np.array_equal(g[h:], strip):
        emit(29, [0, 0, h - 1, 2 * w - 1])
        emit(30, [h, 0, 0, 0])
        g[h:] = strip
    if not np.array_equal(strip, np.flipud(strip)):
        fr = 0 if r0 > 0 else h                   # panel was bottom half -> flip the top strip
        emit(27, [fr, 0, h - 1, 2 * w - 1])

    emit(34, [0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
