"""
ARC Task: 28e73c20 (RE-ARC) — LLM-generated grid_maker
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
    # Colour 3 (the spiral) is hard-coded. The input's noise palette is the only
    # random colour choice. The rule ignores which noise colours are used, but we
    # fix the palette here so every instance in an episode looks alike.
    cols = [c for c in range(10) if c != 3]
    ncols = random.randint(1, 9)
    ccols = random.sample(cols, ncols)
    return {"ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if ccols is None:
        cols = [c for c in range(10) if c != 3]
        ccols = random.sample(cols, random.randint(1, 9))
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    # The internal canvas is one column wider. Column 0 is cropped away at the end.
    w = unifint(diff_lb, diff_ub, (4, max(4, min(30, max_w + 1))))
    gi = np.array([[random.choice(ccols) for _ in range(w)] for _ in range(h)], dtype=int)
    go = gi.copy()
    go[0, :] = 3
    sp = (0, w - 1)
    direc = (1, 0)
    lw, lh, ld = w, h, h
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        ep = (sp[0] + direc[0] * (ld - 1), sp[1] + direc[1] * (ld - 1))
        for k in range(ld):
            go[sp[0] + direc[0] * k, sp[1] + direc[1] * k] = 3
        direc = direcmapper[direc]
        ld = lh if isverti else lw
        isverti = not isverti
        sp = ep
    gi = gi[:, 1:]
    go = go[:, 1:]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """The spiral depends only on I's SHAPE.

    The golf entry reads the task as: "paint the top edge of the remaining
    rectangle, drop past it and the gap row, turn the rest a quarter turn, and
    repeat." That treats each step as one edge line of a rectangle that keeps
    shrinking and turning, so each step is one spiral arm. The maker's reading
    uses the same unit: arms traced in order, each one shorter than the last.
    The two readings agree, so the route stays the same: one Color3 per arm,
    in drawing order (top, down the right side, left along the bottom, up the
    left side, ...). Rotating the grid itself to follow the golf literally is
    unsafe here, because ARCLE's Rotate breaks on non-square grids.

    Each selection is a straight line of cells found only from I.shape.
    Nothing is read from O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    H, W = hi, wi + 1  # internal canvas, including the virtual (cropped) column 0
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}

    segments = [[(0, c) for c in range(W)]]  # arm 0: the whole top row
    sp = (0, W - 1)
    direc = (1, 0)
    lw, lh, ld = W, H, H
    isverti = False
    guard = 0
    while ld > 0 and guard < 4 * (H + W) + 8:
        guard += 1
        lw -= 1
        lh -= 1
        segments.append([(sp[0] + direc[0] * k, sp[1] + direc[1] * k) for k in range(ld)])
        ep = (sp[0] + direc[0] * (ld - 1), sp[1] + direc[1] * (ld - 1))
        direc = direcmapper[direc]
        ld = lh if isverti else lw
        isverti = not isverti
        sp = ep

    ops, sels = [], []
    painted = set()
    for cells in segments:
        real, seen = [], set()
        for (r, c) in cells:
            if 0 <= r < H and 1 <= c < W:
                rc = (r, c - 1)
                if rc not in seen:
                    seen.add(rc)
                    real.append(rc)
        if not real:
            continue  # this arm lies entirely in the cropped virtual column
        if all(x in painted for x in real):
            continue  # this arm would only repaint cells that are already 3
        ops.append(3)  # Color3: draw this spiral arm
        sels.append(sel_of(real))
        painted.update(real)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
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
                                f"for task 28e73c20"
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
                    f"Failed to build a complete episode for task 28e73c20 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"28e73c20-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
