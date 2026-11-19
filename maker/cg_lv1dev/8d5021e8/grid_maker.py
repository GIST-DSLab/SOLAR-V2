"""
ARC Task: 8d5021e8 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are full rectangles)


def sample_colors(num_examples=None) -> dict:
    # The rule is purely geometric. Only the background colour is fixed per episode.
    # The scattered foreground colours are drawn per instance inside generate().
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    # The output is 3h x 2w, so it has to fit inside max_h x max_w.
    hmax = max(1, min(10, max_h // 3))
    wmax = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gi[i][j] = col
        cs = set(chos)
        inds = [x for x in inds if x not in cs]
    gi_np = np.array(gi, dtype=int)
    go1 = np.hstack([np.fliplr(gi_np), gi_np])
    go = np.vstack([np.flipud(go1), go1, np.flipud(go1)])
    return {"input": gi_np.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf entry): first build one vertical COLUMN
    #   [flipud(I); I; flipud(I)]
    # and then treat that column as the unit that is mirrored left/right:
    #   each row r -> r[::-1] + r.
    # The maker does the opposite order (horizontal concat first, then vertical).
    # Every op depends only on I. The grid is simulated so that any op with no
    # visible effect (h==1, w==1, symmetric or all-zero input) is left out.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 2 * wi

    ops, sels = [], []
    g = np.zeros((ho, wo), dtype=int)
    g[:hi, :wi] = I
    clip = None

    def emit(op, sel):
        ops.append(op)
        sels.append(sel)

    # 1. Resize the canvas to 3h x 2w. I stays at the top-left (whole-canvas rectangle).
    emit(33, [0, 0, ho - 1, wo - 1])

    # 2. CopyI the whole input rectangle.
    clip = I.copy()
    copied_any = bool((clip != 0).any())
    if copied_any:
        emit(28, [0, 0, hi - 1, wi - 1])

        # 3-4. Paste I into the middle and bottom bands of the left column (fresh zero area).
        for r0 in (hi, 2 * hi):
            new = g.copy()
            m = clip != 0
            new[r0:r0 + hi, :wi][m] = clip[m]
            if not np.array_equal(new, g):
                g = new
                emit(30, [r0, 0, 0, 0])

    # 5-6. FlipV the top and bottom bands (whole band rectangles) -> column = [hI; I; hI].
    for r0 in (0, 2 * hi):
        band = g[r0:r0 + hi, :wi]
        fl = np.flipud(band)
        if not np.array_equal(fl, band):
            g[r0:r0 + hi, :wi] = fl
            emit(27, [r0, 0, hi - 1, wi - 1])

    # 7-8. CopyO the finished column (full rectangle) and paste it as the right half.
    col = g[:, :wi].copy()
    if (col != 0).any():
        emit(29, [0, 0, ho - 1, wi - 1])
        new = g.copy()
        m = col != 0
        new[:, wi:][m] = col[m]
        if not np.array_equal(new, g):
            g = new
            emit(30, [0, wi, 0, 0])
        else:
            ops.pop(); sels.pop()

    # 9. FlipH the left column (full rectangle) -> each row becomes r[::-1] + r.
    left = g[:, :wi]
    fl = np.fliplr(left)
    if not np.array_equal(fl, left):
        g[:, :wi] = fl
        emit(26, [0, 0, ho - 1, wi - 1])

    emit(34, [0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8d5021e8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d5021e8"
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
                                f"for task 8d5021e8"
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
                    f"Failed to build a complete episode for task 8d5021e8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d5021e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
