"""
ARC Task: a68b268e (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc, c1, c2, c3, c4 = random.sample(cols, 6)
    return {"bgc": bgc, "linc": linc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None,
             c1=None, c2=None, c3=None, c4=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        lo = max(a, min(lo, b))
        hi_ = max(lo, min(hi_, b))
        return random.randint(lo, hi_)

    if None in (bgc, linc, c1, c2, c3, c4):
        bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(4, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]

    def count():
        d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)

    go = [[bgc] * w for _ in range(h)]
    layers = []
    for col in (c1, c2, c3, c4):
        cells = random.sample(inds, count())
        layer = [[bgc] * w for _ in range(h)]
        for (i, j) in cells:
            layer[i][j] = col
            go[i][j] = col
        layers.append(layer)
    LR, LL, UR, UL = layers
    gi = [[linc] * (2 * w + 1) for _ in range(2 * h + 1)]
    for i in range(h):
        for j in range(w):
            gi[i + h + 1][j + w + 1] = LR[i][j]
            gi[i + h + 1][j] = LL[i][j]
            gi[i][j + w + 1] = UR[i][j]
            gi[i][j] = UL[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2

    # four h x w transparency layers (separator row/col excluded)
    quads = [I[0:h, 0:w],              # UL  (top layer)
             I[0:h, wi - w:wi],        # UR
             I[hi - h:hi, 0:w],        # LL
             I[hi - h:hi, wi - w:wi]]  # LR  (bottom layer)

    # background = the one colour shared by all four quadrants
    pals = [set(q.flatten().tolist()) for q in quads]
    common = pals[0] & pals[1] & pals[2] & pals[3]
    if common:
        bgc = min(common)
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # each quadrant = one foreground colour drawn on bgc
    layers = []
    for q in quads:
        fg = sorted(set(q.flatten().tolist()) - {bgc})
        col = fg[0] if fg else None
        cells = set() if col is None else {
            (r, c) for r in range(h) for c in range(w) if q[r, c] == col}
        layers.append((col, cells))

    ops, sels = [], []
    # take the top-left quadrant as the canvas (full rectangle -> bbox ok)
    ops.append(33)
    sels.append([0, 0, h - 1, w - 1])

    # stack the layers bottom-up: LR, then LL, then UR, each painted whole
    # (all cells of that quadrant's colour). A layer that is completely hidden
    # under the layers above it (known from I) would have no effect -> skip.
    included = []
    for k in (3, 2, 1):
        col, cells = layers[k]
        if col is None or not cells:
            continue
        upper = set()
        for j in range(0, k):
            upper |= layers[j][1]
        if cells - upper:
            included.append(k)
            ops.append(col)
            sels.append(sel_of(sorted(cells)))

    # finally lay the top layer (UL) back on — needed only where the lower
    # layers painted over its cells
    ul_col, ul_cells = layers[0]
    painted = set()
    for k in included:
        painted |= layers[k][1]
    if ul_col is not None and (ul_cells & painted):
        ops.append(ul_col)
        sels.append(sel_of(sorted(ul_cells)))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a68b268e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a68b268e"
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
                                f"for task a68b268e"
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
                    f"Failed to build a complete episode for task a68b268e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a68b268e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
