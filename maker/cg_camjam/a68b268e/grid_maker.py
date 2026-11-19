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
             c1=None, c2=None, c3=None, c4=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if None in (bgc, linc, c1, c2, c3, c4):
        bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(4, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]

    def ncount():
        d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)

    ofc = [random.sample(inds, ncount()) for _ in range(4)]
    go = np.full((h, w), bgc, dtype=int)
    for col, cells in zip((c1, c2, c3, c4), ofc):
        for (i, j) in cells:
            go[i, j] = col

    def layer(col, cells):
        g = np.full((h, w), bgc, dtype=int)
        for (i, j) in cells:
            g[i, j] = col
        return g

    LR = layer(c1, ofc[0]); LL = layer(c2, ofc[1])
    UR = layer(c3, ofc[2]); UL = layer(c4, ofc[3])
    gi = np.full((2 * h + 1, 2 * w + 1), linc, dtype=int)
    gi[h + 1:, w + 1:] = LR
    gi[h + 1:, :w] = LL
    gi[:h, w + 1:] = UR
    gi[:h, :w] = UL
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2

    quads = {"UL": I[0:h, 0:w],
             "UR": I[0:h, wi - w:wi],
             "LL": I[hi - h:hi, 0:w],
             "LR": I[hi - h:hi, wi - w:wi]}

    pals = [set(q.flatten().tolist()) for q in quads.values()]
    common = pals[0] & pals[1] & pals[2] & pals[3]
    bgc = min(common) if common else Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    # 1. Crop the canvas down to the top-left quadrant: this is the h x w sheet
    #    onto which the four transparent layers are stacked (full rectangle).
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    cur = I[0:h, 0:w].copy()

    # 2. Painter's algorithm: stack the quadrants from lowest priority to highest
    #    (LR, LL, UR, then UL on top). Each layer = that quadrant's
    #    non-background cells, painted in its own colour.
    for name in ("LR", "LL", "UR", "UL"):
        q = quads[name]
        by_col = {}
        for r in range(h):
            for c in range(w):
                v = int(q[r, c])
                if v == bgc:
                    continue
                if cur[r, c] == v:      # already holds this colour right now
                    continue
                by_col.setdefault(v, []).append((r, c))
        for col in sorted(by_col):
            cells = by_col[col]
            ops.append(col); sels.append(sel_of(cells))
            for (r, c) in cells:
                cur[r, c] = col

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
