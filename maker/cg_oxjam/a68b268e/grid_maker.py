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
    bgc, linc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "linc": linc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, c1, c2, c3, c4) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    h_hi = max(2, min(14, (max_h - 1) // 2))
    w_hi = max(2, min(4, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (2, h_hi))
    w = unifint(diff_lb, diff_ub, (2, w_hi))
    inds = [(i, j) for i in range(h) for j in range(w)]

    def count():
        d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        n = random.choice((d, h * w - d))
        return min(max(1, n), h * w - 1)

    ofcs = [random.sample(inds, count()) for _ in range(4)]
    cols = [c1, c2, c3, c4]

    go = [[bgc] * w for _ in range(h)]
    for col, ofc in zip(cols, ofcs):
        for (i, j) in ofc:
            go[i][j] = col

    def quad(col, ofc):
        q = [[bgc] * w for _ in range(h)]
        for (i, j) in ofc:
            q[i][j] = col
        return q

    LR = quad(c1, ofcs[0])
    LL = quad(c2, ofcs[1])
    UR = quad(c3, ofcs[2])
    UL = quad(c4, ofcs[3])
    gi = [[linc] * (2 * w + 1) for _ in range(2 * h + 1)]
    for q, (dr, dc) in ((LR, (h + 1, w + 1)), (LL, (h + 1, 0)), (UR, (0, w + 1)), (UL, (0, 0))):
        for i in range(h):
            for j in range(w):
                gi[i + dr][j + dc] = q[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = hi // 2, wi // 2

    # Four h x w quadrants (separator row/col excluded). Each is bgc + one colour.
    origins = {"UL": (0, 0), "UR": (0, wi - w), "LL": (hi - h, 0), "LR": (hi - h, wi - w)}
    quads = {k: I[r:r + h, c:c + w] for k, (r, c) in origins.items()}

    pals = [set(q.flatten().tolist()) for q in quads.values()]
    common = pals[0] & pals[1] & pals[2] & pals[3]
    if common:
        bgc = min(common)
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    # Painter's reading: stack the four quadrants as layers, lowest priority first.
    # Base layer = LR quadrant: crop the canvas to that full rectangle, which carries it
    # to the top-left (full rectangle -> bbox selection is exact).
    r0, c0 = origins["LR"]
    ops.append(33)
    sels.append([r0, c0, h - 1, w - 1])

    # Then lay LL, UR and finally UL on top: each layer is the quadrant's non-background
    # cells, painted in that quadrant's colour at the same relative position.
    for key in ("LL", "UR", "UL"):
        q = quads[key]
        groups = {}
        for r in range(h):
            for c in range(w):
                v = int(q[r, c])
                if v != bgc:
                    groups.setdefault(v, []).append((r, c))
        for col in sorted(groups):
            ops.append(col)
            sels.append(sel_of(groups[col]))

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
