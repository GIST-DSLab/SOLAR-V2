"""
ARC Task: 272f95fa (RE-ARC) — LLM-generated grid_maker
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

# Reading of the task: two full lines across and two full lines down cut the grid
# into a 3x3 table of background compartments. Each compartment is one connected
# background object. The centre one (it touches no border) becomes 6. The four
# compartments next to it become 2 (above), 1 (below), 4 (left) and 3 (right).
# The four corner compartments stay as they are.
# The maker paints five rectangles that it computes. The golf entry looks up a
# colour for each of the 9 bands. Here each compartment is treated as a
# connected object (the verifier's view), so each one is recoloured with one
# FloodFill. The seed is a cell picked from the line positions in I.

_ALLOWED = [0, 5, 7, 8, 9]


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    v = lo + int(round(d * (hi - lo)))
    return max(lo, min(hi, v))


def sample_colors(num_examples=None) -> dict:
    bgc, linc = random.sample(_ALLOWED, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(_ALLOWED, 2)
    h = _unifint(diff_lb, diff_ub, 5, max(5, min(30, max_h)))
    w = _unifint(diff_lb, diff_ub, 5, max(5, min(30, max_w)))
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for _ in range(h - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for _ in range(w - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    random.shuffle(hdist)
    random.shuffle(wdist)

    def adjust(dist):
        d1 = _unifint(diff_lb, diff_ub, 0, dist[0] - 1)
        dist[0] -= d1
        dist[1] += d1
        d2 = _unifint(diff_lb, diff_ub, 0, min(dist[1], dist[2]) - 1)
        d2 = random.choice((d2, -d2))
        dist[1] += d2
        dist[2] -= d2
        return dist

    hdist = adjust(hdist)
    wdist = adjust(wdist)

    r1 = hdist[0]
    r2 = r1 + 1 + hdist[1]
    c1 = wdist[0]
    c2 = c1 + 1 + wdist[1]
    g = np.full((h, w), bgc, dtype=int)
    g[[r1, r2], :] = linc
    g[:, [c1, c2]] = linc

    fns = [
        lambda a: a,
        lambda a: a.T,
        lambda a: np.rot90(a, 2).T,
        lambda a: np.fliplr(a),
        lambda a: np.flipud(a),
        lambda a: np.rot90(a, 3),
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a, 1),
    ]
    for fn in random.sample(fns, random.choice((1, 2))):
        g = np.ascontiguousarray(fn(g))

    H, W = g.shape
    rows = [r for r in range(H) if len(set(g[r, :].tolist())) == 1]
    cols = [c for c in range(W) if len(set(g[:, c].tolist())) == 1]
    a1, a2 = min(rows), max(rows)
    b1, b2 = min(cols), max(cols)
    o = g.copy()
    o[a1 + 1:a2, b1 + 1:b2] = 6
    o[:a1, b1 + 1:b2] = 2
    o[a2 + 1:, b1 + 1:b2] = 1
    o[a1 + 1:a2, :b1] = 4
    o[a1 + 1:a2, b2 + 1:] = 3
    return {"input": g.tolist(), "output": o.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Every other row and column is broken by line cells, so the rows and
    # columns that are a single colour are exactly the two lines each way.
    line_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    r1, r2 = min(line_rows), max(line_rows)
    c1, c2 = min(line_cols), max(line_cols)

    # One seed cell inside each compartment that gets coloured. Every
    # compartment is a connected region of background, and the lines keep it
    # apart from the others, so one FloodFill recolours exactly that object.
    seeds = [
        (6, (r1 + 1, c1 + 1)),   # centre: touches no border
        (2, (0, c1 + 1)),        # same columns as the centre, topmost
        (1, (hi - 1, c1 + 1)),   # same columns as the centre, bottommost
        (4, (r1 + 1, 0)),        # same rows as the centre, leftmost
        (3, (r1 + 1, wi - 1)),   # same rows as the centre, rightmost
    ]

    ops, sels = [], []
    for color, (r, c) in seeds:
        ops.append(10 + color)
        sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
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
                                f"for task 272f95fa"
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
                    f"Failed to build a complete episode for task 272f95fa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"272f95fa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
