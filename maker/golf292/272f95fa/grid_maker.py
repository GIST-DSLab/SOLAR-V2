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

_ALLOWED = [0, 5, 7, 8, 9]  # generator excludes 1, 2, 3, 4, 6 (they are the output paint colours)


def sample_colors(num_examples=None) -> dict:
    bgc, linc = random.sample(_ALLOWED, 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, **kwargs) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(_ALLOWED, 2)
    h = _unifint(diff_lb, diff_ub, (5, max(5, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (5, max(5, min(30, max_w))))

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
        d1 = _unifint(diff_lb, diff_ub, (0, dist[0] - 1))
        dist[0] -= d1
        dist[1] += d1
        d2 = _unifint(diff_lb, diff_ub, (0, min(dist[1], dist[2]) - 1))
        d2 = random.choice((d2, -d2))
        dist[1] += d2
        dist[2] -= d2
        return dist

    hdist = adjust(hdist)
    wdist = adjust(wdist)

    g = np.full((h, w), bgc, dtype=int)
    r1 = hdist[0]
    r2 = hdist[0] + 1 + hdist[1]
    c1 = wdist[0]
    c2 = wdist[0] + 1 + wdist[1]
    g[r1, :] = linc
    g[r2, :] = linc
    g[:, c1] = linc
    g[:, c2] = linc

    mfs = [
        lambda x: x,
        lambda x: x.T,
        lambda x: np.rot90(x, 2).T,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 3),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        g = np.ascontiguousarray(fn(g))

    gi = g.copy()
    H, W = gi.shape
    lr = [r for r in range(H) if np.all(gi[r, :] == linc)]
    lc = [c for c in range(W) if np.all(gi[:, c] == linc)]
    R1, R2 = min(lr), max(lr)
    C1, C2 = min(lc), max(lc)
    go = gi.copy()
    go[R1 + 1:R2, C1 + 1:C2] = 6
    go[0:R1, C1 + 1:C2] = 2
    go[R2 + 1:H, C1 + 1:C2] = 1
    go[R1 + 1:R2, 0:C1] = 4
    go[R1 + 1:R2, C2 + 1:W] = 3
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Two full rows and two full cols of the line colour split the grid; every
    # other row/col is interrupted by a line cell, so uniform rows/cols are the lines.
    line_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    r1, r2 = min(line_rows), max(line_rows)
    c1, c2 = min(line_cols), max(line_cols)

    # The five plus-shaped bands are each an exact full rectangle, so the
    # bbox selection is exactly the band's cell set.
    center = [r1 + 1, c1 + 1, r2 - r1 - 2, c2 - c1 - 2]
    top    = [0,      c1 + 1, r1 - 1,      c2 - c1 - 2]
    bottom = [r2 + 1, c1 + 1, hi - r2 - 2, c2 - c1 - 2]
    left   = [r1 + 1, 0,      r2 - r1 - 2, c1 - 1]
    right  = [r1 + 1, c2 + 1, r2 - r1 - 2, wi - c2 - 2]

    ops, sels = [], []
    for color, sel in ((6, center), (2, top), (1, bottom), (4, left), (3, right)):
        ops.append(color)
        sels.append(sel)

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
