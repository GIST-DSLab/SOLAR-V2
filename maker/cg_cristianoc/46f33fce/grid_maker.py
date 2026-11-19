"""
ARC Task: 46f33fce (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(2, min(7, max_h // 4))
    wmax = max(2, min(7, max_w // 4))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    remcols = [c for c in cols if c != bgc]
    go = [[bgc] * w for _ in range(h)]
    gi = [[bgc] * (2 * w) for _ in range(2 * h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    locs = random.sample(inds, nc)
    for (i, j) in locs:
        col = random.choice(remcols)
        go[i][j] = col
        gi[2 * i + 1][2 * j + 1] = col
    out = [[go[r // 4][c // 4] for c in range(4 * w)] for r in range(4 * h)]
    return {"input": gi, "output": out}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Reading: I is a grid of 2x2 cells; each cell collapses to one colour
    # (its non-background pixel, else background) and becomes a 4x4 block of the
    # output. The output canvas is therefore (hi/2*4) x (wi/2*4) = 2hi x 2wi.
    ho, wo = 2 * hi, 2 * wi
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for (ei, _eo) in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        cand = cnt.most_common(1)[0][0]
        if cand in I:
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    nh, nw = hi // 2, wi // 2

    def cell_colour(i, j):
        block = I[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten().tolist()
        fg = [v for v in block if v != bgc]
        return fg[0] if fg else bgc

    ops, sels = [], []
    # Resize the canvas to the full block grid (full canvas rectangle).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    G = np.zeros((ho, wo), dtype=int)
    G[:hi, :wi] = I  # transparent copy; zeros stay zero anyway

    # Paint each 2x2 cell's 4x4 block, last cell first, so the input pixels
    # a block covers belong to cells that have already been rendered.
    for i in reversed(range(nh)):
        for j in reversed(range(nw)):
            v = cell_colour(i, j)
            r0, c0 = 4 * i, 4 * j
            if np.all(G[r0:r0 + 4, c0:c0 + 4] == v):
                continue  # block already holds this colour: op would be a no-op
            ops.append(v)
            sels.append([r0, c0, 3, 3])  # the full 4x4 block rectangle
            G[r0:r0 + 4, c0:c0 + 4] = v

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
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
                                f"for task 46f33fce"
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
                    f"Failed to build a complete episode for task 46f33fce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"46f33fce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
