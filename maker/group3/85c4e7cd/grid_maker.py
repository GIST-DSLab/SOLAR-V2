"""
ARC Task: 85c4e7cd (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule (reverse the order of the nested ring colours) does not depend on
    # which colours are used, and there is no background colour. The generator
    # samples a fresh palette for each instance, and that is fine here, so there
    # are no episode-level colour roles to fix.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    hmax = max(1, min(15, max_h // 2))
    wmax = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    cols = random.sample(range(10), ncols)
    colord = [random.choice(cols) for _ in range(min(h, w))]
    H, W = h * 2, w * 2
    gi = np.zeros((H, W), dtype=int)
    go = np.zeros((H, W), dtype=int)
    for idx, (ci, co) in enumerate(zip(colord, colord[::-1])):
        r0, c0, r1, c1 = idx, idx, H - 1 - idx, W - 1 - idx
        for g, col in ((gi, ci), (go, co)):
            g[r0, c0:c1 + 1] = col
            g[r1, c0:c1 + 1] = col
            g[r0:r1 + 1, c0] = col
            g[r0:r1 + 1, c1] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    n = min(hi, wi) // 2

    # Rule read from I alone: ring idx takes the colour that ring n-1-idx has in I.
    ring_col = [int(I[i, i]) for i in range(n)]
    tgt = [ring_col[n - 1 - i] for i in range(n)]

    # Each step paints one solid rectangle and leaves the rings inside it for later
    # steps. Work from the outside in: paint the FULL solid rectangle of layer idx
    # (its ring plus everything inside it) with that layer's target colour, then
    # paint the next smaller rectangle on top. When the last rectangle is painted,
    # every ring holds its target colour.
    # Start at the outermost ring whose colour actually changes. The rings outside
    # it already hold their final colours.
    start = None
    for i in range(n):
        if ring_col[i] != tgt[i]:
            start = i
            break

    ops, sels = [], []
    if start is not None:
        cur = I.copy()
        for idx in range(start, n):
            r0, c0 = idx, idx
            r1, c1 = hi - 1 - idx, wi - 1 - idx
            region = cur[r0:r1 + 1, c0:c1 + 1]
            if np.all(region == tgt[idx]):
                continue  # this rectangle already holds the colour, so painting it would change nothing
            # The intended cells are this whole solid rectangle, so a bbox selection is exact.
            ops.append(tgt[idx])
            sels.append([r0, c0, r1 - r0, c1 - c0])
            cur[r0:r1 + 1, c0:c1 + 1] = tgt[idx]

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
                        f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 85c4e7cd"
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
                                f"for task 85c4e7cd"
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
                    f"Failed to build a complete episode for task 85c4e7cd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"85c4e7cd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
