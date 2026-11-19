"""
ARC Task: 7b7f7511 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
except Exception:
    pass

VARIANTS = [
    {"stack": "vertical"},    # input = vconcat(go, go): top half repeats bottom half
    {"stack": "horizontal"},  # input = hconcat(go, go): left half repeats right half
]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def _halves_equal(g, axis):
    g = np.asarray(g)
    if axis == 0:
        k = g.shape[0] // 2
        return k >= 1 and g.shape[0] % 2 == 0 and np.array_equal(g[:k], g[k:])
    k = g.shape[1] // 2
    return k >= 1 and g.shape[1] % 2 == 0 and np.array_equal(g[:, :k], g[:, k:])


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, stack=None, **kwargs) -> dict:
    if stack is None:
        stack = random.choice(VARIANTS)["stack"]
    cols = list(range(10))
    for _ in range(200):
        if stack == "horizontal":
            # go: h x w ; gi: h x 2w
            hmax = min(30, max_h)
            wmax = min(15, max_w // 2)
        else:
            # go transposed: w x h ; gi: 2w x h
            hmax = min(30, max_w)
            wmax = min(15, max_h // 2)
        hmax = max(2, hmax)
        wmax = max(2, wmax)
        h = unifint(diff_lb, diff_ub, (2, hmax))
        w = unifint(diff_lb, diff_ub, (2, wmax))
        go = canvas(bgc, (h, w))
        remcols = [c for c in cols if c != bgc]
        numc = unifint(diff_lb, diff_ub, (1, min(9, h * w - 1)))
        colsch = random.sample(remcols, numc)
        inds = list(totuple(asindices(go)))
        for col in colsch:
            num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
            num = min(num, len(inds))
            if num <= 0:
                break
            chos = random.sample(inds, num)
            go = fill(go, col, frozenset(chos))
            chs = set(chos)
            inds = [x for x in inds if x not in chs]
        if stack == "vertical":
            go = dmirror(go)
            gi = vconcat(go, go)
        else:
            gi = hconcat(go, go)
        # reject grids where both halvings are possible (ambiguous reading)
        if _halves_equal(gi, 0) and _halves_equal(gi, 1):
            continue
        return {"input": gi, "output": go}
    raise ValueError("could not build unambiguous instance")


def derive_operations(I, O, examples=None):
    # Reading: the input is one tile repeated twice along a single axis.
    # Look at the row axis first: if the top half repeats as the bottom half,
    # the tile is the top half; otherwise the repeat runs along the columns
    # (the left half repeats as the right half) and the tile is the left half.
    # If both halvings are possible, the verifier's preference (left half) wins.
    # Either way the answer is one rectangular crop: CropGrid over the tile's
    # full rectangle, background included, at the top-left of the canvas.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    rows_repeat = _halves_equal(I, 0)
    cols_repeat = _halves_equal(I, 1)

    if rows_repeat and not cols_repeat:
        ho, wo = hi // 2, wi          # keep the top copy of the vertical repeat
    else:
        ho, wo = hi, wi // 2          # keep the left copy of the horizontal repeat

    ops, sels = [], []
    # Crop to the full tile rectangle (exact rectangle intended).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7b7f7511"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7b7f7511"
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
                                f"for task 7b7f7511"
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
                    f"Failed to build a complete episode for task 7b7f7511 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7b7f7511-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
