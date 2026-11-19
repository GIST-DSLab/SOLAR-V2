"""
ARC Task: 6150a2bd (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour is a fixed role. The scattered foreground colours
    # do not affect the rule (a 180° rotation of the whole grid), so they can vary.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.choice(range(10))
    max_h = max(2, min(30, max_h))
    max_w = max(2, min(30, max_w))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    for _ in range(200):
        h = max(2, unifint(diff_lb, diff_ub, (1, max_h)))
        w = max(2, unifint(diff_lb, diff_ub, (1, max_w)))
        gi = np.full((h, w), bgc, dtype=int)
        numc = max(1, unifint(diff_lb, diff_ub, (0, min(9, h * w))))
        numc = min(numc, 9)
        colsch = random.sample(remcols, numc)
        inds = [(r, c) for r in range(h) for c in range(w)]
        for col in colsch:
            if not inds:
                break
            num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
            num = max(1, min(num, len(inds)))
            chos = random.sample(inds, num)
            for (r, c) in chos:
                gi[r, c] = col
            chos_set = set(chos)
            inds = [x for x in inds if x not in chos_set]
        # Both mirror steps must visibly change the grid.
        if np.array_equal(gi, np.fliplr(gi)) or np.array_equal(gi, np.flipud(gi)):
            continue
        go = np.rot90(gi, 2)
        return {"input": gi.tolist(), "output": go.tolist()}

    # fallback: a guaranteed asymmetric grid
    h, w = 2, 2
    gi = np.full((h, w), bgc, dtype=int)
    gi[0, 0] = remcols[0]
    return {"input": gi.tolist(), "output": np.rot90(gi, 2).tolist()}


def derive_operations(I, O=None, examples=None):
    # The code-golf entry reads the task as: reverse the order of the rows
    # (m[::-1]) and reverse each row (r[::-1]). In other words, the whole grid is
    # one unit turned through 180°. The maker reads it the same way: FlipH and
    # FlipV on the full grid. Both readings treat the same unit and need the same
    # two mirror steps, which commute, so there is nothing to change.
    # Selection: the full grid rectangle, which is exactly the region being
    # transformed (background included).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    sel = [0, 0, hi - 1, wi - 1]
    ops, sels = [], []
    # Reverse the row order (m[::-1]) -> FlipV, only if it visibly changes the grid
    cur = I
    if not np.array_equal(cur, np.flipud(cur)):
        ops.append(27); sels.append(sel)
        cur = np.flipud(cur)
    # Reverse each row (r[::-1]) -> FlipH, only if it visibly changes the grid
    if not np.array_equal(cur, np.fliplr(cur)):
        ops.append(26); sels.append(sel)
        cur = np.fliplr(cur)
    ops.append(34); sels.append(sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 6150a2bd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6150a2bd"
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
                                f"for task 6150a2bd"
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
                    f"Failed to build a complete episode for task 6150a2bd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6150a2bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
