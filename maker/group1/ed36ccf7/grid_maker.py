"""
ARC Task: ed36ccf7 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def _unif(diff_lb, diff_ub, lo, hi):
    # RE-ARC style difficulty-scaled integer sampling
    if hi <= lo:
        return lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    a, b = max(lo, a), min(hi, b)
    if b < a:
        b = a
    return random.randint(a, b)


def sample_colors(num_examples=None) -> dict:
    # Rotation does not depend on colours; only the canvas colour is fixed per episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = _unif(diff_lb, diff_ub, 1, max_h)
    w = _unif(diff_lb, diff_ub, 1, max_w)
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc = _unif(diff_lb, diff_ub, 0, min(9, h * w))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        num = _unif(diff_lb, diff_ub, 1, max(1, len(inds) // numc))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r][c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    go = np.rot90(np.array(gi, dtype=int), 1).tolist()   # rot270 (CCW)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: output = input rotated 90 deg CCW, i.e. I[r, c] lands at (wi-1-c, r).
    # Unit of work here: ONE COLOUR AT A TIME. The canvas is reshaped to the
    # rotated size, a base layer of I's dominant colour is laid over the whole
    # canvas, then every other colour of I is painted onto the cells where its
    # input cells land under the rotation. Every selection is computed from I only.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = wi, hi                     # rotated canvas size
    ops, sels = [], []

    # current working grid simulation (to skip ops with no visible effect)
    cur = I.copy()

    # 1. Reshape canvas to the rotated dimensions (only if it actually changes)
    if (ho, wo) != (hi, wi):
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])   # full new canvas rectangle
        new = np.zeros((ho, wo), dtype=int)
        mh, mw = min(ho, hi), min(wo, wi)
        new[:mh, :mw] = cur[:mh, :mw]
        cur = new

    # where each colour of I lands after rotation
    counts = Counter(I.flatten().tolist())
    order = [c for c, _ in counts.most_common()]
    landing = {}
    for col in order:
        rs, cs = np.nonzero(I == col)
        landing[col] = [(wi - 1 - int(c), int(r)) for r, c in zip(rs, cs)]

    # 2. Base layer: dominant colour over the whole rotated canvas
    base = order[0]
    whole = [(r, c) for r in range(ho) for c in range(wo)]
    if np.any(cur != base):
        ops.append(base); sels.append(sel_of(whole))
        cur[:, :] = base

    # 3. Each remaining colour, painted at its rotated positions
    for col in order[1:]:
        cells = landing[col]
        if not cells:
            continue
        if all(cur[r, c] == col for r, c in cells):
            continue
        ops.append(int(col)); sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task ed36ccf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ed36ccf7"
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
                                f"for task ed36ccf7"
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
                    f"Failed to build a complete episode for task ed36ccf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ed36ccf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
