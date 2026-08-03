"""
ARC Task: c59eb873 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, palette=None) -> dict:
    if palette is None:
        palette = [c for c in range(10) if c != bgc]
        random.shuffle(palette)

    # output is upscale(input, 2) -> input side limited to max/2
    h_ub = max(1, min(15, max_h // 2))
    w_ub = max(1, min(15, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (1, h_ub))
    w = _unifint(diff_lb, diff_ub, (1, w_ub))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    numc = _unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = palette[:numc]
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // max(1, numc))))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r][c] = col
        chosen = set(chos)
        inds = [p for p in inds if p not in chosen]

    go = []
    for r in range(h):
        row = []
        for c in range(w):
            row.append(gi[r][c])
            row.append(gi[r][c])
        go.append(list(row))
        go.append(list(row))

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # rule measured from I vs O: uniform integer upscale factor k
    k = max(1, ho // hi)
    if wi > 0 and wo // wi != k:
        k = max(1, min(ho // hi, wo // wi))
    ho_t, wo_t = hi * k, wi * k

    ops, sels = [], []

    # 1. grow canvas to k*I.shape (derived from I and the measured factor, not from O)
    #    bbox selection = exactly the full target rectangle
    ops.append(33)
    sels.append([0, 0, ho_t - 1, wo_t - 1])

    # simulate: CropGrid/Resize keeps nonzero cells of I at top-left, rest 0
    cur = np.zeros((ho_t, wo_t), dtype=int)
    cur[:hi, :wi] = I

    # 2. stamp each input cell's colour into its k*k block (period-k replication)
    for r in range(hi):
        for c in range(wi):
            col = int(I[r, c])
            r0, c0 = r * k, c * k
            block = cur[r0:r0 + k, c0:c0 + k]
            if np.all(block == col):
                continue  # already exactly this colour -> op would have no effect
            # bbox is exactly the full k*k block being painted
            ops.append(col)
            sels.append([r0, c0, k - 1, k - 1])
            cur[r0:r0 + k, c0:c0 + k] = col

    ops.append(34)
    sels.append([0, 0, ho_t - 1, wo_t - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task c59eb873"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c59eb873"
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
                                f"for task c59eb873"
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
                    f"Failed to build a complete episode for task c59eb873 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c59eb873-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
