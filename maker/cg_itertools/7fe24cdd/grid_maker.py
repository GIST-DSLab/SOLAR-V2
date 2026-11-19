"""
ARC Task: 7fe24cdd (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # Rule is purely geometric; only the canvas background is an episode-level colour.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, **kwargs) -> dict:
    cols = list(range(10))
    hmax = max(1, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (1, hmax))
    w = h
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in cols if c != bgc]
    numc = _unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // max(1, numc))))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r, c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    top = np.hstack([gi, np.rot90(gi, 3)])
    bot = np.hstack([np.rot90(gi, 1), np.rot90(gi, 2)])
    go = np.vstack([top, bot])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the top strip is [I | rot90(I)]; the bottom strip is that whole
    # strip turned 180 degrees. Build the strip once, then copy it and turn it.
    I = np.asarray(I, dtype=int)
    h = I.shape[0]                      # generator forces w == h
    ops, sels = [], []

    g = I.copy()                        # simulated working grid (for no-op checks)
    clip = None

    def emit(op, sel, new_g):
        nonlocal g
        if new_g.shape == g.shape and np.array_equal(new_g, g):
            return False
        ops.append(op); sels.append(sel)
        g = new_g
        return True

    # 1. Widen canvas to the top strip h x 2h (full rectangle -> bbox).
    ng = np.zeros((h, 2 * h), dtype=int)
    ng[:, :h] = I
    emit(33, [0, 0, h - 1, 2 * h - 1], ng)

    # 2. CopyI the whole input square (full rectangle -> bbox).
    ops.append(28); sels.append([0, 0, h - 1, h - 1])
    clip = I.copy()

    # 3. Paste input into the right square of the strip.
    ng = g.copy()
    sub = ng[0:h, h:2 * h]
    sub[clip != 0] = clip[clip != 0]
    emit(30, [0, h, 0, 0], ng)

    # 4. Rotate that right square CW -> right half = rot90(I) (square bbox).
    ng = g.copy()
    ng[0:h, h:2 * h] = np.rot90(ng[0:h, h:2 * h], 3)
    emit(25, [0, h, h - 1, h - 1], ng)

    # 5. Extend canvas downward to 2h x 2h (full rectangle -> bbox).
    ng = np.zeros((2 * h, 2 * h), dtype=int)
    ng[:h, :] = g
    emit(33, [0, 0, 2 * h - 1, 2 * h - 1], ng)

    # 6. CopyO the finished top strip as one unit (full rectangle -> bbox).
    strip = g[0:h, :].copy()
    ops.append(29); sels.append([0, 0, h - 1, 2 * h - 1])

    # 7. Paste the strip into the bottom half.
    ng = g.copy()
    sub = ng[h:2 * h, :]
    sub[strip != 0] = strip[strip != 0]
    emit(30, [h, 0, 0, 0], ng)

    # 8-9. Turn the bottom strip 180 degrees: FlipH then FlipV (whole strip bbox).
    ng = g.copy()
    ng[h:2 * h, :] = np.fliplr(ng[h:2 * h, :])
    emit(26, [h, 0, h - 1, 2 * h - 1], ng)
    ng = g.copy()
    ng[h:2 * h, :] = np.flipud(ng[h:2 * h, :])
    emit(27, [h, 0, h - 1, 2 * h - 1], ng)

    # Drop copy ops whose following paste was skipped (clip unused).
    cleaned_ops, cleaned_sels = [], []
    for i, (op, sel) in enumerate(zip(ops, sels)):
        if op in (28, 29):
            nxt = ops[i + 1] if i + 1 < len(ops) else None
            if nxt != 30 and not (nxt == 33 and i + 2 < len(ops) and ops[i + 2] == 30):
                continue
        cleaned_ops.append(op); cleaned_sels.append(sel)
    ops, sels = cleaned_ops, cleaned_sels

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * h - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
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
                                f"for task 7fe24cdd"
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
                    f"Failed to build a complete episode for task 7fe24cdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7fe24cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
