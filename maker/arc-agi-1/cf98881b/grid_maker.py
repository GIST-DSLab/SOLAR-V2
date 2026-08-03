"""
ARC Task: cf98881b (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, barcol, cola, colb, colc = random.sample(cols, 5)
    return {"bgc": bgc, "barcol": barcol, "cola": cola, "colb": colb, "colc": colc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, cola, colb, colc) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w_ub = min(9, (max_w - 2) // 3)
    w = unifint(diff_lb, diff_ub, (2, max(2, w_ub)))
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    devc = unifint(diff_lb, diff_ub, devrng)
    deva = choice((+1, -1)) * deva
    devb = choice((+1, -1)) * devb
    devc = choice((+1, -1)) * devc
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    numc = max(min(h * w - 1, mp + devc), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    c = sample(inds, numc)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gic = fill(canv, colc, c)
    gi = hconcat(hconcat(hconcat(gia, gbar), hconcat(gib, gbar)), gic)
    go = fill(gic, colb, b)
    go = fill(go, cola, a)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # three panels of width w separated by two 1-wide bar columns
    w = (wi + 1) // 3 - 1
    p1 = I[:, 0:w]
    p2 = I[:, w + 1:2 * w + 1]
    p3 = I[:, 2 * w + 2:3 * w + 2]
    off = 2 * w + 2

    s1 = set(p1.flatten().tolist())
    s2 = set(p2.flatten().tolist())
    s3 = set(p3.flatten().tolist())
    bgc = sorted(s1 & s2 & s3)[0]
    cola = sorted(s1 - {bgc})[0]
    colb = sorted(s2 - {bgc})[0]

    ops, sels = [], []

    # panel3 stays in place; overlay panel2's marks, then panel1's marks on top
    m2 = [(r, off + c) for r in range(hi) for c in range(w) if p2[r, c] != bgc]
    if m2:
        ops.append(int(colb)); sels.append(sel_of(m2))
    m1 = [(r, off + c) for r in range(hi) for c in range(w) if p1[r, c] != bgc]
    if m1:
        ops.append(int(cola)); sels.append(sel_of(m1))

    # keep only the third panel (full rectangle -> bbox selection is exact)
    ops.append(33); sels.append([0, off, hi - 1, w - 1])
    ops.append(34); sels.append([0, 0, hi - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task cf98881b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cf98881b"
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
                                f"for task cf98881b"
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
                    f"Failed to build a complete episode for task cf98881b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cf98881b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
