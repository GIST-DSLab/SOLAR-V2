"""
ARC Task: d6ad076f (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, c1, c2 = random.sample(cols, 3)
    return {"bgc": bgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + round((b - a) * lb), a + round((b - a) * ub))

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    inh = unifint(diff_lb, diff_ub, (3, h))
    inw = unifint(diff_lb, diff_ub, (3, w))

    itv = list(range(0, inh))
    loci2i = unifint(diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv2 = itv[:loci2i - 1][::-1]
    loci1i = unifint(diff_lb, diff_ub, (0, len(itv2) - 1))
    loci1 = itv2[loci1i]

    cp = random.randint(1, inw - 2)
    ajs = random.randint(0, cp - 1)
    aje = random.randint(cp + 1, inw - 1)
    bjs = random.randint(0, cp - 1)
    bje = random.randint(cp + 1, inw - 1)

    c = np.full((inh, inw), bgc, dtype=int)
    c[0:loci1 + 1, ajs:aje + 1] = c1          # obja (rows 0..loci1)
    c[loci2:inh, bjs:bje + 1] = c2            # objb (rows loci2..inh-1)

    loci = random.randint(0, h - inh)
    locj = random.randint(0, w - inw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + inh, locj:locj + inw] = c
    go = gi.copy()

    mr0, mr1 = loci1 + 1, loci2 - 1
    mc0, mc1 = max(ajs, bjs) + 1, min(aje, bje) - 1
    if mr1 >= mr0 and mc1 >= mc0:
        go[loci + mr0:loci + mr1 + 1, locj + mc0:locj + mc1 + 1] = 8

    if random.choice((True, False)):
        gi = gi.T.copy()
        go = go.T.copy()

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # background = dominant color; the two colored rectangles are everything else
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    colors = [int(c) for c in np.unique(I) if int(c) != bgc]

    rects = []
    for col in colors:
        rs, cs = np.where(I == col)
        rects.append((int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())))
    a, b = rects[0], rects[1]
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b

    # rectangles separated along exactly one axis; 8-region = bounded gap between them
    if ar1 < br0 or br1 < ar0:
        # stacked vertically -> gap spans rows between them
        top, bot = (a, b) if ar1 < br0 else (b, a)
        gr0 = top[1] + 1
        gr1 = bot[0] - 1
        gc0 = max(ac0, bc0) + 1
        gc1 = min(ac1, bc1) - 1
    else:
        # placed side by side -> gap spans cols between them
        left, right = (a, b) if ac1 < bc0 else (b, a)
        gc0 = left[3] + 1
        gc1 = right[2] - 1
        gr0 = max(ar0, br0) + 1
        gr1 = min(ar1, br1) - 1

    ops, sels = [], []
    ops.append(8)
    sels.append([int(gr0), int(gc0), int(gr1 - gr0), int(gc1 - gc0)])
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
                        f"num_examples+1 ({num_examples + 1}) for task d6ad076f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d6ad076f"
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
                                f"for task d6ad076f"
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
                    f"Failed to build a complete episode for task d6ad076f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d6ad076f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
