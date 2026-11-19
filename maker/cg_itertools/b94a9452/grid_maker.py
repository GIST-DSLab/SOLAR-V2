"""
ARC Task: b94a9452 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, outer, inner = random.sample(cols, 3)
    return {"bgc": bgc, "outer": outer, "inner": inner}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, outer=None, inner=None, **kwargs) -> dict:
    from random import randint, choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or outer is None or inner is None:
        bgc, outer, inner = random.sample(list(range(10)), 3)

    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    h = max(4, min(h, max_h))
    w = max(4, min(w, max_w))
    gi = np.full((h, w), bgc, dtype=int)
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    oh = max(3, min(oh, h - 1))
    ow = max(3, min(ow, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = randint(loci + 1, loci + oh - oh2 - 1)
    locj2 = randint(locj + 1, locj + ow - ow2 - 1)
    gi[loci:loci + oh, locj:locj + ow] = outer
    gi[loci2:loci2 + oh2, locj2:locj2 + ow2] = inner
    sub = gi[loci:loci + oh, locj:locj + ow].copy()
    go = sub.copy()
    go[sub == outer] = inner
    go[sub == inner] = outer
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (shared by the maker and the golf entry; they agree):
    #   1. discard the background rows/columns -> crop to the object's bbox
    #   2. the two object colours trade places (an involution: x -> a+b-x)
    # Difference made here: frame vs block is told apart by structure (the
    # frame colour is on the bbox border, the block never touches it), not by
    # pixel counts, which can tie or invert when the block is large.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # bgc: the generator never lets the rectangle span a full row or column,
    # so some full row/column is pure background.
    bgc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            bgc = int(I[r, 0])
            break
    if bgc is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                bgc = int(I[0, c])
                break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    cells = np.argwhere(I != bgc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    sub = I[r0:r1 + 1, c0:c1 + 1]

    # frame colour = the colour on the bbox border (the block never touches it)
    outer = int(sub[0, 0])
    inner_vals = [int(v) for v in set(sub.flatten().tolist()) if int(v) != outer]
    inner = inner_vals[0]

    ops, sels = [], []

    # 1. crop the canvas to the object's bounding box (the full rectangle)
    ops.append(33)
    sels.append([r0, c0, ho - 1, wo - 1])

    # after the crop, the grid equals sub; name both colour classes from it
    outer_cells = [(r, c) for r in range(ho) for c in range(wo) if int(sub[r, c]) == outer]
    inner_cells = [(r, c) for r in range(ho) for c in range(wo) if int(sub[r, c]) == inner]

    # 2. all frame-coloured cells take the block colour
    ops.append(inner)
    sels.append(sel_of(outer_cells))

    # 3. all block-coloured cells (as named before step 2) take the frame colour
    ops.append(outer)
    sels.append(sel_of(inner_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task b94a9452"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b94a9452"
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
                                f"for task b94a9452"
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
                    f"Failed to build a complete episode for task b94a9452 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b94a9452-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
