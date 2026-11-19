"""
ARC Task: 27a28665 (RE-ARC) — LLM-generated grid_maker
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

# The four 3x3 stamps the generator can draw, with the colour each one maps to.
MAPPING = [
    (1, frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)})),
    (2, frozenset({(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)})),
    (3, frozenset({(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)})),
    (6, frozenset({(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})),
]
PATTERN_SET = {cells: col for col, cells in MAPPING}

# The code-golf reading: only the stamp's TOP block-row is needed to tell the four apart.
TOP_ROW_RULE = {
    (True, True, False): 1,
    (True, False, True): 2,
    (False, True, True): 3,
    (False, True, False): 6,
}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    variants = list(range(len(MAPPING)))
    if n_ex >= len(variants):
        ex = variants[:] + [random.choice(variants) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = random.sample(variants, n_ex)
    plan = [{"variant": v} for v in ex]
    plan.append({"variant": random.choice(ex)})
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, variant=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if variant is None:
        variant = random.randrange(len(MAPPING))
    col, obj = MAPPING[variant]
    max_h = max(3, min(30, max_h))
    max_w = max(3, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (3, max_h))
        w = unifint(diff_lb, diff_ub, (3, max_w))
        fac = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 3)))
        # if the stamp filled the whole grid, plus and corners would be complements
        # of each other and could not be told apart, so skip that case
        if not (h == 3 * fac and w == 3 * fac):
            break
        if max_h == 3 and max_w == 3:
            break
    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 3 * fac)
    locj = random.randint(0, w - 3 * fac)
    for (i, j) in obj:
        gi[loci + i * fac:loci + (i + 1) * fac, locj + j * fac:locj + (j + 1) * fac] = objc
    go = [[col]]
    return {"input": gi.tolist(), "output": go}


def _find_stamp(I):
    """Find the colour whose cells form an upscaled 3x3 stamp; return (r0, c0, f, blocks)."""
    for cand in sorted(set(I.flatten().tolist())):
        pos = np.argwhere(I == cand)
        r0, c0 = pos.min(0).tolist()
        r1, c1 = pos.max(0).tolist()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh != bw or bh % 3 != 0:
            continue
        f = bh // 3
        blocks = np.zeros((3, 3), dtype=bool)
        ok = True
        for i in range(3):
            for j in range(3):
                vals = set(I[r0 + i * f:r0 + (i + 1) * f,
                             c0 + j * f:c0 + (j + 1) * f].flatten().tolist())
                if vals == {cand}:
                    blocks[i, j] = True
                elif cand in vals:
                    ok = False
        if not ok:
            continue
        cells = frozenset((i, j) for i in range(3) for j in range(3) if blocks[i, j])
        if cells in PATTERN_SET:
            return r0, c0, f, blocks
    # fallback: background = majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pos = np.argwhere(I != bgc)
    r0, c0 = pos.min(0).tolist()
    r1, _ = pos.max(0).tolist()
    f = max(1, (r1 - r0 + 1) // 3)
    blocks = np.array([[I[min(r0 + i * f, I.shape[0] - 1), min(c0 + j * f, I.shape[1] - 1)] != bgc
                        for j in range(3)] for i in range(3)], dtype=bool)
    return r0, c0, f, blocks


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    r0, c0, f, blocks = _find_stamp(I)

    # Reading: the stamp's top block-row on its own decides the answer colour.
    top = tuple(bool(x) for x in blocks[0])
    if top in TOP_ROW_RULE:
        color = TOP_ROW_RULE[top]
    else:
        color = PATTERN_SET.get(
            frozenset((i, j) for i in range(3) for j in range(3) if blocks[i, j]), 1)

    ops, sels = [], []
    # 1. Shrink the canvas to the stamp's top-left cell, which starts the deciding top row.
    ops.append(33); sels.append(sel_of([(r0, c0)]))
    # 2. Paint that one remaining cell with the colour the top row selects
    #    (skip it if the cell already holds that colour, since the op would change nothing).
    cell_val = int(I[r0, c0])
    if cell_val != color:
        ops.append(color); sels.append(sel_of([(0, 0)]))
    ops.append(34); sels.append(sel_of([(0, 0)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 27a28665"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 27a28665"
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
                                f"for task 27a28665"
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
                    f"Failed to build a complete episode for task 27a28665 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"27a28665-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
