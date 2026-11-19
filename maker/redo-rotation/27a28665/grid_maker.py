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
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of

# (output color, 3x3 shape cells) — exactly the generator's `mapping`
MAPPING = [
    (1, frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)})),
    (2, frozenset({(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)})),
    (3, frozenset({(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)})),
    (6, frozenset({(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})),
]
_PATTERNS = {p: c for c, p in MAPPING}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)

    # The 3x3 shape is the discrete structural variant (it決 determines the output
    # color), so every shape that can appear in the test must appear in examples.
    n_ex = num_examples if num_examples else 3
    idxs = list(range(len(MAPPING)))
    if n_ex >= len(idxs):
        ex = idxs[:] + [random.choice(idxs) for _ in range(n_ex - len(idxs))]
        random.shuffle(ex)
    else:
        ex = random.sample(idxs, n_ex)

    plan = [{"shape_idx": i} for i in ex]
    plan.append({"shape_idx": random.choice(ex)})  # test shape was shown
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, shape_idx=None) -> dict:
    if shape_idx is None:
        shape_idx = random.randrange(len(MAPPING))
    col, obj = MAPPING[shape_idx]

    max_h = max(3, int(max_h))
    max_w = max(3, int(max_w))

    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (3, max_w))
    fac = _unifint(diff_lb, diff_ub, (1, min(h, w) // 3))
    fac = max(1, fac)

    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - 3 * fac)
    locj = random.randint(0, w - 3 * fac)
    for (i, j) in obj:
        for di in range(fac):
            for dj in range(fac):
                gi[loci + i * fac + di][locj + j * fac + dj] = objc

    go = [[col]]
    return {"input": gi, "output": go}


def _find_object(I):
    """Return (object_color, sorted object cells) by matching an upscaled 3x3
    mapping shape. Everything is measured from I: bbox must be a square whose
    side is a multiple of 3, cell count must be 5*fac**2, every fac x fac block
    must be uniform, and the resulting 3x3 pattern must be one of the four."""
    hi, wi = I.shape
    for c in sorted(set(I.flatten().tolist())):
        cells = [(r, k) for r in range(hi) for k in range(wi) if int(I[r, k]) == c]
        rs = [r for r, _ in cells]
        ks = [k for _, k in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(ks), max(ks)
        side = r1 - r0 + 1
        if side != (c1 - c0 + 1) or side % 3 != 0:
            continue
        fac = side // 3
        if len(cells) != 5 * fac * fac:
            continue
        cellset = set(cells)
        pat = set()
        ok = True
        for i in range(3):
            for j in range(3):
                inb = sum(
                    1
                    for di in range(fac)
                    for dj in range(fac)
                    if (r0 + i * fac + di, c0 + j * fac + dj) in cellset
                )
                if inb == fac * fac:
                    pat.add((i, j))
                elif inb != 0:
                    ok = False
        if ok and frozenset(pat) in _PATTERNS:
            return c, sorted(cells)

    # Fallback: rarest color is the object
    cnt = Counter(I.flatten().tolist())
    c = min(cnt, key=lambda k: cnt[k])
    return c, sorted((r, k) for r in range(hi) for k in range(wi) if int(I[r, k]) == c)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    answer = int(O[0, 0])
    objc, obj_cells = _find_object(I)

    ops, sels = [], []

    # 1. The shape identifies the answer color: recolor the whole shape to it.
    #    (Skipped when the shape already wears that color — the op would be a no-op.)
    if objc != answer:
        ops.append(answer)
        sels.append(sel_of(obj_cells))

    # 2. Shrink the canvas down to a single cell of that shape -> 1x1 answer.
    #    Single cell: the mask IS exactly this 1x1 rectangle.
    r, c = obj_cells[0]
    ops.append(33)
    sels.append(sel_of([(r, c)]))

    ops.append(34)
    sels.append(sel_of([(0, 0)]))
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
