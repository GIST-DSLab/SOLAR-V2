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

MAPPING = [
    (1, frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)})),
    (2, frozenset({(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)})),
    (3, frozenset({(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)})),
    (6, frozenset({(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})),
]
PATTERN_SET = {p for _, p in MAPPING}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    idxs = list(range(len(MAPPING)))
    if n_ex >= len(idxs):
        ex = idxs[:] + [random.choice(idxs) for _ in range(n_ex - len(idxs))]
        random.shuffle(ex)
    else:
        ex = random.sample(idxs, n_ex)
    plan = [{"variant": v} for v in ex] + [{"variant": random.choice(ex)}]
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, variant=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None or objc is None:
        bgc, objc = random.sample(range(10), 2)
    if variant is None:
        variant = random.randrange(len(MAPPING))
    col, obj = MAPPING[variant]
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_w))))
    fac = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 3)))
    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 3 * fac)
    locj = random.randint(0, w - 3 * fac)
    for (i, j) in obj:
        gi[loci + i * fac:loci + (i + 1) * fac, locj + j * fac:locj + (j + 1) * fac] = objc
    go = [[col]]
    return {"input": gi.tolist(), "output": go}


def _find_stamp(I):
    # which colour forms an upscaled 3x3 stamp from the known pattern family
    for cand in sorted(set(I.flatten().tolist())):
        pos = np.argwhere(I == cand)
        r0, c0 = pos.min(0).tolist()
        r1, c1 = pos.max(0).tolist()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh != bw or bh % 3 != 0:
            continue
        f = bh // 3
        cells = set()
        ok = True
        for i in range(3):
            for j in range(3):
                blk = I[r0 + i * f:r0 + (i + 1) * f, c0 + j * f:c0 + (j + 1) * f]
                vals = set(blk.flatten().tolist())
                if vals == {cand}:
                    cells.add((i, j))
                elif cand in vals:
                    ok = False
        if ok and frozenset(cells) in PATTERN_SET:
            return cand, frozenset(cells)
    # fallback: non-majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pos = np.argwhere(I != bgc)
    r0, c0 = pos.min(0).tolist()
    r1, c1 = pos.max(0).tolist()
    f = max(1, (r1 - r0 + 1) // 3)
    cand = int(I[pos[0][0], pos[0][1]])
    cells = frozenset((i, j) for i in range(3) for j in range(3)
                      if r0 + i * f < I.shape[0] and c0 + j * f < I.shape[1]
                      and I[r0 + i * f, c0 + j * f] != bgc)
    return cand, cells


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    objc, pattern = _find_stamp(I)

    # rule: largest 4-connected component of the 3x3 stamp picks the verdict colour
    best, seen = 0, set()
    for s in pattern:
        if s in seen:
            continue
        stack, sz = [s], 0
        seen.add(s)
        while stack:
            r, c = stack.pop()
            sz += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in pattern and n not in seen:
                    seen.add(n)
                    stack.append(n)
        best = max(best, sz)
    color = {1: 2, 4: 3, 5: 6}.get(best, 1)

    # the stamp object: every cell of the stamp colour
    obj_cells = [tuple(p) for p in np.argwhere(I == objc).tolist()]
    obj_cells = sorted(obj_cells)

    ops, sels = [], []
    # 1. recolour the whole stamp object with its shape-class verdict colour
    ops.append(color); sels.append(sel_of(obj_cells))
    # 2. shrink the canvas down to one cell of the (now recoloured) stamp
    ops.append(33); sels.append(sel_of([obj_cells[0]]))
    ops.append(34); sels.append([0, 0, 0, 0])
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
