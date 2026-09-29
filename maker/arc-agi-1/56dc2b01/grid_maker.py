"""
ARC Task: 56dc2b01 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    if bgc is None:
        bgc = random.choice(cols)
    if objc is None or objc == bgc:
        objc = random.choice([c for c in cols if c != bgc])
    h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    oh = _unifint(diff_lb, diff_ub, (1, h))
    ow = _unifint(diff_lb, diff_ub, (1, max(1, (w - 1) // 2 - 1)))
    bb = set((i, j) for i in range(oh) for j in range(ow))
    sp = random.choice(sorted(bb))
    obj = {sp}
    bb.discard(sp)
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for _ in range(ncells):
        cand = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    cand.add((i + di, j + dj))
        cand = sorted((bb - obj) & cand)
        if not cand:
            break
        obj.add(random.choice(cand))
    mi = min(i for i, j in obj); mj = min(j for i, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    loci = random.randint(0, h - oh)
    locj = _unifint(diff_lb, diff_ub, (1, w - ow))
    gi = np.full((h, w), bgc, dtype=int)
    barlocji = _unifint(diff_lb, diff_ub, (0, locj))
    barlocj = locj - barlocji
    barlocj = min(max(0, barlocj), locj - 1)
    gi[:, barlocj] = 2
    go = gi.copy()
    for i, j in obj:
        go[loci + i, barlocj + 1 + j] = objc
    if barlocj + ow + 1 < w:
        go[:, barlocj + ow + 1] = 8
    for i, j in obj:
        gi[loci + i, locj + j] = objc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, 3),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = canvas colour (strict majority: object width < half grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    twos = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 2]
    two_rows = set(r for r, c in twos)
    two_cols = set(c for r, c in twos)
    vertical_bar = (len(two_cols) == 1)

    obj = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] not in (bgc, 2)]
    objc = int(I[obj[0]])
    r0 = min(r for r, c in obj); r1 = max(r for r, c in obj)
    c0 = min(c for r, c in obj); c1 = max(c for r, c in obj)

    if vertical_bar:
        b = next(iter(two_cols))
        ow = c1 - c0 + 1
        if c0 > b:
            steps = c0 - (b + 1)
            move_op, dstep = 23, (0, -1)
            line_idx = b + ow + 1
        else:
            steps = (b - 1) - c1
            move_op, dstep = 22, (0, 1)
            line_idx = b - ow - 1
        line_cells = [(r, line_idx) for r in range(hi) if 0 <= line_idx < wi]
    else:
        b = next(iter(two_rows))
        oh = r1 - r0 + 1
        if r0 > b:
            steps = r0 - (b + 1)
            move_op, dstep = 20, (-1, 0)
            line_idx = b + oh + 1
        else:
            steps = (b - 1) - r1
            move_op, dstep = 21, (1, 0)
            line_idx = b - oh - 1
        line_cells = [(line_idx, c) for c in range(wi) if 0 <= line_idx < hi]

    if steps > 0:
        dest = [(r + dstep[0] * steps, c + dstep[1] * steps) for r, c in obj]
        hole = sorted(set(obj) - set(dest))
        if objc != 0:
            # slide the object until it touches the 2-line: one grab, then empties
            ops.append(move_op); sels.append(sel_of(obj))
            for _ in range(steps - 1):
                ops.append(move_op); sels.append(sel_of([]))
            # repair only the vacated footprint (ARCLE left it at 0)
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # colour-0 object: ARCLE Move cannot carry 0 cells, so put the
            # object down at its landing place with Color0, then clear the
            # vacated footprint to background.
            ops.append(0); sels.append(sel_of(sorted(dest)))
            if hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))

    # draw the 8 frontier just beyond the object's far edge
    if line_cells:
        ops.append(8); sels.append(sel_of(line_cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
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
                                f"for task 56dc2b01"
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
                    f"Failed to build a complete episode for task 56dc2b01 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"56dc2b01-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
