"""
ARC Task: 694f12f3 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + int(round(diff_lb * (hi - lo)))
    b = lo + int(round(diff_ub * (hi - lo)))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if a > b:
        a, b = b, a
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    max_h = max(9, min(30, max_h))
    max_w = max(9, min(30, max_w))
    rotk = random.choice([0, 1, 2, 3])
    if rotk in (1, 3):
        # rotated: generated (h, w) becomes (w, h)
        h = _unifint(diff_lb, diff_ub, 9, max_w)
        w = _unifint(diff_lb, diff_ub, 9, max_h)
    else:
        h = _unifint(diff_lb, diff_ub, 9, max_h)
        w = _unifint(diff_lb, diff_ub, 9, max_w)
    seploc = random.randint(4, h - 5)
    bigh = _unifint(diff_lb, diff_ub, 4, seploc)
    bigw = _unifint(diff_lb, diff_ub, 3, w - 1)
    bigloci = random.randint(0, seploc - bigh)
    biglocj = random.randint(0, w - bigw)
    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    bigsize = bigh * bigw
    cands = []
    for a in range(3, smallmaxh + 1):
        for b in range(3, smallmaxw + 1):
            if a * b < bigsize:
                cands.append((a, b))
    cands = sorted(cands, key=lambda ab: ab[0] * ab[1])
    idx = _unifint(diff_lb, diff_ub, 0, len(cands) - 1)
    smallh, smallw = cands[idx]
    smallloci = random.randint(seploc + 1, h - smallh)
    smalllocj = random.randint(0, w - smallw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[bigloci:bigloci + bigh, biglocj:biglocj + bigw] = sqc
    gi[smallloci:smallloci + smallh, smalllocj:smalllocj + smallw] = sqc
    go = gi.copy()
    go[bigloci + 1:bigloci + bigh - 1, biglocj + 1:biglocj + bigw - 1] = 2
    go[smallloci + 1:smallloci + smallh - 1, smalllocj + 1:smalllocj + smallw - 1] = 1
    gi = np.rot90(gi, k=rotk)
    go = np.rot90(go, k=rotk)
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, color):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == color and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == color:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = sorted(set(I.flatten().tolist()))
    rects = None
    for col in colors:
        comps = _components(I, col)
        ok = True
        boxes = []
        for cells in comps:
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
                ok = False
                break
            boxes.append((len(cells), r0, r1, c0, c1))
        if ok and len(boxes) == 2:
            rects = boxes
            break
    ops, sels = [], []
    rects = sorted(rects)  # small first, big second
    for (size, r0, r1, c0, c1), newc in zip(rects, (1, 2)):
        # interior of the square (inbox backdrop): a full rectangle
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if cells:
            ops.append(newc)
            sels.append(sel_of(cells))
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 694f12f3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 694f12f3"
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
                                f"for task 694f12f3"
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
                    f"Failed to build a complete episode for task 694f12f3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"694f12f3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
