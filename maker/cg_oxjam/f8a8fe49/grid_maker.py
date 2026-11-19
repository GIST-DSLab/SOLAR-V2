"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (10, min(30, lim_h)))
    w = _unifint(diff_lb, diff_ub, (4, min(30, lim_w)))
    fullh = _unifint(diff_lb, diff_ub, (10, h))
    fullw = _unifint(diff_lb, diff_ub, (3, w))
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    canvi = np.full((fullh, fullw), bgc, dtype=int)
    canvo = np.full((fullh, fullw), bgc, dtype=int)
    objh = (fullh // 2 - 3) // 2
    br = [(objh + 1, j) for j in range(fullw)] + [(objh + 2, 0), (objh + 2, fullw - 1)]
    cands = [(i, j) for i in range(objh) for j in range(1, fullw - 1)]
    for _ in range(2):
        for (i, j) in br:
            canvi[i, j] = boxc
            canvo[i, j] = boxc
        tot = objh * (fullw - 2)
        ncellsd = _unifint(diff_lb, diff_ub, (0, tot // 2))
        ncells = random.choice((ncellsd, tot - ncellsd))
        ncells = min(max(1, ncells), tot)
        cells = set(random.sample(cands, ncells))
        cells.add(random.choice([c for c in cands if c[0] == objh - 1]))
        for (i, j) in cells:
            canvi[i, j] = objc
            canvo[objh - 1 - i + objh + 3, j] = objc
        canvi = canvi[::-1].copy()
        canvo = canvo[::-1].copy()
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + fullh, locj:locj + fullw] = canvi
    go[loci:loci + fullh, locj:locj + fullw] = canvo
    if transpose:
        gi, go = gi.T.copy(), go.T.copy()
    # the pattern lies inside the box in the input and outside it in the output
    return {"input": go.tolist(), "output": gi.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Rule: two parallel box lines enclose a strip. The pattern just inside each
    box line is reflected THROUGH that line to the outside, and the inside is
    left empty.

    Comparison with the code-golf reading: the golf entry finds each crossing
    of a box line (a box-coloured cell with a fixed window on either side) and
    reverses that window. Reversing a window centred on the line is the same
    reflection as flipping a band centred on the line. Both readings use the
    same unit (one reflection per box line, through the line itself) and keep
    the same quantity fixed (the line stays put, content swaps sides). The
    readings agree, so the route stays the same: one in-place flip per box
    line. The flip covers the band over the interior columns or rows; the
    box's corner hooks fall outside the band and stay where they are.

    Everything is measured from I alone:
      - the background is the canvas corner, because the area outside the box
        is always empty in the input;
      - the box colour is the colour with the larger bounding box;
      - the band depth is how far the pattern beside that line reaches.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    def area(col):
        b = bbox(col)
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})
    boxc = max(others, key=area)
    objc = [c for c in others if c != boxc][0]
    r0, r1, c0, c1 = bbox(boxc)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))
    rs, cs = np.where(I == objc)
    ops, sels = [], []
    if horizontal:
        mid = (r0 + r1) / 2.0
        top = [int(r) for r in rs if r < mid]
        bot = [int(r) for r in rs if r > mid]
        # full rectangle on purpose: the band of interior columns centred on box row r0
        m = max(top) - r0
        ops.append(27); sels.append([r0 - m, c0 + 1, 2 * m, c1 - c0 - 2])
        # full rectangle on purpose: the band centred on box row r1
        m = r1 - min(bot)
        ops.append(27); sels.append([r1 - m, c0 + 1, 2 * m, c1 - c0 - 2])
    else:
        mid = (c0 + c1) / 2.0
        left = [int(c) for c in cs if c < mid]
        right = [int(c) for c in cs if c > mid]
        # full rectangle on purpose: the band of interior rows centred on box column c0
        m = max(left) - c0
        ops.append(26); sels.append([r0 + 1, c0 - m, r1 - r0 - 2, 2 * m])
        # full rectangle on purpose: the band centred on box column c1
        m = c1 - min(right)
        ops.append(26); sels.append([r0 + 1, c1 - m, r1 - r0 - 2, 2 * m])
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
