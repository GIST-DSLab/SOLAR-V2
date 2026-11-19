"""
ARC Task: b0c4d837 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- variants
# Structural parameter of this task: ntofill (= how many of the 9 output cells
# get the fill colour) and the rotation applied to the input canvas.
# ntofill is restricted to {4, 5}: those are exactly the counts for which the
# middle output row is non-uniform, i.e. the vmirror of the middle row (the
# rule the verifier states) is a genuinely visible operation.
VARIANTS = [
    {"ntofill": 4, "rot": 0},
    {"ntofill": 5, "rot": 1},
    {"ntofill": 4, "rot": 2},
    {"ntofill": 5, "rot": 3},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, boxc, fillc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    lo = max(a, lo)
    hi = min(b, hi)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc,
             ntofill=None, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    if ntofill is None:
        ntofill = random.choice([4, 5])

    # after a 90/270 rotation the canvas is transposed -> swap the limits
    if rot in (0, 2):
        h_lim, w_lim = max_h, max_w
    else:
        h_lim, w_lim = max_w, max_h
    h_lim = max(5, min(30, int(h_lim)))
    w_lim = max(5, min(30, int(w_lim)))

    # feasibility: box height oh needs oh >= ntofill + 2 and oh <= h - 1
    ntofill = max(1, min(int(ntofill), 9, h_lim - 3))
    h_lo = max(5, ntofill + 3)
    h = _unifint(diff_lb, diff_ub, (h_lo, h_lim))
    w = _unifint(diff_lb, diff_ub, (5, w_lim))

    oh = _unifint(diff_lb, diff_ub, (ntofill + 2, h - 1))
    ow = _unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)

    g = [[bgc] * w for _ in range(h)]
    # solid box of boxc
    for r in range(oh):
        for c in range(ow):
            g[loci + r][locj + c] = boxc
    # interior stripes: first `ntofill` rows bgc, the rest fillc
    for r in range(oh - 1):
        v = bgc if r < ntofill else fillc
        for c in range(1, ow - 1):
            g[loci + r][locj + c] = v

    gi = np.array(g, dtype=int)
    if rot:
        gi = np.rot90(gi, k=rot)

    seq = [fillc] * ntofill + [bgc] * (9 - ntofill)
    go = [list(seq[0:3]), list(seq[3:6][::-1]), list(seq[6:9])]

    return {"input": gi.tolist(), "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    # ---- identify the three colour roles the way the task builds them -------
    cells = {}
    for c in np.unique(I):
        rs, cs = np.where(I == c)
        cells[int(c)] = (rs, cs)

    def bbox_area(c):
        rs, cs = cells[c]
        return (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)

    ordered = sorted(cells.keys(), key=lambda c: (-bbox_area(c), -len(cells[c][0])))
    bgc = ordered[0]                      # background spans the whole canvas
    boxc = ordered[1]                     # the box frame (bigger bbox)
    fillc = ordered[2]                    # the filled block inside the box

    # ---- how many stripes of background sit inside the box -----------------
    rs, cs = cells[boxc]
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    boxset = set(zip(rs.tolist(), cs.tolist()))
    left_full = all((r, c0) in boxset for r in range(r0, r1 + 1))
    right_full = all((r, c1) in boxset for r in range(r0, r1 + 1))
    frs, fcs = cells[fillc]
    if left_full and right_full:          # box stands upright -> measure heights
        box_dim = r1 - r0 + 1
        fill_dim = int(frs.max() - frs.min() + 1)
    else:                                 # box is on its side -> measure widths
        box_dim = c1 - c0 + 1
        fill_dim = int(fcs.max() - fcs.min() + 1)
    k = int(box_dim - fill_dim - 1)
    k = max(0, min(9, k))

    ops, sels = [], []

    # 1. the answer lives on a 3x3 canvas -> shrink the canvas to 3x3.
    #    bbox is exactly the full 3x3 rectangle we keep.
    ops.append(33); sels.append([0, 0, 2, 2])

    # 2. lay the background base over the whole little canvas
    #    (skip only when the canvas already holds nothing but bgc)
    if not np.all(I[:3, :3] == bgc):
        ops.append(int(bgc))
        sels.append(sel_of([(r, c) for r in range(3) for c in range(3)]))

    # 3. write k fill-coloured cells in reading order, row by row
    seq = [fillc] * k + [bgc] * (9 - k)
    for row in range(3):
        seg = [(row, c) for c in range(3) if seq[row * 3 + c] == fillc]
        if seg:
            ops.append(int(fillc)); sels.append(sel_of(seg))

    # 4. the rule's reflection: mirror the middle row left<->right.
    #    bbox = exactly the whole middle row, background included.
    mid = seq[3:6]
    if mid != mid[::-1]:
        ops.append(26); sels.append([1, 0, 0, 2])

    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
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
                                f"for task b0c4d837"
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
                    f"Failed to build a complete episode for task b0c4d837 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b0c4d837-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
