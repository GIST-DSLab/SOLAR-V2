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

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, boxc, fillc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # a 90 degree turn transposes the canvas, so respect the transposed limits
    if rot in ("rot90", "rot270"):
        lim_h, lim_w = max_w, max_h
    else:
        lim_h, lim_w = max_h, max_w
    lim_h = max(5, lim_h)
    lim_w = max(5, lim_w)

    h = unifint(diff_lb, diff_ub, (5, lim_h))
    w = unifint(diff_lb, diff_ub, (5, lim_w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)

    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh - 1, ow - 2))
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow - 2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(asobject(subg), (loci, locj)))

    go = repeat(fillc, ntofill) + repeat(bgc, 9 - ntofill)
    go = (go[:3], go[3:6][::-1], go[6:])

    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule: the grid holds a 3-sided box.  Its interior is a solid block of `fillc`
    plus `n` empty (background) rows/columns on the open side.  The answer is a
    3x3 tally: the colour `fillc` REPEATED n times along a boustrophedon path
    (row0 left->right, row1 right->left, row2 left->right), the remaining cells
    background.

    Route: copy one cell of the replicated colour out of the input, reframe the
    canvas to the 3x3 answer, lay the base colour, then paste the replicated
    colour once per unit of the count, walking the snake path.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- identify the three colour roles by bounding-box area -----------------
    # background spans the whole grid, the frame spans the box, the fill is inside
    info = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        info.append((int((r1 - r0 + 1) * (c1 - c0 + 1)), int(col),
                     (int(r0), int(c0), int(r1), int(c1))))
    info.sort(key=lambda t: -t[0])
    bgc = info[0][1]
    boxc, bbox = info[1][1], info[1][2]
    fillc, fbox = info[2][1], info[2][2]

    # --- which axis is the box closed along? (both side columns solid frame?) --
    r0, c0, r1, c1 = bbox
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    if left_full and right_full:
        box_dim = r1 - r0 + 1
        fill_dim = fbox[2] - fbox[0] + 1
    else:
        box_dim = c1 - c0 + 1
        fill_dim = fbox[3] - fbox[1] + 1
    n = box_dim - fill_dim - 1
    n = max(0, min(9, n))

    # --- boustrophedon path of the 3x3 answer --------------------------------
    path = []
    for k in range(9):
        r, m = divmod(k, 3)
        c = m if r != 1 else 2 - m
        path.append((r, c))

    # Paste cannot carry colour 0, so the colour that is 0 (if any) becomes the
    # painted base and the other colour is the one that gets replicated.
    if fillc != 0:
        base_col, rep_col = bgc, fillc
        rep_cells = path[:n]
        src_r, src_c = fbox[0], fbox[1]          # solid fill rectangle
    else:
        base_col, rep_col = fillc, bgc
        rep_cells = path[n:]
        bg_cells = np.argwhere(I == bgc)
        src_r, src_c = int(bg_cells[0][0]), int(bg_cells[0][1])

    ops, sels = [], []

    # 1. take the replicated colour from the object it comes from in the input
    if rep_cells:
        ops.append(28); sels.append(sel_of([(src_r, src_c)]))

    # 2. reframe the canvas to the 3x3 answer grid (full rectangle -> bbox ok)
    out_cells = [(r, c) for r in range(3) for c in range(3)]
    ops.append(33); sels.append(sel_of(out_cells))

    # 3. lay the base colour over the whole answer grid (skip if already so)
    cur = I[0:3, 0:3]
    if not np.all(cur == base_col):
        ops.append(int(base_col)); sels.append(sel_of(out_cells))

    # 4. repeat the other colour once per counted unit, along the snake path
    for (r, c) in rep_cells:
        ops.append(30); sels.append(sel_of([(r, c)]))

    ops.append(34); sels.append(sel_of(out_cells))
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
