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
import numpy as np
from random import randint, choice, sample as rsample

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, fillc = rsample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        from random import shuffle
        shuffle(examples)
    else:
        examples = [{"rot": r} for r in rsample(ROTS, n_ex)]
    plan = examples + [dict(choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc, rot=None) -> dict:
    if rot is None:
        rot = choice(ROTS)
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
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
    if rot == "rot90":
        gi = rot90(gi)
    elif rot == "rot180":
        gi = rot180(gi)
    elif rot == "rot270":
        gi = rot270(gi)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- read the three color roles out of I by bounding-box extent ---
    def bb(col):
        rs, cs = np.nonzero(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        return ((r1 - r0 + 1) * (c1 - c0 + 1), r0, c0, r1, c1)

    present = [int(c) for c in np.unique(I)]
    info = {c: bb(c) for c in present}
    bgc = max(present, key=lambda c: info[c][0])          # spans the whole canvas
    rest = [c for c in present if c != bgc]
    boxc = max(rest, key=lambda c: info[c][0])            # the frame
    fillc = min(rest, key=lambda c: info[c][0])           # the filled block inside it

    _, br0, bc0, br1, bc1 = info[boxc]
    _, fr0, fc0, fr1, fc1 = info[fillc]

    # frame's two long sides tell us which axis the stripes run along
    side_cols_full = (all(I[r, bc0] == boxc for r in range(br0, br1 + 1)) and
                      all(I[r, bc1] == boxc for r in range(br0, br1 + 1)))
    if side_cols_full:
        n = (br1 - br0 + 1) - (fr1 - fr0 + 1) - 1        # empty rows inside the frame
    else:
        n = (bc1 - bc0 + 1) - (fc1 - fc0 + 1) - 1        # empty cols inside the frame
    n = max(0, min(9, int(n)))

    ops, sels = [], []

    # --- take a 3x3 patch of background from I as the canvas ---
    anchor = None
    for r in range(hi - 2):
        for c in range(wi - 2):
            if np.all(I[r:r + 3, c:c + 3] == bgc):
                anchor = (r, c)
                break
        if anchor:
            break
    if anchor is None:
        anchor = (0, 0)
    ops.append(33); sels.append([anchor[0], anchor[1], 2, 2])
    if not np.all(I[anchor[0]:anchor[0] + 3, anchor[1]:anchor[1] + 3] == bgc):
        ops.append(bgc); sels.append([0, 0, 2, 2])

    # --- stamp n fill cells along the snake path: row0 L->R, row1 R->L, row2 L->R ---
    k0 = min(n, 3)
    if k0 > 0:
        ops.append(fillc); sels.append([0, 0, 0, k0 - 1])
    k1 = min(max(n - 3, 0), 3)
    if k1 > 0:
        ops.append(fillc); sels.append([1, 3 - k1, 0, k1 - 1])
    k2 = min(max(n - 6, 0), 3)
    if k2 > 0:
        ops.append(fillc); sels.append([2, 0, 0, k2 - 1])

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
