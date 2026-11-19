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

# The code-golf entry reads this task the same way the maker does. It measures the
# wall length, subtracts the height of the filled block and two cells for the walls,
# and what is left is the number of empty rows on the open side of the box. That
# count says how many tally cells of the fill colour to lay out. Both readings treat
# the box, the fill block and the count as the same units, so the route is unchanged:
# take the colour from the input, reframe to 3x3, lay the base colour, then paste
# once per counted unit along the snake path.


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, fillc = random.sample(range(10), 3)
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxc=None, fillc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    if bgc is None or boxc is None or fillc is None:
        bgc, boxc, fillc = random.sample(range(10), 3)
    lim = max(5, min(max_h, max_w, 30))
    h = unifint(diff_lb, diff_ub, (5, lim))
    w = unifint(diff_lb, diff_ub, (5, lim))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    subg = np.full((oh, ow), boxc, dtype=int)
    subg2 = np.full((oh - 1, ow - 2), fillc, dtype=int)
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        subg2[j, :] = bgc
    subg[0:oh - 1, 1:ow - 1] = subg2
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = subg
    go = [fillc] * ntofill + [bgc] * (9 - ntofill)
    go = [go[:3], go[3:6][::-1], go[6:]]
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    # colour roles by bounding-box area: background > frame > fill
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

    # axis along which the box is closed: both side columns are solid frame
    r0, c0, r1, c1 = bbox
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    if left_full and right_full:
        box_dim = r1 - r0 + 1
        fill_dim = fbox[2] - fbox[0] + 1
    else:
        box_dim = c1 - c0 + 1
        fill_dim = fbox[3] - fbox[1] + 1
    n = max(0, min(9, box_dim - fill_dim - 1))   # empty rows on the open side

    # boustrophedon path of the 3x3 tally
    path = []
    for k in range(9):
        r, m = divmod(k, 3)
        path.append((r, m if r != 1 else 2 - m))

    # Paste cannot carry colour 0: the colour that is 0 becomes the base
    if fillc != 0:
        base_col, rep_col = bgc, fillc
        rep_cells = path[:n]
        src = (fbox[0], fbox[1])
    else:
        base_col, rep_col = fillc, bgc
        rep_cells = path[n:]
        bgs = np.argwhere(I == bgc)
        src = (int(bgs[0][0]), int(bgs[0][1]))

    ops, sels = [], []
    if rep_cells:
        ops.append(28); sels.append(sel_of([src]))           # take the tally colour

    out_cells = [(r, c) for r in range(3) for c in range(3)]
    ops.append(33); sels.append(sel_of(out_cells))           # reframe to 3x3 (full rect)

    cur = I[0:3, 0:3]                                        # transparent crop keeps these
    if not np.all(cur == base_col):
        ops.append(int(base_col)); sels.append(sel_of(out_cells))

    for (r, c) in rep_cells:                                 # one paste per counted unit
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
