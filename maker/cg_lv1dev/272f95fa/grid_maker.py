"""
ARC Task: 272f95fa (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    # Colours 1, 2, 3, 4 and 6 are reserved for the five painted bands.
    cols = [c for c in range(10) if c not in (1, 2, 3, 4, 6)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c not in (1, 2, 3, 4, 6)]
        bgc, linc = random.sample(cols, 2)
    for _ in range(100):
        h = unifint(diff_lb, diff_ub, (5, max_h))
        w = unifint(diff_lb, diff_ub, (5, max_w))
        c = canvas(bgc, (5, 5))
        l1 = connect((1, 0), (1, 4))
        l2 = connect((3, 0), (3, 4))
        lns = l1 | l2
        gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
        hdist = [0, 0, 0]
        wdist = [0, 0, 0]
        idx = 0
        for k in range(h - 2):
            hdist[idx] += 1
            idx = (idx + 1) % 3
        for k in range(w - 2):
            wdist[idx] += 1
            idx = (idx + 1) % 3
        shuffle(hdist)
        shuffle(wdist)
        hdelt1 = unifint(diff_lb, diff_ub, (0, hdist[0] - 1))
        hdist[0] -= hdelt1
        hdist[1] += hdelt1
        hdelt2 = unifint(diff_lb, diff_ub, (0, min(hdist[1], hdist[2]) - 1))
        hdelt2 = choice((+hdelt2, -hdelt2))
        hdist[1] += hdelt2
        hdist[2] -= hdelt2
        wdelt1 = unifint(diff_lb, diff_ub, (0, wdist[0] - 1))
        wdist[0] -= wdelt1
        wdist[1] += wdelt1
        wdelt2 = unifint(diff_lb, diff_ub, (0, min(wdist[1], wdist[2]) - 1))
        wdelt2 = choice((+wdelt2, -wdelt2))
        wdist[1] += wdelt2
        wdist[2] -= wdelt2
        gi = gi[:1] * hdist[0] + gi[1:2] + gi[2:3] * hdist[1] + gi[3:4] + gi[4:5] * hdist[2]
        gi = dmirror(gi)
        gi = gi[:1] * wdist[0] + gi[1:2] + gi[2:3] * wdist[1] + gi[3:4] + gi[4:5] * wdist[2]
        gi = dmirror(gi)
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        nmfs = choice((1, 2))
        for fn in sample(mfs, nmfs):
            gi = fn(gi)
        # Some of the transforms swap height and width, so check the size again.
        if len(gi) > max_h or len(gi[0]) > max_w:
            continue
        objs = objects(gi, T, T, F)
        bgobjs = colorfilter(objs, bgc)
        cnrs = corners(asindices(gi))
        bgobjs = sfilter(bgobjs, lambda o: len(toindices(o) & cnrs) == 0)
        pinkobj = extract(bgobjs, lambda o: not bordering(o, gi))
        yellobj = argmin(bgobjs, leftmost)
        greenobj = argmax(bgobjs, rightmost)
        redobj = argmin(bgobjs, uppermost)
        blueobj = argmax(bgobjs, lowermost)
        go = fill(gi, 6, pinkobj)
        go = fill(go, 4, yellobj)
        go = fill(go, 3, greenobj)
        go = fill(go, 2, redobj)
        go = fill(go, 1, blueobj)
        return {'input': gi, 'output': go}
    raise ValueError("could not fit grid")


def derive_operations(I, O):
    # The golf reading and the maker's reading agree. Two full lines in each
    # direction split the grid into a 3x3 table of bands. Each cell of the table
    # gets a colour from its (row band, column band) position:
    #   centre -> 6, top-middle -> 2, bottom-middle -> 1,
    #   middle-left -> 4, middle-right -> 3, corners stay as they are.
    # Each painted cell is a full rectangle that can be named from I alone,
    # so a bbox selection is exactly the set of cells meant.
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Only the dividing lines are single-coloured: every other row or column
    # crosses two line cells.
    line_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    r1, r2 = min(line_rows), max(line_rows)
    c1, c2 = min(line_cols), max(line_cols)

    # Row bands: [0, r1), (r1, r2), (r2, hi). Column bands: [0, c1), (c1, c2), (c2, wi).
    row_band = {"top": (0, r1 - 1), "mid": (r1 + 1, r2 - 1), "bot": (r2 + 1, hi - 1)}
    col_band = {"left": (0, c1 - 1), "mid": (c1 + 1, c2 - 1), "right": (c2 + 1, wi - 1)}

    table = [
        (6, "mid", "mid"),    # the enclosed centre cell
        (2, "top", "mid"),
        (1, "bot", "mid"),
        (4, "mid", "left"),
        (3, "mid", "right"),
    ]

    ops, sels = [], []
    for color, rb, cb in table:
        ra, rz = row_band[rb]
        ca, cz = col_band[cb]
        ops.append(color)
        sels.append([ra, ca, rz - ra, cz - ca])  # full rectangle of this table cell

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
                        f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
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
                                f"for task 272f95fa"
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
                    f"Failed to build a complete episode for task 272f95fa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"272f95fa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
