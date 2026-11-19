"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, random.randint(2, 5))

    # numc (how many marked cells) decides how many of the 3 answer rows are
    # filled and whether the reversed middle row is visible at all -> plan it.
    n_ex = num_examples if num_examples else 3
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    vals = [random.choice(g) for g in groups]          # one per row-band
    if n_ex >= 3:
        examples = [{"numc": v} for v in vals]
        examples += [{"numc": random.randint(1, 9)} for _ in range(n_ex - 3)]
    else:
        picks = [vals[2]] + random.sample(vals[:2], max(0, n_ex - 1))
        examples = [{"numc": v} for v in picks[:max(1, n_ex)]]
    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None, numc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    remcols = [c for c in cols if c != bgc]
    if ccols is None:
        numcols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numcols)
    ccols = [c for c in ccols if c != bgc]
    if not ccols:
        ccols = [choice(remcols)]

    max_h = max(3, int(max_h))
    max_w = max(3, int(max_w))

    h = unifint(diff_lb, diff_ub, (3, max_h))
    if numc is None:
        w = unifint(diff_lb, diff_ub, (3, max_w))
        numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    else:
        numc = max(1, min(9, int(numc)))
        wlb = max(3, min(numc, max_w))
        w = unifint(diff_lb, diff_ub, (wlb, max_w))
        numc = min(numc, w)

    inds = list(range(w))
    locs = sorted(sample(inds, numc))

    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour the marks sit on (marks are always sparse:
    # at most w cells in an h*w grid with h >= 3, so bgc is the majority colour)
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # read the marked cells left to right
    marks = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    marks.sort(key=lambda rc: rc[1])
    colors = [int(I[r, c]) for r, c in marks][:9]
    colors = colors + [bgc] * (9 - len(colors))       # leftovers are background

    # answer-grid slots, in reading order of the marks:
    # row 0 left->right, row 1 right->left (the middle row runs the other way),
    # row 2 left->right
    slots = [(0, 0), (0, 1), (0, 2),
             (1, 2), (1, 1), (1, 0),
             (2, 0), (2, 1), (2, 2)]

    # pick the 3x3 patch of canvas to draw the answer on: prefer one that holds
    # no marks, so nothing we still have to read gets covered up
    best = None
    for r0 in range(hi - 2):
        for c0 in range(wi - 2):
            cnt = sum(1 for (r, c) in marks if r0 <= r < r0 + 3 and c0 <= c < c0 + 3)
            if best is None or cnt < best[0]:
                best = (cnt, r0, c0)
            if cnt == 0:
                break
        if best[0] == 0:
            break
    _, r0, c0 = best

    ops, sels = [], []
    region = [r0, c0, 2, 2]   # bbox == exactly the full 3x3 answer patch

    # lay the background base over the whole answer patch (only when something
    # there isn't background already, otherwise the op would do nothing)
    if any(int(I[r0 + dr, c0 + dc]) != bgc for dr in range(3) for dc in range(3)):
        ops.append(bgc)
        sels.append(region)

    # lay the mark colours into the patch, in the order the marks were read
    for k in range(9):
        col = colors[k]
        if col == bgc:
            continue                      # this slot already holds background
        dr, dc = slots[k]
        ops.append(col)
        sels.append(sel_of([(r0 + dr, c0 + dc)]))

    # keep only the answer patch as the grid
    if not (hi == 3 and wi == 3 and r0 == 0 and c0 == 0):
        ops.append(33)
        sels.append(region)

    ops.append(34)
    sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
