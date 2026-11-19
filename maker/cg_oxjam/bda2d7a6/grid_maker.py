"""
ARC Task: bda2d7a6 (RE-ARC) — LLM-generated grid_maker
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
    # Rule is colour-agnostic (every colour, 0 included, is an ordinary band colour).
    # The only discrete structural case is the "merge" quirk: innermost and outermost
    # bands share a colour.  Make sure examples show both cases.
    n_ex = num_examples if num_examples else 3
    if n_ex >= 2:
        flags = [True] + [False] * (n_ex - 1)
        random.shuffle(flags)
    else:
        flags = [False]
    plan = [{"merge": f} for f in flags]
    plan.append(dict(random.choice(plan)))
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, merge=None, **kw) -> dict:
    from dsl import interval, canvas, box, fill
    from random import sample, choice
    from utils import unifint
    if merge is None:
        merge = random.random() < 0.2
    colopts = interval(0, 10, 1)
    hub = max(2, min(14, max_h // 2))
    wub = max(2, min(14, max_w // 2))
    lb = 3 if merge else 2
    for _ in range(1000):
        h = unifint(diff_lb, diff_ub, (min(lb, hub), hub))
        w = unifint(diff_lb, diff_ub, (min(lb, wub), wub))
        ncols = unifint(diff_lb, diff_ub, (2, 10))
        cols = sample(colopts, ncols)
        colord = [choice(cols) for j in range(min(h, w))]
        runs = [colord[0]]
        for c in colord[1:]:
            if c != runs[-1]:
                runs.append(c)
        is_merge = len(runs) > 2 and runs[0] == runs[-1]
        if merge == is_merge and len(runs) >= 2:
            break
    gi = canvas(0, (h * 2, w * 2))
    for idx, ci in enumerate(colord):
        bx = box(frozenset({(idx, idx), (h * 2 - 1 - idx, w * 2 - 1 - idx)}))
        gi = fill(gi, ci, bx)
    I = np.array(gi)
    H, W = I.shape
    rr, cc = np.indices((H, W))
    depth = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))
    bands = []
    for d in range(min(h, w)):
        col = int(I[d, d])
        if bands and bands[-1][0] == col:
            bands[-1][2] = d
        else:
            bands.append([col, d, d])
    n = len(bands)
    merged = n > 2 and bands[0][0] == bands[-1][0]
    O = I.copy()
    for k in range(n):
        col, a, b = bands[k]
        if merged and (k == 0 or k == n - 1):
            new = bands[n - 2][0]
        elif k == 0:
            new = bands[n - 1][0]
        else:
            new = bands[k - 1][0]
        O[(depth >= a) & (depth <= b)] = new
    return {"input": [list(map(int, r)) for r in I],
            "output": [list(map(int, r)) for r in O]}


def derive_operations(I, O=None, examples=None):
    """Reading (from the golf entry): walk the diagonal ray from the corner to the
    centre; the run-length bands of colour along it ARE the nested frames.  The colour
    sequence shifts one band INWARD: every frame takes the colour of the frame just
    outside it, and the outermost frame takes the innermost frame's colour.  A frame
    is selected by its depth range (distance to the nearest border), read from I.
    Painted from the innermost frame outward, following the inward->outward lookup.
    Generator quirk: when outermost and innermost frames share a colour they form one
    unit, which takes the colour of the second-innermost frame."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    rr, cc = np.indices((H, W))
    depth = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))
    maxd = int(depth.max())
    bands = []  # [colour, d_from, d_to] along the diagonal ray, outer -> inner
    for d in range(maxd + 1):
        col = int(I[d, d])
        if bands and bands[-1][0] == col:
            bands[-1][2] = d
        else:
            bands.append([col, d, d])
    n = len(bands)
    merged = n > 2 and bands[0][0] == bands[-1][0]
    targets = []
    for k in range(n):
        if merged and (k == 0 or k == n - 1):
            new = bands[n - 2][0]
        elif k == 0:
            new = bands[n - 1][0]
        else:
            new = bands[k - 1][0]
        targets.append(new)
    ops, sels = [], []
    for k in range(n - 1, -1, -1):  # innermost frame first, moving outward
        col, a, b = bands[k]
        new = targets[k]
        if new == col:
            continue  # frame already holds its new colour: painting would do nothing
        cells = [(int(r), int(c)) for r, c in zip(*np.where((depth >= a) & (depth <= b)))]
        ops.append(int(new))
        sels.append(sel_of(cells))
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
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
                                f"for task bda2d7a6"
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
                    f"Failed to build a complete episode for task bda2d7a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bda2d7a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
