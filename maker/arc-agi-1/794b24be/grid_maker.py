"""
ARC Task: 794b24be (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

# Discrete structural variant: number of colour-1 cells in the input (1..4)
VARIANTS = [{"nblue": 1}, {"nblue": 2}, {"nblue": 3}, {"nblue": 4}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, nblue=None) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1)}
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    if nblue is None:
        nblue = randint(1, 4)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 2, {mpr[k + 1]})
    gi = canvas(bgc, (h, w))
    locs = sample(totuple(asindices(gi)), nblue)
    gi = fill(gi, 1, locs)
    remlocs = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (0, max(0, len(remlocs) // 2 - 1)))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    noise = sample(totuple(remlocs), namt)
    noise = {(choice(ccols), ij) for ij in noise}
    gi = paint(gi, noise)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    # --- rule inference: everything measured from I ---
    # signal = number of colour-1 cells
    n = int((I == 1).sum())
    # background = most frequent colour among all colours except 1 (argmax colorcount)
    cnt = Counter(v for v in I.flatten().tolist() if v != 1)
    bgc = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # answer geometry: horizontal bar of 2 from origin, length n (clipped to 3),
    # plus centre cell (1,1) when n == 4
    bar = [(0, c) for c in range(min(n, 3))]
    extra = [(1, 1)] if n == 4 else []
    two_cells = set(bar) | set(extra)

    ops, sels = [], []

    # 1. fix the workspace: shrink canvas to the 3x3 answer board at the top-left.
    #    full rectangle -> bbox selection is exactly the intended cells.
    ops.append(33); sels.append([0, 0, 2, 2])

    # after the crop the working grid holds I[0:3, 0:3]
    cur = {(r, c): int(I[r, c]) for r in range(3) for c in range(3)}

    # 2. clear the board to the measured background (only cells not already bgc,
    #    and not part of the bar we are about to draw)
    bg_cells = sorted(rc for rc in cur
                      if rc not in two_cells and cur[rc] != bgc)
    if bg_cells:
        ops.append(bgc); sels.append(sel_of(bg_cells))

    # 3. draw the bar (one op = one line object)
    need_bar = [rc for rc in bar if cur[rc] != 2]
    if need_bar:
        ops.append(2); sels.append(sel_of(sorted(need_bar)))

    # 4. the extra centre pixel for n == 4
    need_extra = [rc for rc in extra if cur[rc] != 2]
    if need_extra:
        ops.append(2); sels.append(sel_of(sorted(need_extra)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 794b24be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 794b24be"
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
                                f"for task 794b24be"
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
                    f"Failed to build a complete episode for task 794b24be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"794b24be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
