"""
ARC Task: 3618c87e (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors / plan

_ROTS = ("identity", "rot90", "rot180", "rot270")


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_ROTS):
        examples = [{"rot": r} for r in _ROTS]
        examples += [{"rot": random.choice(_ROTS)} for _ in range(n_ex - len(_ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(list(_ROTS), n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


# ---------------------------------------------------------------- generator

def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, linc=None, dotc=None, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None or dotc is None:
        bgc, linc, dotc = sample(cols, 3)
    if rot is None:
        rot = choice(_ROTS)

    # rot90/rot270 swap the dimensions, so sample within the transposed budget
    if rot in ("rot90", "rot270"):
        hmax, wmax = max(4, max_w), max(4, max_h)
    else:
        hmax, wmax = max(4, max_h), max(4, max_w)

    h = unifint(diff_lb, diff_ub, (4, hmax))
    w = unifint(diff_lb, diff_ub, (4, wmax))

    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch - 1, opts)
        opts = remove(ch + 1, opts)

    gi = fill(c, linc, ln)
    go = fill(c, linc, ln)
    for j in locs:
        hh = randint(1, h - 3)
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        go = fill(go, linc, lnx)
        gi = fill(gi, dotc, {(hh + 1, j)})
        go = fill(go, dotc, {(0, j)})

    rotf = {"identity": identity, "rot90": rot90,
            "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


# ---------------------------------------------------------------- operations

def derive_operations(I, O):
    """Each isolated dot slides along its line until it hits the solid border line.

    The border line is the one full-length edge whose colour is not the background;
    every dot sits one cell past the far end of a stub that grows out of that border,
    and in O the dot has travelled the whole stub and landed on the border itself.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the solid non-background edge = the border line the dots travel to
    edges = (('U', I[0, :]), ('D', I[h - 1, :]), ('L', I[:, 0]), ('R', I[:, w - 1]))
    direction, linc = None, None
    for name, arr in edges:
        vals = set(arr.tolist())
        if len(vals) == 1 and arr[0] != bgc:
            direction, linc = name, int(arr[0])
            break

    if direction is None:                      # degenerate: nothing to do
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # remaining colour = the dot colour
    others = [int(v) for v in sorted(set(I.flatten().tolist())) if v not in (bgc, linc)]
    if not others:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    dotc = others[0]
    if len(others) > 1:                        # pick the sparsest colour, just in case
        cnt = Counter(I.flatten().tolist())
        dotc = min(others, key=lambda v: cnt[v])

    step = {'U': (-1, 0, 20), 'D': (1, 0, 21), 'L': (0, -1, 23), 'R': (0, 1, 22)}
    dr, dc, mop = step[direction]

    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]
    # walk the dots along the border line, one stub at a time
    if direction in ('U', 'D'):
        dots.sort(key=lambda p: (p[1], p[0]))
    else:
        dots.sort(key=lambda p: (p[0], p[1]))

    for (r, c) in dots:
        if direction == 'U':
            steps = r
        elif direction == 'D':
            steps = h - 1 - r
        elif direction == 'L':
            steps = c
        else:
            steps = w - 1 - c
        if steps <= 0:
            continue

        if dotc != 0:
            # grab this dot once, then keep sliding it with empty selections so the
            # stub it glides over is restored automatically
            ops.append(mop); sels.append(sel_of([(r, c)]))
            for _ in range(steps - 1):
                ops.append(mop); sels.append(sel_of([]))
            # the dot's original footprint is the only cell left at 0
            if bgc != 0:
                ops.append(bgc); sels.append(sel_of([(r, c)]))
        else:
            # ARCLE's object mode cannot carry a colour-0 cell (0 == "nothing"),
            # so this dot has to be painted at its landing cell and cleared at home
            dest = (r + dr * steps, c + dc * steps)
            ops.append(0); sels.append(sel_of([dest]))
            ops.append(bgc); sels.append(sel_of([(r, c)]))

    # full-grid rectangle for submit
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
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
                                f"for task 3618c87e"
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
                    f"Failed to build a complete episode for task 3618c87e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3618c87e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
