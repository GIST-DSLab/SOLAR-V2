"""
ARC Task: d037b0a7 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule ("every dot shoots a ray straight down to the bottom") depends only on
    # the presence of the dots, not on which colour each one is.  So only the canvas
    # colour has to be shared across the episode; dot colours may stay free.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(diff_lb, diff_ub, (1, w))
    locs = sample(interval(0, w, 1), nlocs)
    for j in locs:
        col = choice(remcols)
        loci = randint(0, h - 1)
        loc = (loci, j)
        gi = fill(gi, col, {loc})
        go = fill(go, col, connect(loc, (h - 1, j)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Every dot on the canvas is drawn out into a vertical line running from the dot
    down to the last row.  The rule is fixed; what varies between instances is the
    route a solver takes through it, and both choices below are read off the INPUT:

      * sweep order  - if the dots are spread over more rows than columns, work
                       top-down (longest line first); otherwise sweep across the
                       columns, starting from the side the topmost dot sits on.
      * grouping     - if a colour appears on more than one dot, that colour's lines
                       are drawn together in one stroke; if every dot has its own
                       colour, each line is drawn on its own.

    No operation is emitted for a dot already sitting on the last row (its line is
    the dot itself, so drawing it would change nothing), and colour 0 lines are laid
    down with an explicit Color0.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = the colour the canvas is filled with; dots are sparse specks on it.
    # Every grid in the episode is drawn on the same canvas colour, so let the
    # demonstrations confirm the majority colour of I.
    votes = Counter()
    votes[Counter(I.flatten().tolist()).most_common(1)[0][0]] += 1
    for pair in (examples or []):
        ex_in = np.asarray(pair[0], dtype=int)
        votes[Counter(ex_in.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]

    dots = [(r, c, int(I[r, c]))
            for c in range(wi) for r in range(hi) if int(I[r, c]) != bgc]

    if dots:
        rows = [d[0] for d in dots]
        colsx = [d[1] for d in dots]
        rspread = max(rows) - min(rows)
        cspread = max(colsx) - min(colsx)
        topdot = min(dots, key=lambda d: (d[0], d[1]))

        if rspread > cspread:
            # tall scatter -> take them from the top down, longest ray first
            keyf = lambda d: (d[0], d[1])
        elif topdot[1] * 2 <= wi - 1:
            # the eye-catching top dot is on the left -> sweep left to right
            keyf = lambda d: (d[1], d[0])
        else:
            # ... on the right -> sweep right to left
            keyf = lambda d: (-d[1], d[0])

        ordered = sorted(dots, key=keyf)
        counts = Counter(d[2] for d in dots)
        group_by_colour = any(v >= 2 for v in counts.values())

        if group_by_colour:
            colour_order = []
            for d in ordered:
                if d[2] not in colour_order:
                    colour_order.append(d[2])
            for col in colour_order:
                members = [d for d in ordered if d[2] == col]
                if all(d[0] == hi - 1 for d in members):
                    continue  # every dot of this colour already rests on the last row
                cells = [(rr, c) for (r, c, _) in members for rr in range(r, hi)]
                ops.append(col)
                sels.append(sel_of(cells))
        else:
            for (r, c, col) in ordered:
                if r == hi - 1:
                    continue  # the dot is already at the bottom; there is no line to draw
                ops.append(col)
                sels.append(sel_of([(rr, c) for rr in range(r, hi)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task d037b0a7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d037b0a7"
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
                                f"for task d037b0a7"
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
                    f"Failed to build a complete episode for task d037b0a7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d037b0a7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
