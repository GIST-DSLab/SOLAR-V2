"""
ARC Task: 8d510a79 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    barcol = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barcol": barcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, mirror=None) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    # after dmirror the grid is transposed, so bound h/w accordingly
    hb = max_w if mirror else max_h
    wb = max_h if mirror else max_w
    hb = max(5, min(30, hb))
    wb = max(3, min(30, wb))
    h = unifint(diff_lb, diff_ub, (5, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))
    barloci = randint(2, h - 3)
    gi = canvas(bgc, (h, w))
    bar = connect((barloci, 0), (barloci, w - 1))
    gi = fill(gi, barcol, bar)
    go = tuple(e for e in gi)
    jinds = interval(0, w, 1)
    numtop = unifint(diff_lb, diff_ub, (1, w - 1))
    numbot = unifint(diff_lb, diff_ub, (1, w - 1))
    tops = sample(jinds, numtop)
    bots = sample(jinds, numbot)
    for t in tops:
        loci = randint(0, barloci - 2)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (0, t)))
        else:
            go = fill(go, col, connect(loc, (barloci - 1, t)))
    for t in bots:
        loci = randint(barloci + 2, h - 1)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (h - 1, t)))
        else:
            go = fill(go, col, connect(loc, (barloci + 1, t)))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background = dominant color (grid is mostly canvas)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the bar: a full row (or, in the transposed variant, a full column) of one non-bg color
    bar_row, bar_col = None, None
    for r in range(h):
        v = int(I[r, 0])
        if v != bgc and v not in (1, 2) and bool(np.all(I[r, :] == v)):
            bar_row = r
            break
    if bar_row is None:
        for c in range(w):
            v = int(I[0, c])
            if v != bgc and v not in (1, 2) and bool(np.all(I[:, c] == v)):
                bar_col = c
                break

    # each 2-dot grows toward the bar (stopping beside it); each 1-dot grows to the far edge
    dots = [(r, c, int(I[r, c])) for r in range(h) for c in range(w) if int(I[r, c]) in (1, 2)]
    for r, c, col in dots:
        if bar_row is not None:
            if r < bar_row:
                rng = range(r, bar_row) if col == 2 else range(0, r + 1)
            else:
                rng = range(bar_row + 1, r + 1) if col == 2 else range(r, h)
            cells = [(rr, c) for rr in rng]
        else:
            if c < bar_col:
                rng = range(c, bar_col) if col == 2 else range(0, c + 1)
            else:
                rng = range(bar_col + 1, c + 1) if col == 2 else range(c, w)
            cells = [(r, cc) for cc in rng]
        cells = [(rr, cc) for rr, cc in cells if int(I[rr, cc]) != col]
        if cells:
            ops.append(col)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
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
                                f"for task 8d510a79"
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
                    f"Failed to build a complete episode for task 8d510a79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d510a79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
