"""
ARC Task: dbc1a6ce (RE-ARC) — LLM-generated grid_maker
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
import random
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator draws bgc and fgcol from interval(0,10) with 8 removed
    # (8 is reserved for the drawn connector lines).
    colopts = [c for c in range(10) if c != 8]
    bgc = random.choice(colopts)
    fgc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc, fgc) -> dict:
    hub = max(3, min(30, int(max_h)))
    wub = max(3, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, wub))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(diff_lb, diff_ub, card_bounds)
    s = sample(inds, num)
    fgcol = fgc
    gi = fill(c, fgcol, s)
    resh = frozenset()
    for x, r in enumerate(gi):
        if r.count(fgcol) > 1:
            resh = combine(resh, connect((x, r.index(fgcol)), (x, -1 + w - r[::-1].index(fgcol))))
    go = fill(c, 8, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgcol) > 1:
            resv = combine(resv, connect((x, r.index(fgcol)), (x, -1 + h - r[::-1].index(fgcol))))
    go = dmirror(fill(dmirror(go), 8, resv))
    go = fill(go, fgcol, s)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: the sparse marker colour (the least frequent colour in I) defines, in every
    row that holds at least two markers, a horizontal LINE from its first marker to its
    last; likewise in every column that holds at least two markers, a vertical LINE.
    Those lines are drawn in the connector colour; the markers themselves stay as they are.

    The account below is grouped by LINE: one operation draws one whole line segment.
    Nothing is replicated, translated or turned here, so no Copy/Paste/Move/Flip appears —
    the rule really is "draw this line", and each op draws exactly one of them.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Connector colour is a convention the input alone does not fix -> read it from the
    # demonstrations (the colour their outputs introduce that their inputs never contain).
    fill_c = None
    if examples:
        for ex in examples:
            if isinstance(ex, dict):
                ei, eo = ex.get("input"), ex.get("output")
            else:
                ei, eo = ex[0], ex[1]
            ea = np.asarray(ei, dtype=int)
            eb = np.asarray(eo, dtype=int)
            newc = set(eb.flatten().tolist()) - set(ea.flatten().tolist())
            if len(newc) == 1:
                fill_c = int(newc.pop())
                break
    if fill_c is None:
        fill_c = 8

    # Marker colour = least frequent colour in I (background covers >= 75% of the canvas).
    cnt = Counter(I.flatten().tolist())
    fgc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    marks = (I == fgc)

    grid = I.copy()
    ops, sels = [], []

    # --- horizontal lines: one op per row that spans two or more markers ---
    for r in range(h):
        cols = np.flatnonzero(marks[r])
        if len(cols) < 2:
            continue
        cells = [(r, c) for c in range(int(cols[0]) + 1, int(cols[-1]))
                 if not marks[r, c] and grid[r, c] != fill_c]
        if not cells:
            continue
        ops.append(int(fill_c))
        sels.append(sel_of(cells))
        for (rr, cc) in cells:
            grid[rr, cc] = fill_c

    # --- vertical lines: one op per column that spans two or more markers ---
    # Cells already carrying the connector colour (crossings with a drawn horizontal
    # line) are left out: painting them again would change nothing there.
    for c in range(w):
        rows = np.flatnonzero(marks[:, c])
        if len(rows) < 2:
            continue
        cells = [(r, c) for r in range(int(rows[0]) + 1, int(rows[-1]))
                 if not marks[r, c] and grid[r, c] != fill_c]
        if not cells:
            continue
        ops.append(int(fill_c))
        sels.append(sel_of(cells))
        for (rr, cc) in cells:
            grid[rr, cc] = fill_c

    # Submit: selection is the whole grid rectangle (a genuine full rectangle).
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
                        f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dbc1a6ce"
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
                                f"for task dbc1a6ce"
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
                    f"Failed to build a complete episode for task dbc1a6ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dbc1a6ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
