"""
ARC Task: 23581191 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
# The rule is: every row/column holding a dot of colour A becomes an A-line,
# every row/column holding a dot of colour B becomes a B-line, the background
# cells where an A-line and a B-line cross become 2, and the dots stay put.
# Nothing structural varies between instances (only grid size and dot density),
# so no instance_plan is needed -- but bgc / acol / bcol must be fixed for the
# whole episode so the colour roles are learnable from the demonstrations.
def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 2]      # 2 is reserved for crossings
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, acol: int, bcol: int) -> dict:
    hb = (min(3, max_h), max_h)
    wb = (min(3, max_w), max_w)
    h = unifint(diff_lb, diff_ub, hb)
    w = unifint(diff_lb, diff_ub, wb)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (1, max(1, (h * w) // 4))
    na = unifint(diff_lb, diff_ub, card_bounds)
    nb = unifint(diff_lb, diff_ub, card_bounds)
    na = max(1, min(na, len(inds) - 1))
    nb = max(1, min(nb, len(inds) - na))
    a = sample(inds, na)
    b = sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)
    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ---
def derive_operations(I, O):
    """Build the answer LINE BY LINE, never cell by cell.

    Everything below is read off I alone (O is never consulted):
      rows1/cols1 = the rows and columns that hold a dot of colour c1
      rows2/cols2 = the rows and columns that hold a dot of colour c2
    A background cell is c1-covered if its row is in rows1 or its column is in
    cols1 (same for c2).  Covered by both -> 2, by one -> that colour, by
    neither -> stays background.  The dots themselves are never repainted, so
    every selection is a *line through the background*, i.e. the whole row /
    column minus the dots sitting on it.

    Order of work (each op's paint survives into the submitted grid -- nothing
    is drawn only to be buried):
      1. the lines that carry BOTH colours: their whole background is a
         crossing, so it turns 2 (one op per such row, then per such column);
      2. the remaining crossings: a single-colour row meeting the other
         colour's columns (one op per such row);
      3. each c1-line fills the background it has left with c1;
      4. each c2-line fills the background it has left with c2.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = the colour the canvas was painted with before the dots were
    # dropped on it; each dot colour covers at most a quarter of the cells, so
    # the background is always the strict majority.
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    dot_cols = sorted([c for c in cnt if c != bgc], key=lambda c: (-cnt[c], c))

    ops, sels = [], []
    grid = I.copy()          # working grid, kept in step with the ops emitted

    def draw(color, cells):
        # a line only fills what is still background: dots are left alone and
        # cells an earlier line already claimed keep what they were given.
        cells = [(r, c) for (r, c) in cells if grid[r, c] == bgc]
        if not cells:
            return                      # nothing visible would change -> no op
        for (r, c) in cells:
            grid[r, c] = color
        ops.append(int(color))
        sels.append(sel_of(cells))      # exact cells of this line, not a bbox

    if len(dot_cols) >= 2:
        c1, c2 = dot_cols[0], dot_cols[1]
        p1 = np.argwhere(I == c1)
        p2 = np.argwhere(I == c2)
        rows1, cols1 = set(p1[:, 0].tolist()), set(p1[:, 1].tolist())
        rows2, cols2 = set(p2[:, 0].tolist()), set(p2[:, 1].tolist())

        # 1. lines that are an A-line and a B-line at once -> wholly crossings
        for r in sorted(rows1 & rows2):
            draw(2, [(r, c) for c in range(w)])
        for c in sorted(cols1 & cols2):
            draw(2, [(r, c) for r in range(h)])

        # 2. a one-colour row crossing the other colour's columns
        for r in sorted(rows1 - rows2):
            draw(2, [(r, c) for c in sorted(cols2)])
        for r in sorted(rows2 - rows1):
            draw(2, [(r, c) for c in sorted(cols1)])

        # 3. the c1 line system fills its leftover background
        for r in sorted(rows1):
            draw(c1, [(r, c) for c in range(w)])
        for c in sorted(cols1):
            draw(c1, [(r, c) for r in range(h)])

        # 4. the c2 line system fills its leftover background
        for r in sorted(rows2):
            draw(c2, [(r, c) for c in range(w)])
        for c in sorted(cols2):
            draw(c2, [(r, c) for r in range(h)])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # full-grid bbox: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 23581191"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23581191"
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
                                f"for task 23581191"
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
                    f"Failed to build a complete episode for task 23581191 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23581191-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
