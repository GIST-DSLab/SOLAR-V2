"""
ARC Task: 178fcbfb (RE-ARC) — LLM-generated grid_maker
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
    # The generator hardcodes 1, 2 and 3 as the marker colours; only the canvas
    # colour is sampled (from 0..9 minus {1,2,3}) -- 0 included, it is an
    # ordinary picture colour here.
    bgc = random.choice([0, 4, 5, 6, 7, 8, 9])
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    iforb = set()
    jforb = set()
    for col in (2, 1, 3):
        bnd = unifint(diff_lb, diff_ub, (1, w if col == 2 else max(1, h // 2)))
        for ndots in range(bnd):
            if col == 2:
                cands = sfilter(inds, lambda ij: last(ij) not in jforb)
                if len(cands) == 0:
                    break
                ij = choice(totuple(cands))
                jforb.add(last(ij))
            else:
                cands = sfilter(inds, lambda ij: first(ij) not in iforb)
                if len(cands) == 0:
                    break
                ij = choice(totuple(cands))
                iforb.add(first(ij))
            gi = fill(gi, col, initset(ij))
            go = fill(go, col, (vfrontier if col == 2 else hfrontier)(ij))
            inds = remove(ij, inds)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off the examples): every 2 marker grows into its full column,
    every 3 and every 1 marker grows into its full row, and the rows cover the
    columns where they cross.  Only I is consulted; O is never read.

    The *route* through that rule is chosen from the input itself (which marker
    the eye meets first, where it sits, how many of each kind there are), so a
    given instance always yields the same trajectory while different instances
    are solved in visibly different orders / styles.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def dots(col):
        return [(int(r), int(c)) for r, c in zip(*np.where(I == col))]

    d2, d3, d1 = dots(2), dots(3), dots(1)
    cols2 = sorted({c for _, c in d2})
    set1 = {r for r, _ in d1}
    set3 = {r for r, _ in d3}
    rows1, rows3 = sorted(set1), sorted(set3)
    line_rows = set1 | set3
    free_rows = [r for r in range(hi) if r not in line_rows]

    all_dots = sorted(d2 + d3 + d1)
    if not all_dots:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle
        return ops, sels

    # --- route choices, all read out of the input -------------------------
    ar, ac = all_dots[0]                      # first marker in reading order
    anchor_col = int(I[ar, ac])
    # start with whichever kind of marker the eye meets first; if the rows
    # already cover every line of the grid the columns leave no trace at all,
    # so the rows are drawn first and the columns only in what is left.
    rows_first = (anchor_col != 2) or (len(free_rows) == 0)
    col_dir = 1 if ac * 2 < wi else -1        # sweep from the anchor's side
    row_dir = 1 if ar * 2 < hi else -1
    mixed_rows = ((len(rows1) + len(rows3)) % 2 == 1)

    def emit(lines, color, style):
        """lines: [(cells, paste_origin)] already in handling order."""
        lines = [ln for ln in lines if ln[0]]
        if not lines:
            return
        if style == "grouped" and len(lines) >= 2:
            union = set()
            for cells, _ in lines:
                union.update(cells)
            ops.append(color)
            sels.append(sel_of(sorted(union)))
            return
        if style == "stamp" and len(lines) >= 2:
            cells0, _ = lines[0]
            ops.append(color)
            sels.append(sel_of(cells0))       # draw the first line by hand
            # every cell of that line now holds `color` (1/2/3, never 0), so
            # the clipboard carries the whole line -- safe even when bgc == 0.
            ops.append(29)
            sels.append(sel_of(cells0))       # CopyO: the line I just made
            for cells, origin in lines[1:]:
                ops.append(30)
                sels.append(sel_of([origin]))  # stamp it at the next line
            return
        for cells, _ in lines:
            ops.append(color)
            sels.append(sel_of(cells))

    def col_lines(masked):
        order = cols2 if col_dir == 1 else list(reversed(cols2))
        out = []
        for c in order:
            if masked:
                cells = [(r, c) for r in free_rows]
                # skip a column whose remaining cells already read 2
                if cells and all(I[r, c] == 2 for r, c in cells):
                    continue
                origin = (cells[0] if cells else None)
            else:
                cells = [(r, c) for r in range(hi)]
                origin = (0, c)
            if cells:
                out.append((cells, origin))
        return out

    def row_line(r):
        return ([(r, c) for c in range(wi)], (r, 0))

    rows_sorted = sorted(line_rows, reverse=(row_dir == -1))

    def emit_rows():
        if mixed_rows:
            # one sweep down (or up) the grid, each row in its own colour
            for r in rows_sorted:
                cells, _ = row_line(r)
                ops.append(1 if r in set1 else 3)
                sels.append(sel_of(cells))
            return
        # otherwise finish one colour before starting the other, bigger first
        if len(rows1) != len(rows3):
            first_c = 1 if len(rows1) > len(rows3) else 3
        else:
            first_c = 1 if (rows_sorted and rows_sorted[0] in set1) else 3
        for cval in (first_c, 3 if first_c == 1 else 1):
            members = set1 if cval == 1 else set3
            seq = [r for r in rows_sorted if r in members]
            lines = [row_line(r) for r in seq]
            style = ("individual", "grouped", "stamp")[len(seq) % 3] if len(seq) >= 2 else "individual"
            emit(lines, cval, style)

    if rows_first:
        cstyle = ("individual", "grouped")[len(cols2) % 2] if len(cols2) >= 2 else "individual"
        emit_rows()
        if free_rows:
            emit(col_lines(True), 2, cstyle)
    else:
        cstyle = ("individual", "grouped", "stamp")[len(cols2) % 3] if len(cols2) >= 2 else "individual"
        emit(col_lines(False), 2, cstyle)
        emit_rows()

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])       # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 178fcbfb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 178fcbfb"
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
                                f"for task 178fcbfb"
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
                    f"Failed to build a complete episode for task 178fcbfb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"178fcbfb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
