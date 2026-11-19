"""
ARC Task: 834ec97d (RE-ARC) — LLM-generated grid_maker
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
    # The generator samples exactly two colours (bgc, fgc) from 0..9 minus 4;
    # 4 is hardcoded as the beam colour, so it is NOT a sampled role.
    cols = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgc=None) -> dict:
    cols = remove(4, interval(0, 10, 1))
    if bgc is None or fgc is None:
        bgc, fgc = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    loci = unifint(diff_lb, diff_ub, (0, h - 2))
    locjd = unifint(diff_lb, diff_ub, (0, w // 2))
    locj = choice((locjd, w - locjd))
    locj = min(max(0, locj), w - 1)
    loc = (loci, locj)
    c = canvas(bgc, (h, w))
    gi = fill(c, fgc, {loc})
    go = fill(c, fgc, {add(loc, (1, 0))})
    for jj in range(w // 2 + 1):
        go = fill(go, 4, connect((0, locj + 2 * jj), (loci, locj + 2 * jj)))
        go = fill(go, 4, connect((0, locj - 2 * jj), (loci, locj - 2 * jj)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the worked examples, never off O):
      * the single dot drops exactly one row;
      * from the dot's ORIGINAL row upwards, a beam is drawn in every second
        column measured from the dot's column, in both directions, spanning
        rows 0..loci.  The beam covers the dot's old cell, so the vacated cell
        needs no separate repair.

    The ROUTE is chosen from the instance itself (dot position + grid shape),
    so the same instance always yields the same trajectory while different
    instances get genuinely different — but equally valid — routes:
      * which end to sweep from (outward from the dot / left-to-right / right-to-left)
      * whether to drop the dot first or comb the grid first and drop it at the end
      * whether to stroke every beam, stroke the whole comb in one go, or draw
        one beam and copy it onto the others.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # --- background: the generator paints a uniform canvas and puts ONE dot on
    #     it, so the majority colour is the canvas colour (reliable here).
    counts = np.bincount(I.reshape(-1), minlength=10)
    bgc = int(counts.argmax())

    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    loci, locj = dots[0]
    fgc = int(I[loci, locj])

    # --- beam colour: the colour the demonstrations ADD (present in their
    #     outputs, absent from their inputs).  Falls back to the task's 4.
    beam = 4
    if examples:
        extra = set()
        for ex in examples:
            if isinstance(ex, dict):
                ei, eo = ex.get('input'), ex.get('output')
            else:
                ei, eo = ex[0], ex[1]
            ei = np.asarray(ei)
            eo = np.asarray(eo)
            extra |= set(np.unique(eo).tolist()) - set(np.unique(ei).tolist())
        if len(extra) == 1:
            beam = int(extra.pop())

    # --- the comb: every second column measured from the dot's column
    beam_cols = [c for c in range(w) if (c - locj) % 2 == 0]
    beam_rows = list(range(loci + 1))

    def col_cells(cc):
        return [(r, cc) for r in beam_rows]

    # --- route choices, all derived from the visible input -------------------
    oidx = (locj + h) % 3           # sweep order
    style = (loci + locj) % 3       # stroke style
    dot_last = ((loci + w) % 2 == 1)

    if oidx == 0:                   # outward from the dot, right side first
        ordered, seen = [], set()
        for d in range(w + 1):
            for cc in ((locj,) if d == 0 else (locj + 2 * d, locj - 2 * d)):
                if 0 <= cc < w and cc not in seen:
                    seen.add(cc)
                    ordered.append(cc)
    elif oidx == 1:                 # left to right
        ordered = sorted(beam_cols)
    else:                           # right to left
        ordered = sorted(beam_cols, reverse=True)

    ops, sels = [], []

    def drop_dot():
        # The dot translates one row down -> Move it.  Colour 0 cannot be
        # grabbed by ARCLE's object ops (0 reads as "nothing there"), so a
        # 0-coloured dot is laid down at its landing cell with Color0 instead.
        if fgc != 0:
            ops.append(21)                                   # MoveD: grab + slide
            sels.append(sel_of([(loci, locj)]))
        else:
            ops.append(0)                                    # Color0 at the landing cell
            sels.append(sel_of([(loci + 1, locj)]))
        # No repair of the vacated cell: the beam in column locj paints it.

    if style == 1:
        # ---- one confident stroke: drop the dot, then lay the whole comb.
        drop_dot()
        cells = [(r, cc) for cc in ordered for r in beam_rows]
        ops.append(beam)
        sels.append(sel_of(cells))
    else:
        # column locj is painted last when the dot is dropped last, because the
        # beam would otherwise bury the dot before it can be moved.
        first_pass = [cc for cc in ordered if not (dot_last and cc == locj)]
        tail = [locj] if dot_last else []

        use_copy = (style == 2 and len(first_pass) >= 2)

        if not dot_last:
            drop_dot()

        if use_copy:
            # ---- draw the first beam, then stamp it onto the others.
            src = first_pass[0]
            ops.append(beam)
            sels.append(sel_of(col_cells(src)))
            ops.append(29)                                   # CopyO: the beam just drawn
            sels.append([0, src, loci, 0])                   # exactly that column rectangle
            for cc in first_pass[1:]:
                ops.append(30)                               # Paste beam at its column
                sels.append([cc == cc and 0, cc, 0, 0])
        else:
            # ---- stroke each beam in turn.
            for cc in first_pass:
                ops.append(beam)
                sels.append(sel_of(col_cells(cc)))

        if dot_last:
            drop_dot()
            for cc in tail:
                if use_copy:
                    ops.append(30)                           # Paste the same beam here
                    sels.append([0, cc, 0, 0])
                else:
                    ops.append(beam)
                    sels.append(sel_of(col_cells(cc)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 834ec97d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 834ec97d"
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
                                f"for task 834ec97d"
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
                    f"Failed to build a complete episode for task 834ec97d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"834ec97d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
