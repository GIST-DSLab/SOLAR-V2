"""
ARC Task: a699fb00 (RE-ARC) — LLM-generated grid_maker
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
    """The generator samples exactly two colours: the canvas colour and the dot
    colour, both drawn from 0..9 minus 2 (2 is the colour the gaps get filled
    with).  Either of them may legitimately be 0 — the route below is written so
    that 0 never has to travel inside a Paste/Copy."""
    cols = [c for c in range(10) if c != 2]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    hub = max(4, min(30, max_h))
    wub = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, hub))
    w = unifint(diff_lb, diff_ub, (4, wub))
    numls = unifint(diff_lb, diff_ub, (1, h - 1))
    opts = interval(0, h, 1)
    locs = sample(opts, numls)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in locs:
        endidx = unifint(diff_lb, diff_ub, (2, w - 2))
        ofs = unifint(diff_lb, diff_ub, (1, endidx // 2)) * 2
        ofs = min(max(2, ofs), endidx)
        startidx = endidx - ofs
        ln = connect((ii, startidx), (ii, endidx))
        go = fill(go, 2, ln)
        sparseln = {(ii, jj) for jj in range(startidx, endidx + 1, 2)}
        go = fill(go, fgc, sparseln)
        gi = fill(gi, fgc, sparseln)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule (read off the worked examples, then measured from I alone): every row
    carries at most one dotted line whose dots sit two apart; each one-cell hole
    between two consecutive dots of a line is filled with the marker colour (2 in
    the demonstrations).  O is used only for its shape.

    Which ROUTE is taken through that rule is decided by the input itself
    (colours + number of lines + where the first line sits), so a given instance
    always replays identically while different instances are solved in different
    orders / styles — the way two people would differ.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- colours -------------------------------------------------------
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]           # canvas colour (dots are sparse)

    # the marker colour comes from the demonstrations, not from O
    fill_col = 2
    if examples:
        for pair in examples:
            try:
                a = np.asarray(pair[0], dtype=int)
                b = np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            if a.shape != b.shape:
                continue
            introduced = set(b[a != b].tolist())
            if len(introduced) == 1:
                fill_col = int(introduced.pop())
                break

    # ---- find the dotted lines (rows whose marks are spaced two apart) ---
    lines = []
    for r in range(hi):
        dots = [c for c in range(wi) if I[r, c] != bgc]
        if len(dots) < 2:
            continue
        runs, cur = [], [dots[0]]
        for prev, cc in zip(dots, dots[1:]):
            if cc - prev == 2:
                cur.append(cc)
            else:
                runs.append(cur)
                cur = [cc]
        runs.append(cur)
        for run in runs:
            if len(run) < 2:
                continue
            lines.append({
                "row": r,
                "start": run[0],
                "end": run[-1],
                "dots": [(r, c) for c in run],
                "gaps": [(r, run[k] + 1) for k in range(len(run) - 1)],
            })

    ops, sels = [], []
    if not lines:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    first = lines[0]
    dotc = int(I[first["dots"][0]])
    route = (int(bgc) + 3 * dotc + len(lines) + first["row"] + first["start"]) % 7

    def paint_gaps(ln):
        ops.append(fill_col)
        sels.append(sel_of(ln["gaps"]))

    if route == 0:
        # work down the picture, one dotted line at a time
        for ln in lines:
            paint_gaps(ln)

    elif route == 1:
        # start at the bottom of the picture and work upwards
        for ln in sorted(lines, key=lambda l: -l["row"]):
            paint_gaps(ln)

    elif route == 2:
        # see every hole at once and fill them all in one stroke
        allgaps = [g for ln in lines for g in ln["gaps"]]
        ops.append(fill_col)
        sels.append(sel_of(allgaps))

    elif route == 3:
        # leftmost line first, and fill its holes one by one, left to right
        for ln in sorted(lines, key=lambda l: (l["start"], l["row"])):
            for g in ln["gaps"]:
                ops.append(fill_col)
                sels.append(sel_of([g]))

    elif route == 4:
        # deal with the longest lines first, shortest last
        for ln in sorted(lines, key=lambda l: (-(l["end"] - l["start"]), l["row"])):
            paint_gaps(ln)

    elif route == 5:
        # lines that occupy the same columns are the same shape: finish the first
        # one, then copy that finished line onto its twins
        groups = {}
        for ln in lines:
            groups.setdefault((ln["start"], ln["end"]), []).append(ln)
        for gkey in sorted(groups, key=lambda k: groups[k][0]["row"]):
            grp = groups[gkey]
            head = grp[0]
            paint_gaps(head)
            if len(grp) > 1:
                # bbox == exactly the head line's full span (one row, no bgc inside)
                ops.append(29)
                sels.append([head["row"], head["start"], 0, head["end"] - head["start"]])
                for ln in grp[1:]:
                    # Paste writes the marker cells; if the dot colour happens to be 0
                    # those cells simply do not travel — and they need not, the twin
                    # row already carries its own dots at exactly those columns.
                    ops.append(30)
                    sels.append([ln["row"], ln["start"], 0, 0])

    else:
        # draw each line solid first, then put its dots back on top of the bar
        for ln in lines:
            span = [(ln["row"], c) for c in range(ln["start"], ln["end"] + 1)]
            ops.append(fill_col)
            sels.append(sel_of(span))
            # explicit Color<dot colour> — works even when that colour is 0
            ops.append(int(I[ln["dots"][0]]))
            sels.append(sel_of(ln["dots"]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a699fb00"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a699fb00"
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
                                f"for task a699fb00"
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
                    f"Failed to build a complete episode for task a699fb00 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a699fb00-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
