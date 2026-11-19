"""
ARC Task: 1e0a9b12 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int = 0) -> dict:
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        h = unifint(diff_lb, diff_ub, (3, max_h))
        w = unifint(diff_lb, diff_ub, (3, max_w))
        nc = unifint(diff_lb, diff_ub, (1, w))
        cols = interval(0, 10, 1)
        gi = canvas(bgc, (h, w))
        remcols = remove(bgc, cols)
        scols = [choice(remcols) for j in range(nc)]
        slocs = sample(interval(0, w, 1), nc)
        inds = totuple(connect(ORIGIN, (h - 1, 0)))
        for c, l in zip(scols, slocs):
            nc2 = randint(1, h - 1)
            sel = sample(inds, nc2)
            gi = fill(gi, c, shift(sel, tojvec(l)))
        go = replace(ff(replace(gi, bgc, -1)), -1, bgc)
        if colorcount(gi, bgc) > argmax(remove(bgc, palette(gi)), lbind(colorcount, gi)):
            break
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    MOVE_D = 21

    # ── background colour ────────────────────────────────────────────────────
    # The generator paints a canvas of bgc, then drops cells of ONE extra colour
    # into a few DISTINCT columns, at most h-1 cells per column.  So bgc is the
    # colour present in every single column; another colour can only manage that
    # in the degenerate case where every column carries the same mark.
    common = set(range(10))
    for c in range(w):
        common &= set(I[:, c].tolist())
    bgc = None
    if len(common) == 1:
        bgc = common.pop()
    if bgc is None and examples:
        # gravity empties the top row of every worked output: it shows the background
        votes = Counter()
        for ex in examples:
            eo = ex.get("output") if isinstance(ex, dict) else (
                ex[1] if isinstance(ex, (tuple, list)) and len(ex) == 2 else None)
            if eo is None:
                continue
            eo = np.asarray(eo, dtype=int)
            if eo.size:
                votes[int(eo[0, 0])] += 1
        if votes:
            bgc = votes.most_common(1)[0][0]
    if bgc is None:
        cnt = Counter(I.flatten().tolist())
        pool = common if common else set(cnt)
        bgc = max(pool, key=lambda k: cnt[k])

    # ── read each column as a stack of falling pieces ────────────────────────
    # Cells of one colour that touch each other fall together; a gap or a colour
    # change starts a new piece.  Everything drops to the bottom, keeping order.
    column = {}
    for c in range(w):
        rows = [r for r in range(h) if I[r, c] != bgc]
        if not rows:
            continue
        runs = []
        a = p = rows[0]
        for r in rows[1:]:
            if r == p + 1 and I[r, c] == I[p, c]:
                p = r
            else:
                runs.append((a, p))
                a = p = r
        runs.append((a, p))
        column[c] = (rows, runs)

    if not column:
        return [34], [[0, 0, h - 1, w - 1]]

    # ── choose a route through the columns, reading it off the input ─────────
    # Start where the work is: if the fullest column sits near an edge, sweep
    # from that edge across; if it sits in the middle, take the fullest columns
    # first instead.
    tallest = max(len(column[c][0]) for c in column)
    heaviest = min(c for c in column if len(column[c][0]) == tallest)
    if heaviest * 3 < w:
        order = sorted(column)                                      # left to right
    elif heaviest * 3 >= 2 * w:
        order = sorted(column, reverse=True)                        # right to left
    else:
        order = sorted(column, key=lambda c: (-len(column[c][0]), c))
    # and either tidy the emptied top of each column as we leave it, or leave
    # every leftover for one last sweep — decided by how many columns are marked.
    sweep_at_end = (len(column) % 2 == 0)

    grid = I.copy()
    ops, sels = [], []
    leftovers = []

    for c in order:
        rows, runs = column[c]
        floor = h                              # lowest row still free in this column
        for (a, b) in reversed(runs):          # the bottom piece lands first
            m = b - a + 1
            d = (floor - 1) - b                # how far this piece has to fall
            floor -= m
            if d <= 0:
                continue                       # already resting on the stack
            colour = int(I[a, c])
            src = [(r, c) for r in range(a, b + 1)]
            dst = [(r + d, c) for r in range(a, b + 1)]
            if colour != 0:
                # grab the piece once, then keep sliding it with empty selections:
                # ARCLE re-pastes it over its snapshot, so its trail heals itself
                ops.append(MOVE_D)
                sels.append(sel_of(src))
                for _ in range(d - 1):
                    ops.append(MOVE_D)
                    sels.append(sel_of([]))
                for (r, cc) in src:
                    grid[r, cc] = 0            # ARCLE zeroes the grabbed footprint
                for (r, cc) in dst:
                    grid[r, cc] = colour
            else:
                # colour 0 is transparent to ARCLE's object ops — a 0 piece cannot
                # be grabbed, so draw it where it comes to rest instead
                if any(grid[r, cc] != 0 for (r, cc) in dst):
                    ops.append(0)
                    sels.append(sel_of(dst))
                    for (r, cc) in dst:
                        grid[r, cc] = 0
        # above the settled stack the column must read as empty background
        top = h - len(rows)
        gap = [(r, c) for r in range(top) if grid[r, c] != bgc]
        if gap:
            if sweep_at_end:
                leftovers.extend(gap)
            else:
                ops.append(int(bgc))
                sels.append(sel_of(gap))
                for (r, cc) in gap:
                    grid[r, cc] = bgc

    if leftovers:
        ops.append(int(bgc))
        sels.append(sel_of(leftovers))

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
                        f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
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
                                f"for task 1e0a9b12"
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
                    f"Failed to build a complete episode for task 1e0a9b12 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e0a9b12-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
