"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(cols, 2)

    # portrait and landscape tables are the same rule seen from two sides — show both
    VARIANTS = [{"transposed": False}, {"transposed": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int, transposed=None) -> dict:
    import random

    if transposed is None:
        transposed = random.choice((False, True))

    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    tset = (dmirror, cmirror, rot90, rot270)
    nmfs = random.choice((1, 2))
    while True:
        fns = random.sample(mfs, nmfs)
        parity = sum(1 for f in fns if f in tset) % 2 == 1
        if parity == bool(transposed):
            break

    # axis limits, accounting for the transposition the mirror set will apply
    if transposed:
        w_lim, h_lim = max_h, max_w
    else:
        w_lim, h_lim = max_w, max_h
    w_hi = max(2, min(10, w_lim, h_lim - 1))
    w = unifint(diff_lb, diff_ub, (2, w_hi))
    h_hi = max(w + 1, min(30, h_lim))
    h = unifint(diff_lb, diff_ub, (w + 1, h_hi))

    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2

    for fn in fns:
        gi = fn(gi)
        go = fn(go)

    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off I alone): the grid is a billiard table whose two SHORT walls
    reflect.  Every corner of I carrying the line colour launches a diagonal ray that
    bounces between those two walls until it runs off the far end of the table.

    The ROUTE is measured from I as well, so one grid always draws the same way while
    different grids draw differently:
      * which lit corner's ray is drawn first       <- length of the long side
      * how big a bite: the whole ray / one bounce-to-bounce stroke / one cell
                                                    <- table width + number of lit corners
      * draw the whole table, or draw the first repeat of the pattern and stamp the
        rest of the table with Copy/Paste           <- how many whole repeats fit
    O is never read; handed a different answer grid this returns the same operations.
    """
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, h - 1, w - 1]          # full rectangle: the whole canvas

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    corner_set = set(corners)
    # the line colour only ever sits on corners, so the non-corner cells are the table
    noncorner = [int(I[r, c]) for r in range(h) for c in range(w)
                 if (r, c) not in corner_set]
    bgc = Counter(noncorner if noncorner else I.flatten().tolist()).most_common(1)[0][0]
    rest = [int(v) for v in np.unique(I) if int(v) != bgc]
    if not rest:
        return [34], [submit]
    linc = rest[0]

    portrait = h > w                       # the ray runs along the long axis
    long_len = h if portrait else w
    short_len = w if portrait else h
    if short_len < 2:
        return [34], [submit]
    period = 2 * (short_len - 1)           # the drawing repeats this often along the long axis

    def along(p):
        return p[0] if portrait else p[1]

    seeds = [p for p in corners if int(I[p[0], p[1]]) == linc]

    def strokes_of(r0, c0):
        """the ray from this corner, cut into its straight diagonal strokes"""
        out, run = [], [(r0, c0)]
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1
        r, c = r0, c0
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break                      # ran off the far end -> this ray is done
            run.append((r, c))
            wall = (c == 0 or c == w - 1) if portrait else (r == 0 or r == h - 1)
            if wall:                       # bounce: this stroke ends here
                if portrait:
                    dc = -dc
                else:
                    dr = -dr
                out.append(run)
                run = [(r, c)]
        if len(run) > 1 or not out:
            out.append(run)
        return out

    # ── route choices, all read off I ────────────────────────────────────────
    repeats = long_len // period           # whole copies of the pattern that fit
    grain = (short_len + len(seeds)) % 3   # 0 = whole ray, 1 = stroke, 2 = cell
    use_copy = repeats >= 2 and linc != 0  # a pattern drawn twice over is worth stamping
    order = sorted(seeds)
    if long_len % 2 == 1:                  # odd table -> start from the far corner
        order = order[::-1]

    work = I.copy()

    def draw(cells):
        fresh = [(r, c) for (r, c) in cells if work[r, c] != linc]
        if not fresh:                      # rays cross: already drawn here
            return
        for (r, c) in fresh:
            work[r, c] = linc
        ops.append(int(linc))
        sels.append(sel_of(fresh))

    def draw_stretch(lo, hi):
        for (r0, c0) in order:
            strokes = []
            for stroke in strokes_of(r0, c0):
                part = [p for p in stroke if lo <= along(p) < hi]
                if not part:
                    continue
                if grain == 2:
                    for p in part:         # follow the ray, cell by cell
                        draw([p])
                else:
                    strokes.append(part)
            if grain == 0:                 # the whole bouncing line in one go
                draw([p for part in strokes for p in part])
            else:
                for part in strokes:       # stroke by stroke, bounce to bounce
                    draw(part)

    if use_copy:
        draw_stretch(0, period)            # one full repeat of the pattern
        ops.append(29)                     # CopyO: the stretch just drawn (full rectangle)
        sels.append([0, 0, period - 1, w - 1] if portrait else [0, 0, h - 1, period - 1])
        for k in range(1, repeats):        # stamp it down the rest of the table
            ops.append(30)
            sels.append([k * period, 0, 0, 0] if portrait else [0, k * period, 0, 0])
            if portrait:
                work[k * period:(k + 1) * period, :] = work[0:period, :]
            else:
                work[:, k * period:(k + 1) * period] = work[:, 0:period]
        if repeats * period < long_len:    # leftover strip, shorter than one repeat
            draw_stretch(repeats * period, long_len)
    else:
        draw_stretch(0, long_len)

    ops.append(34)
    sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
