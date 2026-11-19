"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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

# The only discrete structural case in this task: the generator flips a coin and
# dmirrors (transposes) the grid, so the bands run either horizontally (every row
# of O uniform) or vertically (every column of O uniform).  Both must be shown.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # The band colours are sampled randomly by the generator -> fix the palette
    # for the whole episode.  Noise colours stay random (they are pure
    # distractors and are erased by the rule).
    palette = list(range(10))
    random.shuffle(palette)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transpose=None, **_) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if palette is None:
        palette = list(range(10))
        random.shuffle(palette)
    if transpose is None:
        transpose = random.choice((True, False))

    # band axis length limits (before the optional transpose)
    if transpose:
        maxrows, maxcols = max_w, max_h
    else:
        maxrows, maxcols = max_h, max_w
    maxrows = max(2, min(30, maxrows))
    maxcols = max(2, min(30, maxcols))

    nr_ub = max(1, min(10, maxrows // 2))
    nr = unifint(diff_lb, diff_ub, (1, nr_ub))

    w_lb = 4 if maxcols >= 4 else maxcols
    w = unifint(diff_lb, diff_ub, (w_lb, maxcols))

    scols = list(palette[:nr])
    random.shuffle(scols)

    sgs = [[[col] * w for _ in range(2)] for col in scols]

    slack = maxrows - 2 * nr
    numexp = unifint(diff_lb, diff_ub, (0, max(0, min(30 - nr, slack))))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))

    sgs2 = []
    for idx, col in enumerate(scols):
        sg = [list(row) for row in sgs[idx]]
        a = len(sg)
        b = w
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = [(i, j) for i in range(a) for j in range(b)]
        noise = random.sample(inds, min(nnoise, len(inds)))
        oc = [c for c in range(10) if c != col]
        for (i, j) in noise:
            sg[i][j] = random.choice(oc)
        # every row must stay dominated by its band colour, so the majority rule
        # is well defined for every row of the band
        for i in range(a):
            while sum(1 for e in sg[i] if e == col) <= b // 2:
                locs = [j for j, e in enumerate(sg[i]) if e != col]
                if not locs:
                    break
                sg[i][random.choice(locs)] = col
        sgs2.append(sg)

    gi = [list(row) for sg in sgs2 for row in sg]
    go = [list(row) for sg in sgs for row in sg]

    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """The grid is a stack of noisy bands; each band collapses to its own colour.

    Route: clean ONE strip (the leftmost column for horizontal bands, the top row
    for vertical ones) so it holds each band's colour, blank the rest of the
    canvas, then stamp that strip across the remaining columns / rows -- the
    replication (h/v upscale) the rule is made of.  The canvas is blanked first
    because Paste never writes the colour-0 cells of the clipboard.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]

    if np.array_equal(I, O):
        ops.append(34); sels.append(full)
        return ops, sels

    rows_uniform = all(len(set(O[r].tolist())) == 1 for r in range(h))
    cols_uniform = all(len(set(O[:, c].tolist())) == 1 for c in range(w))

    if rows_uniform:
        # ---- horizontal bands: reference strip = column 0, stamped rightwards
        colors = [int(O[r, 0]) for r in range(h)]

        r = 0
        while r < h:
            r2 = r
            while r2 + 1 < h and colors[r2 + 1] == colors[r]:
                r2 += 1
            c = colors[r]
            cells = [(rr, 0) for rr in range(r, r2 + 1) if int(I[rr, 0]) != c]
            if cells:                              # clean this band's strip cell run
                ops.append(c); sels.append(sel_of(cells))
            r = r2 + 1

        if w > 1:
            clear = [(rr, cc) for rr in range(h) for cc in range(1, w)
                     if int(I[rr, cc]) != 0]
            if clear:                              # blank the area to be stamped
                ops.append(0); sels.append(sel_of(clear))
            if any(c != 0 for c in colors):
                # full-rectangle selection: the whole strip column, on purpose
                ops.append(29); sels.append([0, 0, h - 1, 0])
                for cc in range(1, w):
                    ops.append(30); sels.append([0, cc, 0, 0])

    elif cols_uniform:
        # ---- vertical bands: reference strip = row 0, stamped downwards
        colors = [int(O[0, c]) for c in range(w)]

        c = 0
        while c < w:
            c2 = c
            while c2 + 1 < w and colors[c2 + 1] == colors[c]:
                c2 += 1
            col = colors[c]
            cells = [(0, cc) for cc in range(c, c2 + 1) if int(I[0, cc]) != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))
            c = c2 + 1

        if h > 1:
            clear = [(rr, cc) for rr in range(1, h) for cc in range(w)
                     if int(I[rr, cc]) != 0]
            if clear:
                ops.append(0); sels.append(sel_of(clear))
            if any(v != 0 for v in colors):
                # full-rectangle selection: the whole strip row, on purpose
                ops.append(29); sels.append([0, 0, 0, w - 1])
                for rr in range(1, h):
                    ops.append(30); sels.append([rr, 0, 0, 0])

    else:
        # defensive fallback: paint whatever differs, grouped by target colour
        for col in sorted(set(O.flatten().tolist())):
            cells = [(r, c) for r in range(h) for c in range(w)
                     if int(O[r, c]) == col and int(I[r, c]) != col]
            if cells:
                ops.append(int(col)); sels.append(sel_of(cells))

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
