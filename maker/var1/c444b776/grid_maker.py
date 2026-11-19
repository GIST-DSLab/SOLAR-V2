"""
ARC Task: c444b776 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------- 
# 1. sample_colors
# -----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """bgc (canvas background) and linc (divider-line colour) are the two colours
    the rule is stated in terms of, so both are fixed for the whole episode.
    The pattern colours are irrelevant to the rule (only the pattern's *shape*
    matters), so they stay free."""
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


# ----------------------------------------------------------------------------- 
# 2. generate
# -----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(int(a + lb * (b - a)), int(a + ub * (b - a)))

    cols = list(range(10))

    # how many panels fit in max_h / max_w (each panel side >= 2, +1 divider between)
    feas_nh = [n for n in range(1, 4) if (max_h - (n - 1)) // n >= 2]
    feas_nw = [n for n in range(1, 4) if (max_w - (n - 1)) // n >= 2]
    nh_ub = max(feas_nh) if feas_nh else 1
    nw_ub = max(feas_nw) if feas_nw else 1

    nh = unifint(diff_lb, diff_ub, (1, nh_ub))
    nw_lb = 1 if nh > 1 else 2
    if nw_ub < nw_lb:
        # need at least one divider somewhere -> force a horizontal split
        nh = 2 if 2 in feas_nh else nh
        nw_lb = 1 if nh > 1 else 2
    nw = unifint(diff_lb, diff_ub, (nw_lb, max(nw_ub, nw_lb)))

    h_ub = min(9, (max_h - (nh - 1)) // nh)
    w_ub = min(9, (max_w - (nw - 1)) // nw)
    h = unifint(diff_lb, diff_ub, (2, max(2, h_ub)))
    w = unifint(diff_lb, diff_ub, (2, max(2, w_ub)))

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)

    # the pattern that lives in one panel
    remcols = [c for c in cols if c != bgc and c != linc]
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    numcol = min(numcol, len(remcols))
    ccols = random.sample(remcols, numcol)
    inds = [(i, j) for i in range(h) for j in range(w)]
    numcels = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2)))
    cels = random.sample(inds, numcels)
    small = [[bgc] * w for _ in range(h)]
    for (i, j) in cels:
        small[i][j] = random.choice(ccols)

    locs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(locs)

    gi = [[linc] * fullw for _ in range(fullh)]
    go = [[linc] * fullw for _ in range(fullh)]
    for (a, b) in locs:
        for i in range(h):
            for j in range(w):
                go[a + i][b + j] = small[i][j]
                gi[a + i][b + j] = small[i][j] if (a, b) == srcloc else bgc

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ----------------------------------------------------------------------------- 
# 3. derive_operations
# -----------------------------------------------------------------------------
def _bands(divs, n):
    bands, prev = [], 0
    for d in divs:
        if d > prev:
            bands.append((prev, d - 1))
        prev = d + 1
    if prev < n:
        bands.append((prev, n - 1))
    return bands


def _structure(I):
    """Return (bgc, linc, row_bands, col_bands, (src_i, src_j)) read from I alone."""
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = max(cnt.items(), key=lambda kv: kv[1])[0]

    for linc in sorted(set(I.flatten().tolist())):
        if linc == bgc:
            continue
        div_rows = [r for r in range(hi) if all(int(I[r, k]) == linc for k in range(wi))]
        div_cols = [c for c in range(wi) if all(int(I[k, c]) == linc for k in range(hi))]
        if not div_rows and not div_cols:
            continue
        row_bands = _bands(div_rows, hi)
        col_bands = _bands(div_cols, wi)
        if not row_bands or not col_bands:
            continue
        if len(row_bands) * len(col_bands) < 2:
            continue
        if len({r1 - r0 for r0, r1 in row_bands}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_bands}) != 1:
            continue
        contentful = []
        for i, (r0, r1) in enumerate(row_bands):
            for j, (c0, c1) in enumerate(col_bands):
                sub = I[r0:r1 + 1, c0:c1 + 1]
                if np.any(sub != bgc):
                    contentful.append((i, j, sub))
        if len(contentful) != 1:
            continue
        si, sj, sub = contentful[0]
        if np.any(sub == linc):          # a real panel never contains divider colour
            continue
        return bgc, linc, row_bands, col_bands, (si, sj)
    return None


def derive_operations(I, O, examples=None):
    """Rule: a grid of equal panels separated by divider lines; exactly one panel
    holds a pattern, every other panel is empty.  Stamp that pattern into every
    empty panel.  Everything below is measured from I only — the route a solver
    takes (which panel first, sweep direction, stamp-a-copy vs. draw-by-colour)
    is decided by features of I, so one instance always yields one route."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    st = _structure(I)
    if st is None:
        return [34], [[0, 0, hi - 1, wi - 1]]
    bgc, linc, row_bands, col_bands, (si, sj) = st

    sr0, sr1 = row_bands[si]
    sc0, sc1 = col_bands[sj]
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1

    # the pattern, in reading order, as offsets inside its panel
    pattern = [(dr, dc, int(I[sr0 + dr, sc0 + dc]))
               for dr in range(sh) for dc in range(sw)
               if int(I[sr0 + dr, sc0 + dc]) != bgc]

    targets = [(i, j) for i in range(len(row_bands)) for j in range(len(col_bands))
               if (i, j) != (si, sj)]

    # ---- route choice, derived purely from the input ------------------------
    sig = (hi * 13 + wi * 7 + si * 5 + sj * 3 + len(pattern) * 11
           + sum(c for _, _, c in pattern) + bgc * 2 + linc)
    method = sig % 3            # 0 stamp a copy, 1 redraw panel by panel, 2 colour by colour
    order_mode = (sig // 3) % 4  # which panel to serve first / sweep direction
    color_mode = (sig // 12) % 2

    if order_mode == 0:                      # top-left onwards, row by row
        targets.sort(key=lambda t: (t[0], t[1]))
    elif order_mode == 1:                    # column by column
        targets.sort(key=lambda t: (t[1], t[0]))
    elif order_mode == 2:                    # bottom-right backwards
        targets.sort(key=lambda t: (-t[0], -t[1]))
    else:                                    # outwards from the filled panel
        targets.sort(key=lambda t: (abs(t[0] - si) + abs(t[1] - sj), t[0], t[1]))

    colour_order = []
    for _, _, cc in pattern:
        if cc not in colour_order:
            colour_order.append(cc)
    if color_mode == 1:
        freq = Counter(cc for _, _, cc in pattern)
        colour_order.sort(key=lambda c: (-freq[c], c))

    def origin(t):
        return row_bands[t[0]][0], col_bands[t[1]][0]

    # ---- carry out the rule --------------------------------------------------
    if method == 0:
        # Stamp the filled panel into each empty one.
        # Selection is exactly the full source-panel rectangle (background included).
        ops.append(28)
        sels.append(sel_of([(r, c) for r in range(sr0, sr1 + 1)
                            for c in range(sc0, sc1 + 1)]))
        zeros = [(dr, dc) for dr, dc, cc in pattern if cc == 0]
        for t in targets:
            r0, c0 = origin(t)
            ops.append(30)
            sels.append(sel_of([(r0, c0)]))
            if zeros:
                # Paste never writes colour 0; those pattern cells are drawn in
                ops.append(0)
                sels.append(sel_of([(r0 + dr, c0 + dc) for dr, dc in zeros]))

    elif method == 1:
        # Redraw the pattern into one panel at a time, one colour group per stroke.
        for t in targets:
            r0, c0 = origin(t)
            for col in colour_order:
                cells = [(r0 + dr, c0 + dc) for dr, dc, cc in pattern if cc == col]
                ops.append(col)
                sels.append(sel_of(cells))

    else:
        # Work colour by colour: place every cell of one colour in all panels, then
        # move on to the next colour.
        for col in colour_order:
            cells = []
            for t in targets:
                r0, c0 = origin(t)
                cells += [(r0 + dr, c0 + dc) for dr, dc, cc in pattern if cc == col]
            ops.append(col)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task c444b776"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c444b776"
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
                                f"for task c444b776"
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
                    f"Failed to build a complete episode for task c444b776 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c444b776-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
