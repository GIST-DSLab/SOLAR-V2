"""
ARC Task: 363442ee (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    dotcol = random.choice(rem)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, barcol=None, dotcol=None, **kw) -> dict:
    from dsl import (interval, canvas, product, totuple, asindices, fill, asobject,
                     add, paint, shift, hconcat, vconcat, identity, dmirror, cmirror,
                     vmirror, hmirror, rot90, rot180, rot270, remove)
    from utils import unifint
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = random.choice(cols)
    if barcol is None or barcol == bgc:
        barcol = random.choice([c for c in cols if c != bgc])
    if dotcol is None or dotcol in (bgc, barcol):
        dotcol = random.choice([c for c in cols if c not in (bgc, barcol)])
    lim = min(max_h, max_w)  # mirrors may transpose the grid
    h = unifint(diff_lb, diff_ub, (1, 3))
    w = unifint(diff_lb, diff_ub, (1, 3))
    h = h * 2 + 1
    w = w * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, max(2, lim // h)))
    nremw = unifint(diff_lb, diff_ub, (2, max(2, (lim - w - 1) // w)))
    rsh = nremh * h
    rsw = nremw * w
    rss = (rsh, rsw)
    remcols = [c for c in cols if c not in (bgc, barcol)]
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * h, w))
    ulc = canvas(bgc, (h, w))
    bar = canvas(barcol, (nremh * h, 1))
    dotcands = totuple(product(interval(0, rsh, h), interval(0, rsw, w)))
    dev = unifint(diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = random.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(list(dotcands), ndots)
    nfullremcols = unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = random.sample(remcols, min(nfullremcols, len(remcols)))
    for ij in asindices(ulc):
        ulc = fill(ulc, random.choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (h // 2, w // 2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    go = hconcat(hconcat(vconcat(ulc, ls), bar), rso)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): the side of the bar away from the key
    is a lattice of key-sized tiles that starts at the bar. Each tile carries a
    single on/off bit, which is its centre cell. Every tile that is switched on
    becomes a copy of the key. So the unit is the lattice tile, not the dot. We
    walk the tiles in reading order, read each tile's centre, and stamp the key
    onto every tile that is on. Paste is transparent to 0, so when the key holds
    0s we first lay Color0 over the whole tile and then paste the key on top.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the bar: a full-length uniform non-background line
    bar_row = bar_col = None
    for r in range(hi):
        if I[r, 0] != bgc and np.all(I[r, :] == I[r, 0]):
            bar_row = r
    for c in range(wi):
        if I[0, c] != bgc and np.all(I[:, c] == I[0, c]):
            bar_col = c

    # the key: the large solid block of non-background cells that is not part of the bar
    mask = I != bgc
    if bar_row is not None:
        mask[bar_row, :] = False
    if bar_col is not None:
        mask[:, bar_col] = False
    seen = np.zeros_like(mask)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if mask[r0, c0] and not seen[r0, c0]:
                st = [(r0, c0)]; seen[r0, c0] = True; cells = []
                while st:
                    a, b = st.pop(); cells.append((a, b))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + dr, b + dc
                        if 0 <= na < hi and 0 <= nb < wi and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True; st.append((na, nb))
                comps.append(cells)
    comps.sort(key=len, reverse=True)
    key = comps[0]
    br = min(r for r, _ in key); bc = min(c for _, c in key)
    bh = max(r for r, _ in key) - br + 1
    bw = max(c for _, c in key) - bc + 1
    kv = I[br:br + bh, bc:bc + bw]
    has_zero = bool(np.any(kv == 0))
    has_nonzero = bool(np.any(kv != 0))

    # the tile lattice on the far side of the bar, anchored at the bar
    tiles = []
    if bar_col is not None and (bar_row is None or bh * bw and hi % bh == 0):
        rows = list(range(0, hi - bh + 1, bh))
        if bc < bar_col:
            colss = list(range(bar_col + 1, wi - bw + 1, bw))
        else:
            colss = sorted(bar_col - (m + 1) * bw for m in range(bar_col // bw))
        tiles = [(t, l) for t in rows for l in colss]
    else:
        colss = list(range(0, wi - bw + 1, bw))
        if br < bar_row:
            rows = list(range(bar_row + 1, hi - bh + 1, bh))
        else:
            rows = sorted(bar_row - (m + 1) * bh for m in range(bar_row // bh))
        tiles = [(t, l) for t in rows for l in colss]

    on = [(t, l) for (t, l) in tiles if I[t + bh // 2, l + bw // 2] != bgc]

    if on and has_nonzero:
        # the key block is exactly this full rectangle
        ops.append(28); sels.append([br, bc, bh - 1, bw - 1])
    for (t, l) in on:
        rect = [(r, c) for r in range(t, t + bh) for c in range(l, l + bw)]
        if has_zero:
            ops.append(0); sels.append(sel_of(rect))
        if has_nonzero:
            ops.append(30); sels.append(sel_of([(t, l)]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 363442ee"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 363442ee"
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
                                f"for task 363442ee"
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
                    f"Failed to build a complete episode for task 363442ee "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"363442ee-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
