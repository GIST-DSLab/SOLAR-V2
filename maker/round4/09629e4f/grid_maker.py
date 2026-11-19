"""
ARC Task: 09629e4f (RE-ARC) — LLM-generated grid_maker
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
    barcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "barcol": barcol}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barcol=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if barcol is None:
        barcol = random.choice([c for c in cols if c != bgc])

    hmax = 2
    for k in range(2, 6):
        if k * k + k - 1 <= max_h:
            hmax = k
    wmax = 2
    for k in range(2, 6):
        if k * k + k - 1 <= max_w:
            wmax = k
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    nrows, ncolumns = h, w
    remcols = [c for c in cols if c != bgc and c != barcol]
    ncols = _unifint(diff_lb, diff_ub, (2, min(7, h * w - 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = np.full((fullh, fullw), barcol, dtype=int)
    locs = [(r, c) for r in range(0, fullh, h + 1) for c in range(0, fullw, w + 1)]
    trgloc = random.choice(locs)
    remlocs = [l for l in locs if l != trgloc]
    colssf = random.sample(remcols, ncols)
    dropped = random.choice(colssf)
    colsss = [c for c in colssf if c != dropped]
    trgssf = random.sample(inds, ncols - 1)
    tr, tc = trgloc
    gi[tr:tr + h, tc:tc + w] = bgc
    for (i, j), cl in zip(trgssf, colsss):
        gi[tr + i, tc + j] = cl
    for (rr, rc) in remlocs:
        gi[rr:rr + h, rc:rc + w] = bgc
        trgss = random.sample(inds, ncols)
        for (i, j), cl in zip(trgss, colssf):
            gi[rr + i, rc + j] = cl
    go = np.full((fullh, fullw), bgc, dtype=int)
    go[gi == barcol] = barcol
    for (i, j), cl in zip(trgssf, colsss):
        r0, c0 = i * (h + 1), j * (w + 1)
        go[r0:r0 + h, c0:c0 + w] = cl
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # bar colour = colour of a fully uniform row/column
    barcol = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            barcol = int(I[r, 0])
            break
    if barcol is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                barcol = int(I[0, c])
                break

    bar_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1 and I[r, 0] == barcol]
    bar_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1 and I[0, c] == barcol]

    h = bar_rows[0] if bar_rows else hi
    w = bar_cols[0] if bar_cols else wi

    nrows = (hi + 1) // (h + 1)
    ncolumns = (wi + 1) // (w + 1)

    non_bar_vals = I[I != barcol].tolist()
    bgc = int(Counter(non_bar_vals).most_common(1)[0][0])

    # target block: the one with the fewest distinct non-bgc colours
    best = None
    best_count = None
    for i in range(nrows):
        for j in range(ncolumns):
            r0 = i * (h + 1)
            c0 = j * (w + 1)
            block = I[r0:r0 + h, c0:c0 + w]
            cnt = len(set(block.flatten().tolist()) - {bgc})
            if best is None or cnt < best_count:
                best = block.copy()
                best_count = cnt
    tblock = best

    def block_cells(i, j):
        r0 = i * (h + 1)
        c0 = j * (w + 1)
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]

    ops, sels = [], []

    # clear to bgc every sub-cell that the target pattern leaves empty
    for i in range(nrows):
        for j in range(ncolumns):
            if int(tblock[i, j]) == bgc:
                ops.append(bgc)
                sels.append(sel_of(block_cells(i, j)))

    # paint the sub-cells the target pattern marks, in that mark's colour
    for i in range(h):
        for j in range(w):
            cl = int(tblock[i, j])
            if cl != bgc:
                ops.append(cl)
                sels.append(sel_of(block_cells(i, j)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 09629e4f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 09629e4f"
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
                                f"for task 09629e4f"
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
                    f"Failed to build a complete episode for task 09629e4f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"09629e4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
