"""
ARC Task: a3325580 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is fixed for the episode. The rule ("colours whose
    # objects have the most cells, in leftmost order") works the same for any
    # set of object colours.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return randint(min(lo, hi), max(lo, hi))

    if bgc is None:
        bgc = choice(range(10))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, 9))
    remcols = [c for c in range(10) if c != bgc]
    ccols = sample(remcols, nobjs)
    gi = [[bgc] * w for _ in range(h)]
    lmocc = set()
    inds = {(i, j) for i in range(h) for j in range(w)}

    def dn(cells):
        return {(a + da, b + db) for a, b in cells
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))}

    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = randint(nobjs + 1, 30)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(1, 6)
        ow = randint(1, 6)
        while oh * ow < mxncells:
            oh = randint(1, 6)
            ow = randint(1, 6)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = choice(sorted(bounds))
        obj = {sp}
        for _ in range(ncells - 1):
            obj.add(choice(sorted((bounds - obj) & dn(obj))))
        if frozenset(obj) in seenobjs:
            continue
        mr = min(a for a, _ in obj)
        mc = min(b for _, b in obj)
        obj = {(a - mr, b - mc) for a, b in obj}
        oh = max(a for a, _ in obj) + 1
        ow = max(b for _, b in obj) + 1
        cands = [ij for ij in inds
                 if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc]
        if not cands:
            continue
        loc = choice(sorted(cands))
        plcd = {(a + loc[0], b + loc[1]) for a, b in obj}
        if plcd <= inds:
            inds = (inds - plcd) - dn(plcd)
            for a, b in plcd:
                gi[a][b] = ccols[succ]
            succ += 1
            lmocc.add(loc[1])

    comps = _components(np.array(gi), bgc)
    mx = max(len(c) for _, c in comps)
    big = sorted([x for x in comps if len(x[1]) == mx],
                 key=lambda x: min(b for _, b in x[1]))
    go = [[col for col, _ in big] for _ in range(mx)]
    return {"input": gi, "output": go}


def _components(I, bgc):
    """Single-colour, 4-connected objects that are not background: (colour, cells)."""
    hi, wi = I.shape
    seen = np.zeros_like(I, dtype=bool)
    out = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                a, b = st.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if (0 <= na < hi and 0 <= nb < wi and not seen[na, nb]
                            and I[na, nb] == col):
                        seen[na, nb] = True
                        st.append((na, nb))
            out.append((col, cells))
    return out


def derive_operations(I, O, examples=None):
    """Reading (from the golf entry): the unit is ONE ROW, not one stripe per
    object. Count each colour's cells. The colours tied for the largest count m,
    ordered by where they first appear going column by column (= leftmost),
    make up a single row. The answer is that row stacked m times.
    So: cut the canvas to the m x n frame, write the row once, then copy it
    down onto every other row. No vertical stripes and no rotate/mirror."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background is fixed for the episode: take the majority colour over this
    # input and all the demonstration inputs.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    comps = _components(I, bgc)
    m = max(len(c) for _, c in comps)                 # the largest count
    row = [col for col, cells in sorted([x for x in comps if len(x[1]) == m],
                                        key=lambda x: min(b for _, b in x[1]))]
    n = len(row)

    ops, sels = [], []

    # 1. Cut the canvas to the answer's frame: m rows x n columns (full rectangle).
    ops.append(33); sels.append([0, 0, m - 1, n - 1])
    g = np.zeros((m, n), dtype=int)                   # track the visible canvas
    sub = I[:min(m, hi), :min(n, wi)]
    g[:sub.shape[0], :sub.shape[1]] = sub             # resize copies only non-zero cells

    # 2. Paste can't write colour 0. If 0 is one of the row's colours and the
    #    canvas still holds leftovers, first fill the whole frame with 0
    #    (full rectangle).
    if 0 in row and g.any():
        ops.append(0); sels.append([0, 0, m - 1, n - 1])
        g[:, :] = 0

    # 3. Write the row once: cell (0, j) gets the j-th colour, left to right.
    for j, col in enumerate(row):
        if g[0, j] != col:
            ops.append(int(col)); sels.append(sel_of([(0, j)]))
            g[0, j] = col

    # 4. Copy that row down onto each remaining row that doesn't already match.
    targets = [r for r in range(1, m)
               if any(g[r, j] != row[j] for j in range(n) if row[j] != 0)]
    if targets:
        ops.append(29); sels.append([0, 0, 0, n - 1])  # the whole first row (rectangle)
        for r in targets:
            ops.append(30); sels.append([r, 0, 0, 0])  # paste origin = start of row r
            for j in range(n):
                if row[j] != 0:
                    g[r, j] = row[j]

    ops.append(34); sels.append([0, 0, m - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3325580"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3325580"
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
                                f"for task a3325580"
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
                    f"Failed to build a complete episode for task a3325580 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3325580-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
