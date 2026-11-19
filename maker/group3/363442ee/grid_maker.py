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


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, barcol=1, dotcol=2, **kw) -> dict:
    lim = min(max_h, max_w)  # mirrors/rotations may transpose the grid
    h = _unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    w = _unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    nremh = _unifint(diff_lb, diff_ub, (2, max(2, lim // h)))
    nremw = _unifint(diff_lb, diff_ub, (2, max(2, (lim - w - 1) // w)))
    rsh, rsw = nremh * h, nremw * w
    remcols = [c for c in range(10) if c not in (bgc, barcol)]
    rsi = np.full((rsh, rsw), bgc, dtype=int)
    rso = rsi.copy()
    dotcands = [(i, j) for i in range(0, rsh, h) for j in range(0, rsw, w)]
    dev = _unifint(diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = random.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(dotcands, ndots)
    nfull = _unifint(diff_lb, diff_ub, (1, 8))
    fullrem = random.sample(remcols, nfull)
    ulc = np.array([[random.choice(fullrem) for _ in range(w)] for _ in range(h)], dtype=int)
    for (i, j) in dots:
        rsi[i + h // 2, j + w // 2] = dotcol
        rso[i:i + h, j:j + w] = ulc
    left = np.vstack([ulc, np.full(((nremh - 1) * h, w), bgc, dtype=int)])
    bar = np.full((rsh, 1), barcol, dtype=int)
    gi = np.hstack([left, bar, rsi])
    go = np.hstack([left, bar, rso])
    fns = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(fns, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Rule: each isolated dot is replaced by a copy of the key block, centred on
    the dot.

    Unit of work: ONE COLOUR OF THE KEY. The original solution stamps one dot at
    a time. This one paints colour by colour instead. For each colour k in the
    key, a single Color(k) op paints the key's colour-k cells at every dot at
    once. The selection is the key's colour-k mask shifted onto each dot centre,
    so it comes from I alone. Colour 0 is just another colour here, so no base
    layer is needed.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the bar: a uniform line of one non-background colour running the full length
    bar = np.zeros((hi, wi), dtype=bool)
    for r in range(hi):
        v = int(I[r, 0])
        if v != bgc and np.all(I[r, :] == v):
            bar[r, :] = True
    for c in range(wi):
        v = int(I[0, c])
        if v != bgc and np.all(I[:, c] == v):
            bar[:, c] = True

    mask = (I != bgc) & (~bar)
    seen = np.zeros_like(mask)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if mask[r0, c0] and not seen[r0, c0]:
                st = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while st:
                    r, c = st.pop()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < hi and 0 <= nc < wi and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            st.append((nr, nc))
                comps.append(cells)

    if comps:
        comps.sort(key=len, reverse=True)
        block = comps[0]  # the key block (solid rectangle)
        br = min(r for r, _ in block)
        bc = min(c for _, c in block)
        bh = max(r for r, _ in block) - br + 1
        bw = max(c for _, c in block) - bc + 1
        key = I[br:br + bh, bc:bc + bw]

        dots = []
        for comp in comps[1:]:
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            dots.append(((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2))
        dots.sort()

        # paint colours from most to least frequent in the key
        colours = sorted(set(key.flatten().tolist()),
                         key=lambda k: (-int(np.sum(key == k)), k))
        grid = I.copy()
        for k in colours:
            offs = [(r, c) for r in range(bh) for c in range(bw) if key[r, c] == k]
            cells = []
            for (dr, dc) in dots:
                top, left = dr - bh // 2, dc - bw // 2
                for (r, c) in offs:
                    rr, cc = top + r, left + c
                    if 0 <= rr < hi and 0 <= cc < wi:
                        cells.append((rr, cc))
            cells = sorted(set(cells))
            if cells and any(grid[r, c] != k for r, c in cells):
                ops.append(int(k))
                sels.append(sel_of(cells))
                for r, c in cells:
                    grid[r, c] = k

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
