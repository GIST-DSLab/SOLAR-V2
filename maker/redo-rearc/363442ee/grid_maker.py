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


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem2 = [c for c in rem if c != barcol]
    dotcol = random.choice(rem2)
    nfull = random.randint(1, 8)
    fullremcols = random.sample(rem2, nfull)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol,
            "fullremcols": fullremcols}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, dotcol,
             fullremcols) -> dict:
    # After the random mirror/rotation at the end rows and cols may swap,
    # so bound BOTH extents by the tighter of the two limits.
    M = min(max_h, max_w, 30)

    # h = 2*kh+1 needs nremh >= 2 blocks stacked   -> h <= M // 2
    # w = 2*kw+1 needs w + 1 + 2*w <= M            -> w <= (M-1)//3
    kh_max = max(1, min(3, (M // 2 - 1) // 2))
    kw_max = max(1, min(3, ((M - 1) // 3 - 1) // 2))
    kh = _unifint(diff_lb, diff_ub, (1, kh_max))
    kw = _unifint(diff_lb, diff_ub, (1, kw_max))
    h = 2 * kh + 1
    w = 2 * kw + 1

    nremh = _unifint(diff_lb, diff_ub, (2, max(2, M // h)))
    nremw = _unifint(diff_lb, diff_ub, (2, max(2, (M - w - 1) // w)))

    rsh, rsw = nremh * h, nremw * w
    total_w = w + 1 + rsw

    gi = np.full((rsh, total_w), bgc, dtype=int)
    go = np.full((rsh, total_w), bgc, dtype=int)

    # the key pattern block (upper-left corner), colors from fullremcols only
    ulc = np.array([[random.choice(fullremcols) for _ in range(w)]
                    for _ in range(h)], dtype=int)
    gi[0:h, 0:w] = ulc
    go[0:h, 0:w] = ulc

    # separating bar
    gi[:, w] = barcol
    go[:, w] = barcol

    # dots on the cell grid of the right region
    cands = [(i * h, j * w) for i in range(nremh) for j in range(nremw)]
    dev = _unifint(diff_lb, diff_ub, (1, max(1, len(cands) // 2)))
    ndots = random.choice((dev, len(cands) - dev))
    ndots = min(max(1, ndots), len(cands))
    dots = random.sample(cands, ndots)

    off = w + 1
    for (r, c) in dots:
        gi[r + h // 2, off + c + w // 2] = dotcol
        go[r:r + h, off + c:off + c + w] = ulc

    # random orientation (identity, dmirror, cmirror, vmirror, hmirror,
    #                     rot90, rot180, rot270)
    fns = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    nmfs = random.choice((1, 2))
    for fn in random.sample(fns, nmfs):
        gi = fn(gi)
        go = fn(go)

    return {"input": np.array(gi).tolist(), "output": np.array(go).tolist()}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """
    Rule: one solid multi-coloured rectangular 'key' block sits on one side of a
    full-length bar; the other side holds isolated dots.  Each dot is replaced by
    a copy of the key block, centred on the dot (block is odd-sized, so the dot
    is exactly its centre cell).  Orientation of the block is preserved, so the
    rule is invariant to the mirrors/rotations the generator applies.

    Copy/Paste is transparent to colour 0, so when the block contains 0s we first
    lay a Color0 base over the whole destination cell and then Paste the block's
    non-zero cells on top of it.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # background = the colour the canvas was painted with before anything else
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the bar: a full-length uniform line of a non-background colour
    bar = np.zeros((hi, wi), dtype=bool)
    for r in range(hi):
        v = int(I[r, 0])
        if v != bgc and bool(np.all(I[r, :] == v)):
            bar[r, :] = True
    for c in range(wi):
        v = int(I[0, c])
        if v != bgc and bool(np.all(I[:, c] == v)):
            bar[:, c] = True

    # connected components of everything that is neither background nor bar
    mask = (I != bgc) & (~bar)
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if mask[r0, c0] and not seen[r0, c0]:
                stack = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and mask[nr, nc] \
                                and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(cells)

    if not comps:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    comps.sort(key=len, reverse=True)
    block = comps[0]                       # the key pattern: biggest component
    brs = [r for r, _ in block]
    bcs = [c for _, c in block]
    br, bc = min(brs), min(bcs)
    bh = max(brs) - br + 1
    bw = max(bcs) - bc + 1

    # dot markers: every other component (each is a single isolated cell)
    dot_centers = []
    for comp in comps[1:]:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        dot_centers.append(((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2))
    dot_centers.sort()

    blockvals = I[br:br + bh, bc:bc + bw]
    has_zero = bool(np.any(blockvals == 0))
    has_nonzero = bool(np.any(blockvals != 0))

    # grab the key block from the INPUT into the clipboard.
    # The block is exactly this full rectangle, so a bbox selection is the
    # object's true cell set.
    if has_nonzero:
        ops.append(28); sels.append([br, bc, bh - 1, bw - 1])

    for (dr, dc) in dot_centers:
        top = dr - bh // 2
        left = dc - bw // 2
        rect = [(r, c)
                for r in range(top, top + bh)
                for c in range(left, left + bw)
                if 0 <= r < hi and 0 <= c < wi]
        if not rect:
            continue
        # base layer: Paste is transparent to 0, so when the key block carries
        # 0s, paint the whole destination cell 0 first and draw on top of it.
        if has_zero:
            ops.append(0); sels.append(sel_of(rect))
        if has_nonzero:
            ops.append(30); sels.append(sel_of([(top, left)]))

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
