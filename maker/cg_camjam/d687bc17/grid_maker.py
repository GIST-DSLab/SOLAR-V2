"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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

from dsl import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h=30, max_w=30,
             bgc=None, c1=None, c2=None, c3=None, c4=None) -> dict:
    cols = interval(0, 10, 1)
    if None in (bgc, c1, c2, c3, c4):
        bgc, c1, c2, c3, c4 = sample(cols, 5)
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c1, connect((0, 0), (0, w - 1)))
    gi = fill(gi, c2, connect((0, 0), (h - 1, 0)))
    gi = fill(gi, c3, connect((h - 1, w - 1), (0, w - 1)))
    gi = fill(gi, c4, connect((h - 1, w - 1), (h - 1, 0)))
    inds = asindices(gi)
    gi = fill(gi, bgc, corners(inds))
    go = tuple(e for e in gi)
    cands = backdrop(inbox(inbox(inds)))
    ndots = unifint(diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    dots = sample(totuple(cands), ndots)
    dots = {(choice((c1, c2, c3, c4)), ij) for ij in dots}
    n1 = toindices(sfilter(dots, lambda cij: cij[0] == c1))
    n1coverage = apply(last, n1)
    if len(n1coverage) == w - 4 and w > 5:
        n1coverage = remove(choice(totuple(n1coverage)), n1coverage)
    for jj in n1coverage:
        loci = choice([ij[0] for ij in sfilter(n1, lambda ij: ij[1] == jj)])
        gi = fill(gi, c1, {(loci, jj)})
        go = fill(go, c1, {(1, jj)})
    n2 = toindices(sfilter(dots, lambda cij: cij[0] == c2))
    n2coverage = apply(first, n2)
    if len(n2coverage) == h - 4 and h > 5:
        n2coverage = remove(choice(totuple(n2coverage)), n2coverage)
    for ii in n2coverage:
        locj = choice([ij[1] for ij in sfilter(n2, lambda ij: ij[0] == ii)])
        gi = fill(gi, c2, {(ii, locj)})
        go = fill(go, c2, {(ii, 1)})
    n3 = toindices(sfilter(dots, lambda cij: cij[0] == c4))
    n3coverage = apply(last, n3)
    if len(n3coverage) == w - 4 and w > 5:
        n3coverage = remove(choice(totuple(n3coverage)), n3coverage)
    for jj in n3coverage:
        loci = choice([ij[0] for ij in sfilter(n3, lambda ij: ij[1] == jj)])
        gi = fill(gi, c4, {(loci, jj)})
        go = fill(go, c4, {(h - 2, jj)})
    n4 = toindices(sfilter(dots, lambda cij: cij[0] == c3))
    n4coverage = apply(first, n4)
    if len(n4coverage) == h - 4 and h > 5:
        n4coverage = remove(choice(totuple(n4coverage)), n4coverage)
    for ii in n4coverage:
        locj = choice([ij[1] for ij in sfilter(n4, lambda ij: ij[0] == ii)])
        gi = fill(gi, c3, {(ii, locj)})
        go = fill(go, c3, {(ii, w - 2)})
    noisecands = ofcolor(gi, bgc)
    noisecols = difference(cols, (bgc, c1, c2, c3, c4))
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
    noise = sample(totuple(noisecands), nnoise)
    noiseobj = {(choice(noisecols), ij) for ij in noise}
    gi = paint(gi, noiseobj)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (golf-style): each line that contains its side's border colour gets
    # that colour next to the border; afterwards the interior is cleared wholesale.
    # Dots are carried to the border with Move; cleanup (noise, holes, corners)
    # comes LAST: the inner rectangle is painted bgc in one go, then stray cells
    # on the frame ring and the corners.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    c1 = int(I[0, 1])        # top
    c2 = int(I[1, 0])        # left
    c3 = int(I[1, w - 1])    # right
    c4 = int(I[h - 1, 1])    # bottom
    interior_vals = [int(I[r, c]) for r in range(1, h - 1) for c in range(1, w - 1)]
    bgc = Counter(interior_vals).most_common(1)[0][0]

    G = I.copy()             # python-side view of the working grid
    ops, sels = [], []

    # dots live in the inner rectangle rows 2..h-3, cols 2..w-3
    tops, lefts, bottoms, rights = [], [], [], []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            v = int(I[r, c])
            if v == bgc:
                continue
            if v == c1:
                tops.append((r, c))
            elif v == c2:
                lefts.append((r, c))
            elif v == c4:
                bottoms.append((r, c))
            elif v == c3:
                rights.append((r, c))

    def slide(r, c, dr, dc, steps, color):
        if steps <= 0:
            return
        dest = (r + dr * steps, c + dc * steps)
        if color == 0:
            # a 0-coloured dot can't be grabbed by ARCLE Move; mark its border cell
            ops.append(0)
            sels.append(sel_of([dest]))
            G[dest] = 0
            return
        op = 20 if dr < 0 else 21 if dr > 0 else 22 if dc > 0 else 23
        ops.append(op)
        sels.append(sel_of([(r, c)]))       # grab the dot
        for _ in range(steps - 1):
            ops.append(op)
            sels.append(sel_of([]))         # keep sliding the same dot
        G[r, c] = 0                          # grab footprint left at 0
        G[dest] = color

    for (r, c) in sorted(tops, key=lambda p: (p[1], p[0])):
        slide(r, c, -1, 0, r - 1, c1)
    for (r, c) in sorted(lefts, key=lambda p: (p[0], p[1])):
        slide(r, c, 0, -1, c - 1, c2)
    for (r, c) in sorted(bottoms, key=lambda p: (p[1], p[0])):
        slide(r, c, 1, 0, (h - 2) - r, c4)
    for (r, c) in sorted(rights, key=lambda p: (p[0], p[1])):
        slide(r, c, 0, 1, (w - 2) - c, c3)

    # clear the inner rectangle wholesale (noise + vacated dot cells)
    if h >= 5 and w >= 5:
        inner = G[2:h - 2, 2:w - 2]
        if (inner != bgc).any():
            ops.append(bgc)
            sels.append([2, 2, h - 5, w - 5])   # full inner rectangle
            G[2:h - 2, 2:w - 2] = bgc

    # stray cells on the frame ring (not bgc, not a border colour) and corners
    keep = {bgc, c1, c2, c3, c4}
    stray = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r in (1, h - 2) or c in (1, w - 2)) and int(G[r, c]) not in keep:
                stray.append((r, c))
    for (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if int(G[r, c]) != bgc:
            stray.append((r, c))
    if stray:
        ops.append(bgc)
        sels.append(sel_of(stray))
        for rc in stray:
            G[rc] = bgc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
