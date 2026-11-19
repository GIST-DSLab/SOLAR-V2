"""
ARC Task: d43fd935 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import unifint
from random import randint, choice, sample


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    nd = random.randint(1, 8)
    dcols = random.sample(rem, nd)
    return {"bgc": bgc, "ccol": ccol, "dcols": dcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccol is None:
        ccol = choice(remove(bgc, cols))
    if dcols is None:
        rc = [c for c in cols if c not in (bgc, ccol)]
        dcols = sample(rc, unifint(diff_lb, diff_ub, (1, 8)))
    dcols = tuple(dcols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - bd)
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    truenoise = sfilter(noise, lambda ij: (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1))
    rem = difference(noise, truenoise)
    top = sfilter(rem, lambda ij: ij[0] < loci)
    bottom = sfilter(rem, lambda ij: ij[0] > loci + boxh - 1)
    left = sfilter(rem, lambda ij: ij[1] < locj)
    right = sfilter(rem, lambda ij: ij[1] > locj + boxw - 1)
    truenoiseobj = {(choice(dcols), ij) for ij in truenoise}
    gi = paint(gi, truenoiseobj)
    go = tuple(e for e in gi)
    for jj in apply(last, top):
        col = choice(dcols)
        subs = sfilter(top, matcher(last, jj))
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmin(subs, first), jj), (loci - 1, jj)))
    for jj in apply(last, bottom):
        col = choice(dcols)
        subs = sfilter(bottom, matcher(last, jj))
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmax(subs, first), jj), (loci + boxh, jj)))
    for ii in apply(first, left):
        col = choice(dcols)
        subs = sfilter(left, matcher(first, ii))
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmin(subs, last)), (ii, locj - 1)))
    for ii in apply(first, right):
        col = choice(dcols)
        subs = sfilter(right, matcher(first, ii))
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmax(subs, last)), (ii, locj + boxw)))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (following the golf solution, which differs from the maker's):
    # the unit is not "one line per row/column from the farthest marker to the box",
    # but "each marker grows a ray toward the box until it hits something".
    # The nearest marker grows until it reaches the box. Each farther marker grows
    # until it reaches the next nearer marker. So every marker gets its own segment,
    # in its own colour, ordered from the box outward. Each selection is the run of
    # background cells in I between a marker and its next obstacle toward the box.
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]

    # Box: a solid rectangle (>=2x2) whose colour appears nowhere else in I.
    visited = np.zeros_like(I, dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c]); stack = [(r, c)]; cells = []
            visited[r, c] = True
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == col:
                        visited[ny, nx] = True; stack.append((ny, nx))
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            bh = max(rs) - min(rs) + 1; bw = max(cs) - min(cs) + 1
            if bh >= 2 and bw >= 2 and bh * bw == len(cells):
                key = (counts[col] == len(cells), len(cells))
                if best is None or key > best[0]:
                    best = (key, min(rs), min(cs), bh, bw)

    if best is not None:
        _, br, bc, bh, bw = best
        lines = []
        for j in range(bc, bc + bw):
            lines.append([(r, j) for r in range(br - 1, -1, -1)])      # above the box
        for j in range(bc, bc + bw):
            lines.append([(r, j) for r in range(br + bh, hi)])         # below the box
        for i in range(br, br + bh):
            lines.append([(i, c) for c in range(bc - 1, -1, -1)])      # left of the box
        for i in range(br, br + bh):
            lines.append([(i, c) for c in range(bc + bw, wi)])         # right of the box
        for line in lines:
            # walk outward from the box; each marker grows over the gap in front of it
            gap = []
            for (y, x) in line:
                v = int(I[y, x])
                if v == bgc:
                    gap.append((y, x))
                else:
                    if gap:
                        ops.append(v); sels.append(sel_of(gap))
                    gap = []

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d43fd935"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d43fd935"
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
                                f"for task d43fd935"
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
                    f"Failed to build a complete episode for task d43fd935 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d43fd935-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
