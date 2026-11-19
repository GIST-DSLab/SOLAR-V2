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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccol = random.choice(remcols)
    remcols = [c for c in remcols if c != ccol]
    ndcols = random.randint(1, 8)
    dcols = random.sample(remcols, ndcols)
    return {"bgc": bgc, "ccol": ccol, "dcols": dcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if ccol is None:
        ccol = choice([c for c in cols if c != bgc])
    if dcols is None:
        rem = [c for c in cols if c not in (bgc, ccol)]
        dcols = sample(rem, randint(1, 8))
    dcols = list(dcols)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)

    gi = [[bgc] * w for _ in range(h)]
    bd = set()
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            gi[r][c] = ccol
            bd.add((r, c))
    reminds = [(r, c) for r in range(h) for c in range(w) if (r, c) not in bd]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)

    def outside(ij):
        return (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1)

    truenoise = [ij for ij in noise if outside(ij)]
    rem = [ij for ij in noise if not outside(ij)]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]
    for (r, c) in truenoise:
        gi[r][c] = choice(dcols)
    go = [row[:] for row in gi]

    for jj in sorted(set(c for _, c in top)):
        col = choice(dcols)
        subs = [(r, c) for r, c in top if c == jj]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for r in range(min(r for r, _ in subs), loci):
            go[r][jj] = col
    for jj in sorted(set(c for _, c in bottom)):
        col = choice(dcols)
        subs = [(r, c) for r, c in bottom if c == jj]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for r in range(loci + boxh, max(r for r, _ in subs) + 1):
            go[r][jj] = col
    for ii in sorted(set(r for r, _ in left)):
        col = choice(dcols)
        subs = [(r, c) for r, c in left if r == ii]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for c in range(min(c for _, c in subs), locj):
            go[ii][c] = col
    for ii in sorted(set(r for r, _ in right)):
        col = choice(dcols)
        subs = [(r, c) for r, c in right if r == ii]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for c in range(locj + boxw, max(c for _, c in subs) + 1):
            go[ii][c] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background: the canvas colour (the box covers <= 1/4, noise <= 1/4 of the rest).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # The box: the largest colour class that exactly fills its own bounding rectangle.
    box = None
    best = 0
    for col in sorted(set(I.flatten().tolist())):
        if col == bgc:
            continue
        pts = np.argwhere(I == col)
        r0, c0 = pts.min(0)
        r1, c1 = pts.max(0)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if len(pts) == bh * bw and bh >= 2 and bw >= 2 and len(pts) > best:
            best = len(pts)
            box = (int(r0), int(c0), int(bh), int(bw))
    if box is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    br, bc, bh, bw = box

    # Reading: each pixel lying in the box's rows/columns falls toward the box and
    # drags its colour along until it lands on the next thing in its path (a nearer
    # pixel, or the box itself). Pixels are handled nearest-to-box first, outward.
    def segment(color, cells):
        if cells:
            ops.append(color); sels.append(sel_of(cells))

    # Above the box: pixels drop downward.
    for j in range(bc, bc + bw):
        stop = br
        for r in range(br - 1, -1, -1):
            if I[r, j] != bgc:
                segment(int(I[r, j]), [(y, j) for y in range(r + 1, stop)])
                stop = r
    # Below the box: pixels rise upward.
    for j in range(bc, bc + bw):
        stop = br + bh - 1
        for r in range(br + bh, hi):
            if I[r, j] != bgc:
                segment(int(I[r, j]), [(y, j) for y in range(stop + 1, r)])
                stop = r
    # Left of the box: pixels slide right.
    for i in range(br, br + bh):
        stop = bc
        for c in range(bc - 1, -1, -1):
            if I[i, c] != bgc:
                segment(int(I[i, c]), [(i, x) for x in range(c + 1, stop)])
                stop = c
    # Right of the box: pixels slide left.
    for i in range(br, br + bh):
        stop = bc + bw - 1
        for c in range(bc + bw, wi):
            if I[i, c] != bgc:
                segment(int(I[i, c]), [(i, x) for x in range(stop + 1, c)])
                stop = c

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
