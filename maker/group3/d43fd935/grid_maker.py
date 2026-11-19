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
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


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
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if not dcols:
        rem = [c for c in cols if c not in (bgc, ccol)]
        dcols = random.sample(rem, random.randint(1, 8))
    dcols = list(dcols)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = _unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = _unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    gi = [[bgc] * w for _ in range(h)]
    bd = set((i, j) for i in range(loci, loci + boxh) for j in range(locj, locj + boxw))
    for (i, j) in bd:
        gi[i][j] = ccol
    reminds = [(i, j) for i in range(h) for j in range(w) if (i, j) not in bd]
    noiseb = max(1, len(reminds) // 4)
    nnoise = _unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, nnoise)

    def outside_rows(i):
        return i < loci or i > loci + boxh - 1

    def outside_cols(j):
        return j < locj or j > locj + boxw - 1

    truenoise = [ij for ij in noise if outside_rows(ij[0]) and outside_cols(ij[1])]
    tn = set(truenoise)
    rem = [ij for ij in noise if ij not in tn]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]
    for (i, j) in truenoise:
        gi[i][j] = random.choice(dcols)
    go = [row[:] for row in gi]
    for jj in sorted(set(j for _, j in top)):
        col = random.choice(dcols)
        subs = [ij for ij in top if ij[1] == jj]
        for (i, j) in subs:
            gi[i][j] = col
        for i in range(min(i for i, _ in subs), loci):
            go[i][jj] = col
    for jj in sorted(set(j for _, j in bottom)):
        col = random.choice(dcols)
        subs = [ij for ij in bottom if ij[1] == jj]
        for (i, j) in subs:
            gi[i][j] = col
        for i in range(loci + boxh, max(i for i, _ in subs) + 1):
            go[i][jj] = col
    for ii in sorted(set(i for i, _ in left)):
        col = random.choice(dcols)
        subs = [ij for ij in left if ij[0] == ii]
        for (i, j) in subs:
            gi[i][j] = col
        for j in range(min(j for _, j in subs), locj):
            go[ii][j] = col
    for ii in sorted(set(i for i, _ in right)):
        col = random.choice(dcols)
        subs = [ij for ij in right if ij[0] == ii]
        for (i, j) in subs:
            gi[i][j] = col
        for j in range(locj + boxw, max(j for _, j in subs) + 1):
            go[ii][j] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = int(np.bincount(I.flatten()).argmax())

    # The box: a colour whose cells together form one full rectangle (>=2x2); largest such.
    box = None
    best = 0
    for col in np.unique(I):
        col = int(col)
        if col == bgc:
            continue
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh >= 2 and bw >= 2 and bh * bw == len(rs) and len(rs) > best:
            best = len(rs)
            box = (int(r0), int(c0), int(bh), int(bw))

    if box is not None:
        br, bc, bh, bw = box
        # Collect each pixel-aligned ray (from the farthest aligned pixel up to the box),
        # grouped by its colour: one Color op paints every ray of one colour at once.
        rays = {}
        order = []

        def add(color, cells):
            if color not in rays:
                rays[color] = []
                order.append(color)
            rays[color].extend(cells)

        for j in range(bc, bc + bw):
            rows = [r for r in range(br) if I[r, j] != bgc]
            if rows:
                add(int(I[rows[0], j]), [(r, j) for r in range(rows[0], br)])
            rows = [r for r in range(br + bh, hi) if I[r, j] != bgc]
            if rows:
                add(int(I[rows[-1], j]), [(r, j) for r in range(br + bh, rows[-1] + 1)])
        for i in range(br, br + bh):
            cs_ = [c for c in range(bc) if I[i, c] != bgc]
            if cs_:
                add(int(I[i, cs_[0]]), [(i, c) for c in range(cs_[0], bc)])
            cs_ = [c for c in range(bc + bw, wi) if I[i, c] != bgc]
            if cs_:
                add(int(I[i, cs_[-1]]), [(i, c) for c in range(bc + bw, cs_[-1] + 1)])

        for color in order:
            cells = sorted(set(rays[color]))
            if all(int(I[r, c]) == color for r, c in cells):
                continue  # this colour's rays are already complete
            ops.append(color)
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
