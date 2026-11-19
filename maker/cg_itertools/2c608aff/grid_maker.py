"""
ARC Task: 2c608aff (RE-ARC) — LLM-generated grid_maker
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

# How this differs from the maker: the maker draws each ray as one finished segment,
# one segment per row and side. The code-golf solution reads the task as growth
# instead. In each sweep, every noise cell that lines up with the box pushes its tip
# one cell toward the box, and the sweeps repeat in all four directions until no tip
# can move. derive_operations follows that reading: each op paints the current
# "front" for one direction. The front is all background cells that sit next to a
# noise-coloured cell, on the side facing the box, in a row or column the box covers.
# The front is worked out from the grid as it stands after the earlier ops, never from O.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return int(min(max(a, round(a + (b - a) * d)), b))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    dcol = random.choice(rem)
    return {"bgc": bgc, "ccol": ccol, "dcol": dcol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccol=1, dcol=2, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = _unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = _unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + boxh, locj:locj + boxw] = ccol
    reminds = [(i, j) for i in range(h) for j in range(w)
               if not (loci - 1 <= i <= loci + boxh and locj - 1 <= j <= locj + boxw)]
    noiseb = max(1, len(reminds) // 4)
    nnoise = _unifint(diff_lb, diff_ub, (1, noiseb))   # at least one noise cell
    noise = random.sample(reminds, nnoise)
    for (i, j) in noise:
        gi[i, j] = dcol
    go = gi.copy()
    for (a, b) in noise:
        if loci <= a < loci + boxh:
            lo, hi_ = sorted((b, locj))
            go[a, lo:hi_ + 1] = dcol
        elif locj <= b < locj + boxw:
            lo, hi_ = sorted((a, loci))
            go[lo:hi_ + 1, b] = dcol
    go[loci:loci + boxh, locj:locj + boxw] = ccol
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    full = [0, 0, hi - 1, wi - 1]

    # box: the colour whose cells exactly fill their bounding rectangle (largest such)
    box = None
    for col in sorted(set(I.flatten().tolist())):
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        if len(rs) == (r1 - r0 + 1) * (c1 - c0 + 1) and r1 > r0 and c1 > c0:
            if box is None or len(rs) > box[0]:
                box = (len(rs), col, int(r0), int(r1), int(c0), int(c1))
    rest = [c for c in set(I.flatten().tolist()) if box is None or c != box[1]]
    if box is None or len(rest) < 2:
        return [34], [full]
    _, ccol, r_lo, r_hi, c_lo, c_hi = box
    # of the other colours: rarest = noise (ray colour), most common = background
    rest.sort(key=lambda c: int((I == c).sum()))
    dcol, bgc = rest[0], rest[-1]

    # Growth reading: sweep the four directions in turn. In each sweep, every ray
    # tip that lines up with the box moves one background cell toward the box.
    # Repeat until no tip can move.
    G = I.copy()
    ops, sels = [], []
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # rays travelling right, down, left, up
    while True:
        progressed = False
        for dr, dc in dirs:
            front = []
            for r in range(hi):
                for c in range(wi):
                    if G[r, c] != bgc:
                        continue
                    pr, pc = r - dr, c - dc
                    if not (0 <= pr < hi and 0 <= pc < wi) or G[pr, pc] != dcol:
                        continue
                    if dc == 1 and r_lo <= r <= r_hi and c < c_lo:
                        front.append((r, c))
                    elif dc == -1 and r_lo <= r <= r_hi and c > c_hi:
                        front.append((r, c))
                    elif dr == 1 and c_lo <= c <= c_hi and r < r_lo:
                        front.append((r, c))
                    elif dr == -1 and c_lo <= c <= c_hi and r > r_hi:
                        front.append((r, c))
            if front:
                ops.append(int(dcol)); sels.append(sel_of(front))   # advance this direction's fronts
                for r, c in front:
                    G[r, c] = dcol
                progressed = True
        if not progressed:
            break

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
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
                                f"for task 2c608aff"
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
                    f"Failed to build a complete episode for task 2c608aff "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2c608aff-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
