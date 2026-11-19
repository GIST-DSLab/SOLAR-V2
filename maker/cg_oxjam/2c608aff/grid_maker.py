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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    dcol = random.choice(rem)
    return {"bgc": bgc, "ccol": ccol, "dcol": dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcol=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if dcol is None:
        dcol = random.choice([c for c in cols if c not in (bgc, ccol)])
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            gi[r][c] = ccol
    reminds = [(r, c) for r in range(h) for c in range(w)
               if not (loci - 1 <= r <= loci + boxh and locj - 1 <= c <= locj + boxw)]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, min(nnoise, len(reminds)))
    for r, c in noise:
        gi[r][c] = dcol
    go = [row[:] for row in gi]
    for a, b in noise:
        if loci <= a < loci + boxh:
            for c in range(min(b, locj), max(b, locj) + 1):
                go[a][c] = dcol
        elif locj <= b < locj + boxw:
            for r in range(min(a, loci), max(a, loci) + 1):
                go[r][b] = dcol
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            go[r][c] = ccol
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    colors = sorted(set(I.flatten().tolist()))

    # box: the colour whose cells exactly fill their bounding rectangle (largest such)
    box = None
    for col in colors:
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh >= 2 and bw >= 2 and len(rs) == bh * bw and len(rs) < hi * wi:
            if box is None or len(rs) > box[0]:
                box = (len(rs), col, int(r0), int(c0), int(r1), int(c1))
    rest = [c for c in colors if box is None or c != box[1]]
    if box is None or len(rest) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]
    _, ccol, r_lo, c_lo, r_hi, c_hi = box
    cnt = Counter(I.flatten().tolist())
    bgc = max(rest, key=lambda c: cnt[c])          # background: dominant non-box colour
    dcol = [c for c in rest if c != bgc][0]        # the scattered noise colour
    G = I.copy()

    def shoot(line):
        # line: cells of one row/column ordered from the grid edge toward the box
        # (box excluded). The first noise cell met fires a ray to the box edge.
        hit = None
        for k, (r, c) in enumerate(line):
            if I[r, c] == dcol:
                hit = k
                break
        if hit is None:
            return
        cells = [(r, c) for (r, c) in line[hit:] if G[r, c] != dcol]
        if not cells:
            return
        ops.append(int(dcol)); sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = dcol

    # Sweep the four sides of the box one at a time (the golf reading: rotate the
    # grid, always shooting rays into the box from one side per pass).
    for r in range(r_lo, r_hi + 1):                       # left side: rays go right
        shoot([(r, c) for c in range(0, c_lo)])
    for c in range(c_lo, c_hi + 1):                       # top side: rays go down
        shoot([(r, c) for r in range(0, r_lo)])
    for r in range(r_lo, r_hi + 1):                       # right side: rays go left
        shoot([(r, c) for c in range(wi - 1, c_hi, -1)])
    for c in range(c_lo, c_hi + 1):                       # bottom side: rays go up
        shoot([(r, c) for r in range(hi - 1, r_hi, -1)])

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
