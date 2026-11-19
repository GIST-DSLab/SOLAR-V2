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
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if dcol is None:
        dcol = random.choice([c for c in cols if c not in (bgc, ccol)])

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

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
    # cells outside the box's outbox backdrop
    reminds = [(r, c) for r in range(h) for c in range(w)
               if not (loci - 1 <= r <= loci + boxh and locj - 1 <= c <= locj + boxw)]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    nnoise = min(nnoise, len(reminds))
    noise = random.sample(reminds, nnoise)
    for r, c in noise:
        gi[r][c] = dcol
    go = [row[:] for row in gi]
    for (a, b) in noise:
        if loci <= a < loci + boxh:
            lo, hi_ = sorted((b, locj))
            for c in range(lo, hi_ + 1):
                go[a][c] = dcol
        elif locj <= b < locj + boxw:
            lo, hi_ = sorted((a, loci))
            for r in range(lo, hi_ + 1):
                go[r][b] = dcol
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            go[r][c] = ccol
    return {"input": [tuple(r) for r in gi], "output": [tuple(r) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    full = [0, 0, hi - 1, wi - 1]
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    others = [c for c in cnt if c != bgc]

    def solid_box(col):
        rs, cs = np.where(I == col)
        if len(rs) == 0:
            return None
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh >= 2 and bw >= 2 and len(rs) == bh * bw:
            return (int(r0), int(c0), int(r1), int(c1))
        return None

    # noise colour convention: learned from demonstrations (the colour that grows)
    dcol = None
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            for c in range(10):
                if (eo == c).sum() > (ei == c).sum():
                    dcol = c
                    break
            if dcol is not None:
                break
    ccol = None
    if dcol is not None and dcol in others:
        cand = [c for c in others if c != dcol and solid_box(c) is not None]
        if cand:
            ccol = max(cand, key=lambda c: cnt[c])
    if ccol is None:
        cand = [c for c in others if solid_box(c) is not None]
        if not cand:
            return [34], [full]
        ccol = max(cand, key=lambda c: cnt[c])
        rest = [c for c in others if c != ccol]
        if not rest:
            return [34], [full]
        dcol = min(rest, key=lambda c: cnt[c])
    if dcol not in others:
        return [34], [full]

    r0, c0, r1, c1 = solid_box(ccol)
    G = I.copy()
    noise = [(int(r), int(c)) for r, c in zip(*np.where(I == dcol))]

    # each aligned noise cell gravitates toward the box: a ray from the cell,
    # stepping toward the box until it hits the box or something non-background.
    # Per side, nearest-first, so each outer ray links onto the one inside it.
    sides = [
        ([p for p in noise if r0 <= p[0] <= r1 and p[1] < c0], lambda p: c0 - p[1], (0, 1)),
        ([p for p in noise if r0 <= p[0] <= r1 and p[1] > c1], lambda p: p[1] - c1, (0, -1)),
        ([p for p in noise if c0 <= p[1] <= c1 and p[0] < r0], lambda p: r0 - p[0], (1, 0)),
        ([p for p in noise if c0 <= p[1] <= c1 and p[0] > r1], lambda p: p[0] - r1, (-1, 0)),
    ]
    for cells, dist, (dr, dc) in sides:
        for (r, c) in sorted(cells, key=dist):
            ray = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < hi and 0 <= cc < wi and not (r0 <= rr <= r1 and c0 <= cc <= c1) \
                    and G[rr, cc] == bgc:
                ray.append((rr, cc))
                rr += dr; cc += dc
            if ray:
                ops.append(int(dcol)); sels.append(sel_of(ray))
                for p in ray:
                    G[p] = dcol

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
