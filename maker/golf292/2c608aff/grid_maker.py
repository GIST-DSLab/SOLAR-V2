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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    dcol = random.choice(rem)
    return {"bgc": bgc, "ccol": ccol, "dcol": dcol}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return int(round(a + (b - a) * d))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcol=None, **kw) -> dict:
    if bgc is None or ccol is None or dcol is None:
        c = sample_colors()
        bgc, ccol, dcol = c["bgc"], c["ccol"], c["dcol"]
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = _unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = _unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            gi[r][c] = ccol
    reminds = [(r, c) for r in range(h) for c in range(w)
               if not (loci - 1 <= r <= loci + boxh and locj - 1 <= c <= locj + boxw)]
    noiseb = max(1, len(reminds) // 4)
    nnoise = _unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, min(nnoise, len(reminds)))
    for r, c in noise:
        gi[r][c] = dcol
    go = [row[:] for row in gi]
    for a, b in noise:
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
    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]

    # the box: colour whose cells form the largest solid rectangle (>=2x2)
    box = None
    for col in sorted(set(I.flatten().tolist())):
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh >= 2 and bw >= 2 and len(rs) == bh * bw:
            if box is None or len(rs) > box[0]:
                box = (len(rs), col, r0, c0, bh, bw)
    if box is None:
        return [34], [full]
    _, ccol, loci, locj, boxh, boxw = box
    rest = [c for c in set(I.flatten().tolist()) if c != ccol]
    if len(rest) < 2:
        return [34], [full]
    rest.sort(key=lambda c: int((I == c).sum()))
    dcol = rest[0]
    r_lo, r_hi = loci, loci + boxh - 1
    c_lo, c_hi = locj, locj + boxw - 1

    def emit_runs(spans, horizontal):
        # spans: list of (line_index, a, b) sorted by line_index; merge
        # consecutive lines with identical [a, b] into one rectangle stroke
        i = 0
        while i < len(spans):
            k0, a, b = spans[i]
            j = i
            while j + 1 < len(spans) and spans[j + 1][0] == spans[j][0] + 1 \
                    and spans[j + 1][1:] == (a, b):
                j += 1
            k1 = spans[j][0]
            if horizontal:
                cells = [(r, c) for r in range(k0, k1 + 1) for c in range(a, b + 1)]
            else:
                cells = [(r, c) for c in range(k0, k1 + 1) for r in range(a, b + 1)]
            ops.append(int(dcol)); sels.append(sel_of(cells))
            i = j + 1

    left, right, up, down = [], [], [], []
    for r in range(r_lo, r_hi + 1):
        L = [c for c in range(0, c_lo) if I[r, c] == dcol]
        if L and min(L) < c_lo - 1 or (L and I[r, c_lo - 1] != dcol):
            left.append((r, min(L), c_lo - 1))
        R = [c for c in range(c_hi + 1, wi) if I[r, c] == dcol]
        if R:
            right.append((r, c_hi + 1, max(R)))
    for c in range(c_lo, c_hi + 1):
        U = [r for r in range(0, r_lo) if I[r, c] == dcol]
        if U:
            up.append((c, min(U), r_lo - 1))
        D = [r for r in range(r_hi + 1, hi) if I[r, c] == dcol]
        if D:
            down.append((c, r_hi + 1, max(D)))

    # drop spans that are already entirely dcol (no visible effect)
    def needed(spans, horizontal):
        out = []
        for k, a, b in spans:
            seg = I[k, a:b + 1] if horizontal else I[a:b + 1, k]
            if np.any(seg != dcol):
                out.append((k, a, b))
        return out

    emit_runs(needed(left, True), True)
    emit_runs(needed(right, True), True)
    emit_runs(needed(up, False), False)
    emit_runs(needed(down, False), False)

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
