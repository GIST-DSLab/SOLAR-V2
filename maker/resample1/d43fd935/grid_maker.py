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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    ndcols = random.randint(1, 8)
    dcols = random.sample(rem, ndcols)
    return {"bgc": bgc, "ccol": ccol, "dcols": dcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi_ = a + int(round((b - a) * ub))
        lo = max(a, min(lo, b))
        hi_ = max(lo, min(hi_, b))
        return random.randint(lo, hi_)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if dcols is None:
        rem = [c for c in cols if c not in (bgc, ccol)]
        dcols = random.sample(rem, random.randint(1, 8))
    H = max(10, max_h)
    W = max(10, max_w)
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    gi = [[bgc] * w for _ in range(h)]
    bd = set()
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            gi[r][c] = ccol
            bd.add((r, c))
    reminds = [(r, c) for r in range(h) for c in range(w) if (r, c) not in bd]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, nnoise)
    inrow = lambda r: loci <= r <= loci + boxh - 1
    incol = lambda c: locj <= c <= locj + boxw - 1
    truenoise = [ij for ij in noise if not inrow(ij[0]) and not incol(ij[1])]
    remn = [ij for ij in noise if inrow(ij[0]) or incol(ij[1])]
    top = [ij for ij in remn if ij[0] < loci]
    bottom = [ij for ij in remn if ij[0] > loci + boxh - 1]
    left = [ij for ij in remn if ij[1] < locj]
    right = [ij for ij in remn if ij[1] > locj + boxw - 1]
    for (r, c) in truenoise:
        gi[r][c] = random.choice(dcols)
    go = [row[:] for row in gi]
    for jj in sorted(set(c for _, c in top)):
        col = random.choice(dcols)
        subs = [ij for ij in top if ij[1] == jj]
        for (r, c) in subs:
            gi[r][c] = col
        for r in range(min(s[0] for s in subs), loci):
            go[r][jj] = col
    for jj in sorted(set(c for _, c in bottom)):
        col = random.choice(dcols)
        subs = [ij for ij in bottom if ij[1] == jj]
        for (r, c) in subs:
            gi[r][c] = col
        for r in range(loci + boxh, max(s[0] for s in subs) + 1):
            go[r][jj] = col
    for ii in sorted(set(r for r, _ in left)):
        col = random.choice(dcols)
        subs = [ij for ij in left if ij[0] == ii]
        for (r, c) in subs:
            gi[r][c] = col
        for c in range(min(s[1] for s in subs), locj):
            go[ii][c] = col
    for ii in sorted(set(r for r, _ in right)):
        col = random.choice(dcols)
        subs = [ij for ij in right if ij[0] == ii]
        for (r, c) in subs:
            gi[r][c] = col
        for c in range(locj + boxw, max(s[1] for s in subs) + 1):
            go[ii][c] = col
    # final output paints lines over gi state (input colors of line cells are col)
    for r in range(h):
        for c in range(w):
            if gi[r][c] != go[r][c] and go[r][c] == bgc:
                go[r][c] = gi[r][c]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Box: largest colour class (non-bgc) that forms a solid rectangle
    best = None
    for col in set(I.flatten().tolist()):
        if col == bgc:
            continue
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == len(rs) and area >= 4:
            if best is None or area > best[0]:
                best = (area, r0, r1, c0, c1)
    _, r0, r1, c0, c1 = best

    G = I.copy()
    ops, sels = [], []

    def paint_line(cells):
        cells = [(int(r), int(c)) for r, c in cells]
        if not cells:
            return
        # colour is that of the stray cell(s) on this line
        col = None
        for (r, c) in cells:
            if G[r, c] != bgc:
                col = int(G[r, c])
                break
        if col is None:
            return
        todo = [(r, c) for (r, c) in cells if G[r, c] != col]
        if not todo:
            return
        ops.append(col)
        sels.append(sel_of(todo))
        for (r, c) in todo:
            G[r, c] = col

    # Top lines
    for j in range(c0, c1 + 1):
        rows = [r for r in range(0, r0) if I[r, j] != bgc]
        if rows:
            paint_line([(r, j) for r in range(r0 - 1, min(rows) - 1, -1)])
    # Bottom lines
    for j in range(c0, c1 + 1):
        rows = [r for r in range(r1 + 1, hi) if I[r, j] != bgc]
        if rows:
            paint_line([(r, j) for r in range(r1 + 1, max(rows) + 1)])
    # Left lines
    for i in range(r0, r1 + 1):
        cs_ = [c for c in range(0, c0) if I[i, c] != bgc]
        if cs_:
            paint_line([(i, c) for c in range(c0 - 1, min(cs_) - 1, -1)])
    # Right lines
    for i in range(r0, r1 + 1):
        cs_ = [c for c in range(c1 + 1, wi) if I[i, c] != bgc]
        if cs_:
            paint_line([(i, c) for c in range(c1 + 1, max(cs_) + 1)])

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
