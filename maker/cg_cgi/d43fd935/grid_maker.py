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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)
    rem = [c for c in rem if c != ccol]
    nd = random.randint(1, 8)
    dcols = random.sample(rem, nd)
    return {"bgc": bgc, "ccol": ccol, "dcols": dcols}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * max(0.0, min(1.0, lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, ub))))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None or ccol == bgc:
        ccol = random.choice([c for c in cols if c != bgc])
    if not dcols:
        dcols = random.sample([c for c in cols if c not in (bgc, ccol)], random.randint(1, 8))
    dcols = [c for c in dcols if c not in (bgc, ccol)]
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = _unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = _unifint(diff_lb, diff_ub, (2, w // 2))
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
    nnoise = _unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, nnoise)
    inrow = lambda r: loci <= r <= loci + boxh - 1
    incol = lambda c: locj <= c <= locj + boxw - 1
    truenoise = [(r, c) for r, c in noise if not inrow(r) and not incol(c)]
    rem = [ij for ij in noise if inrow(ij[0]) or incol(ij[1])]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]
    for r, c in truenoise:
        gi[r][c] = random.choice(dcols)
    go = [row[:] for row in gi]
    for jj in sorted(set(c for r, c in top)):
        col = random.choice(dcols)
        subs = [r for r, c in top if c == jj]
        for r in subs:
            gi[r][jj] = col
        for r in range(min(subs), loci):
            go[r][jj] = col
    for jj in sorted(set(c for r, c in bottom)):
        col = random.choice(dcols)
        subs = [r for r, c in bottom if c == jj]
        for r in subs:
            gi[r][jj] = col
        for r in range(loci + boxh, max(subs) + 1):
            go[r][jj] = col
    for ii in sorted(set(r for r, c in left)):
        col = random.choice(dcols)
        subs = [c for r, c in left if r == ii]
        for c in subs:
            gi[ii][c] = col
        for c in range(min(subs), locj):
            go[ii][c] = col
    for ii in sorted(set(r for r, c in right)):
        col = random.choice(dcols)
        subs = [c for r, c in right if r == ii]
        for c in subs:
            gi[ii][c] = col
        for c in range(locj + boxw, max(subs) + 1):
            go[ii][c] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry, and different from the maker's single
    # far-marker-to-box line): every marker in a lane facing the box shoots its own
    # ray toward the box, and the ray stops at the next non-background cell (the next
    # marker, or the box). Lanes are swept from the far edge toward the box.
    # Ops depend only on I, never on O.
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # The box is the largest colour whose cells all fill one solid rectangle (at least 2x2).
    box = None
    best = 0
    for col in set(I.flatten().tolist()):
        if col == bgc:
            continue
        rs, cs = np.nonzero(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh >= 2 and bw >= 2 and bh * bw == len(rs) and len(rs) > best:
            best = len(rs)
            box = (int(r0), int(c0), int(bh), int(bw))

    if box is not None:
        br, bc, bh, bw = box

        def sweep(lane):
            # lane: cells ordered from the far edge toward the box (box not included)
            idx = [k for k, (r, c) in enumerate(lane) if int(I[r, c]) != bgc]
            for n, k in enumerate(idx):
                stop = idx[n + 1] if n + 1 < len(idx) else len(lane)
                seg = lane[k + 1:stop]   # background cells between this marker and the next obstacle
                if seg:
                    r, c = lane[k]
                    ops.append(int(I[r, c]))
                    sels.append(sel_of(seg))

        for j in range(bc, bc + bw):                       # columns above the box
            sweep([(r, j) for r in range(0, br)])
        for j in range(bc, bc + bw):                       # columns below the box
            sweep([(r, j) for r in range(hi - 1, br + bh - 1, -1)])
        for i in range(br, br + bh):                       # rows left of the box
            sweep([(i, c) for c in range(0, bc)])
        for i in range(br, br + bh):                       # rows right of the box
            sweep([(i, c) for c in range(wi - 1, bc + bw - 1, -1)])

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
