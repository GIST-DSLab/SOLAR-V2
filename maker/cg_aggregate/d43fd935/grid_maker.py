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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None, **kwargs) -> dict:
    from random import randint, sample, choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        hi = max(hi, lo)
        return randint(lo, hi)

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    if ccol is None or ccol == bgc:
        ccol = choice([c for c in cols if c != bgc])
    if not dcols:
        rem = [c for c in cols if c not in (bgc, ccol)]
        dcols = sample(rem, randint(1, 8))
    dcols = list(dcols)

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
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
    truenoise = [ij for ij in noise
                 if (ij[0] < loci or ij[0] > loci + boxh - 1)
                 and (ij[1] < locj or ij[1] > locj + boxw - 1)]
    tn = set(truenoise)
    rem = [ij for ij in noise if ij not in tn]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]
    for (r, c) in truenoise:
        gi[r][c] = choice(dcols)
    go = [row[:] for row in gi]

    for jj in sorted(set(c for r, c in top)):
        col = choice(dcols)
        subs = [(r, c) for r, c in top if c == jj]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for r in range(min(r for r, _ in subs), loci):
            go[r][jj] = col
    for jj in sorted(set(c for r, c in bottom)):
        col = choice(dcols)
        subs = [(r, c) for r, c in bottom if c == jj]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for r in range(loci + boxh, max(r for r, _ in subs) + 1):
            go[r][jj] = col
    for ii in sorted(set(r for r, c in left)):
        col = choice(dcols)
        subs = [(r, c) for r, c in left if r == ii]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for c in range(min(c for _, c in subs), locj):
            go[ii][c] = col
    for ii in sorted(set(r for r, c in right)):
        col = choice(dcols)
        subs = [(r, c) for r, c in right if r == ii]
        for r, c in subs:
            gi[r][c] = col
            go[r][c] = col
        for c in range(locj + boxw, max(c for _, c in subs) + 1):
            go[ii][c] = col

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry; they agree): every marker
    # lying in the box's row/column band is dragged toward the box, so each band
    # line becomes one solid line of the marker colour from the farthest marker
    # up to the box face. One Color op per band line, sides handled in turn.
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def rect_components(G, bg):
        G = np.asarray(G, dtype=int)
        h, w = G.shape
        seen = np.zeros_like(G, dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or G[r, c] == bg:
                    continue
                col = int(G[r, c])
                st = [(r, c)]
                cells = []
                seen[r, c] = True
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and G[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                rs = [y for y, _ in cells]
                cs = [x for _, x in cells]
                bh = max(rs) - min(rs) + 1
                bw = max(cs) - min(cs) + 1
                if bh * bw == len(cells) and bh >= 2 and bw >= 2:
                    out.append((len(cells), col, min(rs), min(cs), bh, bw))
        return out

    # Box colour: the colour the demonstrations use for the big rectangle.
    box_col = None
    if examples:
        votes = Counter()
        for ex in examples:
            EI = np.asarray(ex[0], dtype=int)
            ebg = Counter(EI.flatten().tolist()).most_common(1)[0][0]
            comps = rect_components(EI, ebg)
            if comps:
                votes[max(comps)[1]] += 1
        if votes:
            box_col = votes.most_common(1)[0][0]

    box = None
    if box_col is not None and box_col != bgc:
        pts = np.argwhere(I == box_col)
        if len(pts):
            r0, c0 = pts.min(0)
            r1, c1 = pts.max(0)
            if (r1 - r0 + 1) * (c1 - c0 + 1) == len(pts):
                box = (int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1))
    if box is None:
        comps = rect_components(I, bgc)
        if comps:
            best = max(comps, key=lambda t: t[0])
            box = (best[2], best[3], best[4], best[5])

    def emit_line(color, cells):
        if all(int(I[r, c]) == color for r, c in cells):
            return  # already solid: op would change nothing
        ops.append(color)
        sels.append(sel_of(cells))

    if box is not None:
        br, bc, bh, bw = box
        # columns above the box
        for j in range(bc, bc + bw):
            rows = [r for r in range(br) if I[r, j] != bgc]
            if rows:
                far = min(rows)
                emit_line(int(I[far, j]), [(r, j) for r in range(far, br)])
        # columns below the box
        for j in range(bc, bc + bw):
            rows = [r for r in range(br + bh, hi) if I[r, j] != bgc]
            if rows:
                far = max(rows)
                emit_line(int(I[far, j]), [(r, j) for r in range(br + bh, far + 1)])
        # rows left of the box
        for i in range(br, br + bh):
            cs = [c for c in range(bc) if I[i, c] != bgc]
            if cs:
                far = min(cs)
                emit_line(int(I[i, far]), [(i, c) for c in range(far, bc)])
        # rows right of the box
        for i in range(br, br + bh):
            cs = [c for c in range(bc + bw, wi) if I[i, c] != bgc]
            if cs:
                far = max(cs)
                emit_line(int(I[i, far]), [(i, c) for c in range(bc + bw, far + 1)])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
