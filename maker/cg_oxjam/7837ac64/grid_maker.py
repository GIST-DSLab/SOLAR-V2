"""
ARC Task: 7837ac64 (RE-ARC) — LLM-generated grid_maker
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

# Note on the two readings: the maker and the golf entry use the same rule. A block's
# colour is the marker colour on which all four of its corner intersections agree.
# The difference is the unit of work. The golf entry pairs consecutive lattice-line
# rows and compares them element by element, then pairs consecutive columns of that
# result. So each output row comes from one PAIR of adjacent horizontal lines.
# derive_operations below builds the output that way: one row-pair at a time, and
# each maximal stretch of agreeing colour along the pair is painted as one run.
# The maker instead paints connected components one cell at a time.


def _unif(diff_lb, diff_ub, rng):
    a, b = rng
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo, hi = int(round(lo)), int(round(hi))
    lo = max(a, min(b, lo)); hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    rem = [c for c in cols if c not in (bgc, linc)]
    numcols = random.randint(1, 8)
    ccols = random.sample(rem, numcols)
    return {"bgc": bgc, "linc": linc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, ccols=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    if not ccols:
        rem = [c for c in cols if c not in (bgc, linc)]
        ccols = random.sample(rem, _unif(diff_lb, diff_ub, (1, 8)))
    ccols = list(ccols)
    oh = _unif(diff_lb, diff_ub, (2, 6))
    ow = _unif(diff_lb, diff_ub, (2, 6))
    go = [[bgc] * ow for _ in range(oh)]
    full = {(i, j) for i in range(oh) for j in range(ow)}
    inds = set(full)

    def nbrs(ij):
        i, j = ij
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def ofcolor(g, c):
        return {(i, j) for i in range(len(g)) for j in range(len(g[0])) if g[i][j] == c}

    nocc = _unif(diff_lb, diff_ub, (1, oh * ow))
    for _ in range(nocc):
        mpr = []
        for cc in ccols:
            oc = ofcolor(go, cc)
            cand = set(inds)
            for p in oc:
                cand |= nbrs(p)
            ok = inds | oc
            locs = [ij for ij in cand if (nbrs(ij) & full) <= ok]
            if locs:
                mpr.append((cc, locs))
        if not mpr:
            break
        col, locs = random.choice(mpr)
        loc = random.choice(sorted(locs))
        if loc not in full:
            continue
        go[loc[0]][loc[1]] = col
        inds.discard(loc)
    nz = [(i, j) for i in range(oh) for j in range(ow) if go[i][j] != bgc]
    r0 = min(i for i, _ in nz); r1 = max(i for i, _ in nz)
    c0 = min(j for _, j in nz); c1 = max(j for _, j in nz)
    go = [row[c0:c1 + 1] for row in go[r0:r1 + 1]]
    oh, ow = len(go), len(go[0])
    sqh = _unif(diff_lb, diff_ub, (2, (max_h - oh - 1) // oh))
    sqw = _unif(diff_lb, diff_ub, (2, (max_w - ow - 1) // ow))
    fullh = oh + 1 + oh * sqh
    fullw = ow + 1 + ow * sqw
    if fullh > max_h or fullw > max_w:
        raise ValueError("too big")
    gi = [[linc] * fullw for _ in range(fullh)]
    for i in range(oh):
        for j in range(ow):
            br = 1 + i * (sqh + 1); bc = 1 + j * (sqw + 1)
            for r in range(br, br + sqh):
                for c in range(bc, bc + sqw):
                    gi[r][c] = bgc
            if go[i][j] != bgc:
                for r, c in ((br - 1, bc - 1), (br - 1, bc + sqw), (br + sqh, bc - 1), (br + sqh, bc + sqw)):
                    gi[r][c] = go[i][j]
    gih = _unif(diff_lb, diff_ub, (fullh, max_h))
    giw = _unif(diff_lb, diff_ub, (fullw, max_w))
    loci = random.randint(0, gih - fullh)
    locj = random.randint(0, giw - fullw)
    G = [[bgc] * giw for _ in range(gih)]
    rows = set(range(loci, gih, sqh + 1)) | set(range(loci, -1, -sqh - 1))
    colsl = set(range(locj, giw, sqw + 1)) | set(range(locj, -1, -sqw - 1))
    for a in rows:
        for c in range(giw):
            G[a][c] = linc
    for b in colsl:
        for r in range(gih):
            G[r][b] = linc
    for r in range(fullh):
        for c in range(fullw):
            G[loci + r][locj + c] = gi[r][c]
    gop = {v for row in go for v in row}
    while True:
        go2 = [row[:] for row in go]
        for c in set(ccols) & gop:
            for i in range(oh):
                for j in range(ow):
                    if j + 2 < ow and go[i][j] == c and go[i][j + 1] == bgc and go[i][j + 2] == c:
                        go2[i][j + 1] = c
                    if i + 2 < oh and go[i][j] == c and go[i + 1][j] == bgc and go[i + 2][j] == c:
                        go2[i + 1][j] = c
        if go2 == go:
            break
        go = go2
    # reject layouts whose corner markers are ambiguous (a bgc block whose four
    # corners all carry one colour borrowed from diagonal neighbours)
    for i in range(oh):
        for j in range(ow):
            br = loci + i * (sqh + 1); bc = locj + j * (sqw + 1)
            vals = {G[br][bc], G[br][bc + sqw + 1], G[br + sqh + 1][bc], G[br + sqh + 1][bc + sqw + 1]}
            v = vals.pop() if len(vals) == 1 else None
            want = v if v is not None and v != linc else bgc
            if want != go[i][j]:
                raise ValueError("ambiguous corners")
    return {"input": G, "output": go}


def derive_operations(I, O=None, examples=None):
    # Consecutive lattice-line rows are taken as a PAIR. Along the pair, intersections
    # that carry the same marker colour agree. That agreement is then paired again
    # across consecutive line columns. One output row = one pair of adjacent
    # horizontal lines, so the output is built one row-pair at a time, and each
    # maximal stretch of agreeing colour is painted as one run. Everything is
    # measured from I only.
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # bgc = the block-interior colour: once the rows and columns free of it are
    # removed (the lattice lines), every remaining cell is that colour.
    def lattice_ok(b):
        lr = {r for r in range(hi) if b not in I[r].tolist()}
        lc = {c for c in range(wi) if b not in I[:, c].tolist()}
        if not lr or not lc:
            return False
        return all(I[r, c] == b for r in range(hi) for c in range(wi)
                   if r not in lr and c not in lc)
    cnt = Counter(I.flatten().tolist())
    cands = [b for b, _ in cnt.most_common() if lattice_ok(b)]
    bgc = cands[0] if cands else cnt.most_common(1)[0][0]
    # linc = the colour that dominates the lattice lines
    lr0 = [r for r in range(hi) if bgc not in I[r].tolist()]
    lc0 = [c for c in range(wi) if bgc not in I[:, c].tolist()]
    lv = [int(v) for r in lr0 for v in I[r]] + [int(v) for c in lc0 for v in I[:, c]]
    linc = Counter(lv).most_common(1)[0][0] if lv else bgc

    line_rows = [r for r in range(hi) if bgc not in I[r].tolist()]
    line_cols = [c for c in range(wi) if bgc not in I[:, c].tolist()]
    markers = [(r, c) for r in range(hi) for c in range(wi)
               if I[r, c] != bgc and I[r, c] != linc]
    if markers:
        rmin = min(r for r, _ in markers); rmax = max(r for r, _ in markers)
        cmin = min(c for _, c in markers); cmax = max(c for _, c in markers)
        hlines = [r for r in line_rows if rmin <= r <= rmax]
        vlines = [c for c in line_cols if cmin <= c <= cmax]
    else:
        hlines, vlines = list(line_rows), list(line_cols)
    ho = max(1, len(hlines) - 1)
    wo = max(1, len(vlines) - 1)

    ops, sels = [], []
    # shrink the canvas to one cell per lattice block (top-left crop of I)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    canvas = I[:ho, :wo]
    # lay the block-interior colour as the base over the whole new canvas
    if np.any(canvas != bgc):
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle

    for i in range(ho):
        top, bot = hlines[i], hlines[i + 1]
        # agreement of the two line rows at each vertical-line intersection
        agree = [int(I[top, c]) if I[top, c] == I[bot, c] and I[top, c] not in (bgc, linc)
                 else None for c in vlines]
        # pair consecutive intersections -> block colours for this row pair
        row = [agree[j] if agree[j] is not None and agree[j] == agree[j + 1] else None
               for j in range(wo)]
        j = 0
        while j < wo:
            v = row[j]
            if v is None:
                j += 1; continue
            k = j
            while k + 1 < wo and row[k + 1] == v:
                k += 1
            ops.append(v); sels.append(sel_of([(i, c) for c in range(j, k + 1)]))
            j = k + 1

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
                        f"num_examples+1 ({num_examples + 1}) for task 7837ac64"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7837ac64"
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
                                f"for task 7837ac64"
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
                    f"Failed to build a complete episode for task 7837ac64 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7837ac64-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
