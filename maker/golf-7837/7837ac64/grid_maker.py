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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Fix the two structural colours: the cell-interior background and the lattice-line colour.
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    oh = _unif(diff_lb, diff_ub, (2, max(2, min(6, (max_h - 1) // 3))))
    ow = _unif(diff_lb, diff_ub, (2, max(2, min(6, (max_w - 1) // 3))))
    numcols = _unif(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcols)
    go = [[bgc] * ow for _ in range(oh)]
    full = {(i, j) for i in range(oh) for j in range(ow)}
    inds = set(full)

    def nbrs(ij):
        i, j = ij
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    nocc = _unif(diff_lb, diff_ub, (1, oh * ow))
    for _ in range(nocc):
        mpr = []
        for cc in ccols:
            occ = {(i, j) for (i, j) in full if go[i][j] == cc}
            cand = set(inds)
            for p in occ:
                cand |= nbrs(p)
            ok = inds | occ
            locs = [p for p in cand if (nbrs(p) & full) <= ok]
            if locs:
                mpr.append((cc, sorted(locs)))
        if not mpr:
            break
        col, locs = random.choice(mpr)
        loc = random.choice(locs)
        if loc in full:
            go[loc[0]][loc[1]] = col
        inds.discard(loc)
    obj = [(i, j) for (i, j) in full if go[i][j] != bgc]
    r0 = min(i for i, _ in obj); r1 = max(i for i, _ in obj)
    c0 = min(j for _, j in obj); c1 = max(j for _, j in obj)
    go = [row[c0:c1 + 1] for row in go[r0:r1 + 1]]
    oh, ow = len(go), len(go[0])
    sqh = _unif(diff_lb, diff_ub, (2, max(2, (max_h - oh - 1) // oh)))
    sqw = _unif(diff_lb, diff_ub, (2, max(2, (max_w - ow - 1) // ow)))
    fullh = oh + 1 + oh * sqh
    fullw = ow + 1 + ow * sqw
    if fullh > max_h or fullw > max_w:
        raise ValueError("too big")
    gi = [[linc] * fullw for _ in range(fullh)]
    for i in range(oh):
        for j in range(ow):
            pr, pc = 1 + i * (sqh + 1), 1 + j * (sqw + 1)
            for a in range(sqh):
                for b in range(sqw):
                    gi[pr + a][pc + b] = bgc
            if go[i][j] != bgc:
                for (a, b) in ((pr - 1, pc - 1), (pr - 1, pc + sqw),
                               (pr + sqh, pc - 1), (pr + sqh, pc + sqw)):
                    gi[a][b] = go[i][j]
    gih = _unif(diff_lb, diff_ub, (fullh, max_h))
    giw = _unif(diff_lb, diff_ub, (fullw, max_w))
    li = random.randint(0, gih - fullh)
    lj = random.randint(0, giw - fullw)
    G = [[bgc] * giw for _ in range(gih)]
    for a in range(gih):
        if (a - li) % (sqh + 1) == 0:
            G[a] = [linc] * giw
    for b in range(giw):
        if (b - lj) % (sqw + 1) == 0:
            for a in range(gih):
                G[a][b] = linc
    for a in range(fullh):
        for b in range(fullw):
            G[li + a][lj + b] = gi[a][b]
    # closure: fill single bgc gaps between same-colour cells
    while True:
        go2 = [row[:] for row in go]
        for c in set(ccols):
            for i in range(oh):
                for j in range(ow - 2):
                    if go[i][j] == c and go[i][j + 1] == bgc and go[i][j + 2] == c:
                        go2[i][j + 1] = c
            for i in range(oh - 2):
                for j in range(ow):
                    if go[i][j] == c and go[i + 1][j] == bgc and go[i + 2][j] == c:
                        go2[i + 1][j] = c
        if go2 == go:
            break
        go = go2
    # reject instances where shared corners make the output ambiguous
    for i in range(oh):
        for j in range(ow):
            a, b = li + i * (sqh + 1), lj + j * (sqw + 1)
            vals = {G[a][b], G[a][b + sqw + 1], G[a + sqh + 1][b], G[a + sqh + 1][b + sqw + 1]}
            exp = vals.pop() if len(vals) == 1 else bgc
            if exp == linc:
                exp = bgc
            if exp != go[i][j]:
                raise ValueError("ambiguous corners")
    return {"input": G, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def mode(vals):
        return Counter(vals).most_common(1)[0][0]
    # bgc: cell interiors dominate most rows; linc: the lattice-line colour
    mc = Counter(mode(I[r].tolist()) for r in range(hi))
    bgc = mc.most_common(1)[0][0]
    others = [c for c in mc if c != bgc]
    if others:
        linc = max(others, key=lambda c: mc[c])
    else:
        rest = [c for c, _ in Counter(I.flatten().tolist()).most_common() if c != bgc]
        linc = rest[0] if rest else bgc

    # lattice lines (no interior colour on them), restricted to the marker span
    line_rows = [r for r in range(hi) if bgc not in I[r].tolist()]
    line_cols = [c for c in range(wi) if bgc not in I[:, c].tolist()]
    markers = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] not in (bgc, linc)]
    rmin = min(r for r, _ in markers); rmax = max(r for r, _ in markers)
    cmin = min(c for _, c in markers); cmax = max(c for _, c in markers)
    hl = [r for r in line_rows if rmin <= r <= rmax]
    vl = [c for c in line_cols if cmin <= c <= cmax]
    ho, nv = len(hl) - 1, len(vl)
    wo = nv - 1

    ops, sels = [], []
    # Stage 1 canvas: one row per PAIR of adjacent horizontal lattice lines,
    # one column per vertical lattice line.
    ops.append(33); sels.append([0, 0, ho - 1, nv - 1])
    cur = I[:ho, :nv].copy()
    if np.any(cur != bgc):
        # whole stage-1 canvas rectangle -> background base
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, nv - 1])
        cur[:, :] = bgc

    # Stage 1: line pair (hl[i], hl[i+1]) -> row i. At each vertical line k the
    # two intersections either agree on a marker colour (keep it) or not.
    for i in range(ho):
        by_col = {}
        for k in range(nv):
            a, b = int(I[hl[i], vl[k]]), int(I[hl[i + 1], vl[k]])
            if a == b and a not in (linc, bgc):
                by_col.setdefault(a, []).append((i, k))
        for v in sorted(by_col, key=lambda x: by_col[x][0]):
            ops.append(v); sels.append(sel_of(by_col[v]))
            for (r, c) in by_col[v]:
                cur[r, c] = v

    # Stage 2: column pair (k, k+1) -> column k. A stage-1 value survives only
    # where its right neighbour on the stage-1 canvas holds the same colour.
    for i in range(ho):
        kill = [(i, j) for j in range(wo)
                if cur[i, j] != bgc and cur[i, j + 1] != cur[i, j]]
        if kill:
            ops.append(int(bgc)); sels.append(sel_of(kill))
            for (r, c) in kill:
                cur[r, c] = bgc
    # drop the last vertical-line column: the pairs are consumed
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
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
