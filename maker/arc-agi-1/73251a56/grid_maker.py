"""
ARC Task: 73251a56 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Episode-level colour roles: the noise colour and the palette used for the bands.

    `rot` is a discrete structural variant: rot 0/2 leave the grid mirror-symmetric about
    the MAIN diagonal, rot 1/3 about the ANTI diagonal.  Both classes must be visible in
    the examples for the episode to be learnable, so they are planned per instance.
    """
    cols = list(range(10))
    noisec = random.choice(cols)
    pool = [c for c in cols if c != noisec]
    random.shuffle(pool)

    variants = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= 4:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 4)]
        random.shuffle(examples)
    elif n_ex >= 2:
        examples = [{"rot": random.choice([0, 2])}, {"rot": random.choice([1, 3])}]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants))]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "ccols_pool": pool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, ccols_pool, rot=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def rot_k(g, k):
        out = [row[:] for row in g]
        for _ in range(k % 4):
            out = [list(r) for r in zip(*out[::-1])]
        return out

    # ---- the same detector derive_operations uses; only reads the (noisy) grid ----
    def solve(A):
        n = len(A)
        m = len(A[0])
        best = None
        for kind in ("d", "c"):
            if kind == "d":
                mir = lambda i, j: (j, i)
                axis = [(k, k) for k in range(min(n, m))]
                acol = A[0][0]
            else:
                mir = lambda i, j: (m - 1 - j, n - 1 - i)
                axis = [(k, m - 1 - k) for k in range(min(n, m))]
                acol = A[0][m - 1]
            for c in sorted({v for row in A for v in row}):
                G = [row[:] for row in A]
                for i in range(n):
                    for j in range(m):
                        if A[i][j] != c:
                            mi, mj = mir(i, j)
                            G[mi][mj] = A[i][j]
                for (i, j) in axis:
                    G[i][j] = acol
                ok = True
                for i in range(n):
                    for j in range(m):
                        mi, mj = mir(i, j)
                        if G[i][j] != G[mi][mj]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                rem = [(i, j) for i in range(n) for j in range(m) if G[i][j] == c]
                if rem:
                    remset = set(rem)
                    nb = []
                    for (i, j) in rem:
                        for p in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                            if 0 <= p[0] < n and 0 <= p[1] < m and p not in remset:
                                nb.append(G[p[0]][p[1]])
                    if nb:
                        fv = max(set(nb), key=nb.count)
                        for (i, j) in rem:
                            G[i][j] = fv
                if all(G[i][j] == A[i][j] for i in range(n) for j in range(m)):
                    continue
                cnt = sum(row.count(c) for row in A)
                key = (cnt, 0 if kind == "d" else 1, c)
                if best is None or key < best[0]:
                    best = (key, G, c, kind, acol)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    dmax = min(30, int(max_h), int(max_w))
    dmin = min(10, dmax)
    pool = [c for c in ccols_pool if c != noisec]

    while True:
        d = unifint(diff_lb, diff_ub, (dmin, dmax))
        h = w = d
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        nsl = max(2, min(nsl, len(pool)))
        if nsl - 1 > h - 1:
            continue
        slopes = [0] + sorted(random.sample(range(1, h), nsl - 1))
        ccols = list(pool[:nsl])

        g = [[ccols[0]] * w for _ in range(h)]
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            for i in range(h):
                for j in range(w):
                    if slope * j <= i:
                        g[i][j] = col
        for k in range(d):
            g[k][k] = ccols[-2]
        for i in range(h):
            for j in range(i, w):
                g[j][i] = g[i][j]
        go = [row[:] for row in g]

        ndist = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
        noise = set()
        dnoise = set()
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = random.randint(1, 5)
            ow = random.randint(1, 5)
            if h - oh - 1 < 1 or w - ow - 1 < 1:
                continue
            loci = random.randint(1, h - oh - 1)
            locj = random.randint(1, w - ow - 1)
            cells = [(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)]
            nn = noise | set(cells)
            newdiag = {p for p in cells if p[0] == p[1]}
            if len(dnoise | newdiag) >= d:          # cf1: keep some of the diagonal line
                continue
            if any(p[0] != p[1] and (p[1], p[0]) in nn for p in cells):   # cf2
                continue
            noise = nn
            dnoise |= newdiag
            succ += 1
        if not noise:
            continue

        gi = [row[:] for row in g]
        for (i, j) in noise:
            gi[i][j] = noisec
        gi = rot_k(gi, rot)
        go2 = rot_k(go, rot)

        sol = solve(gi)
        if sol is None or sol[0] != go2:
            continue
        return {"input": gi, "output": go2}


def derive_operations(I, O):
    """Everything below is measured from I only.

    Rule (from the generator): the clean picture is mirror-symmetric about one of the two
    diagonals; rectangular blobs of a single 'noise' colour were stamped on top of it.
    From I we (1) find which diagonal is the symmetry axis and which colour is the noise,
    (2) read every noise cell's replacement from its MIRROR CELL in I (cells lying on the
    axis take the axis-line colour, read from the grid corner on that axis), and
    (3) repaint blob by blob, one Color op per colour band inside each blob.
    O is never inspected.
    """
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    A = np.asarray(I, dtype=int).tolist()
    n = len(A)
    m = len(A[0])

    def solve(A):
        best = None
        for kind in ("d", "c"):
            if kind == "d":
                mir = lambda i, j: (j, i)
                axis = [(k, k) for k in range(min(n, m))]
                acol = A[0][0]
            else:
                mir = lambda i, j: (m - 1 - j, n - 1 - i)
                axis = [(k, m - 1 - k) for k in range(min(n, m))]
                acol = A[0][m - 1]
            for c in sorted({v for row in A for v in row}):
                G = [row[:] for row in A]
                # every cell that is NOT the candidate noise colour paints its mirror
                for i in range(n):
                    for j in range(m):
                        if A[i][j] != c:
                            mi, mj = mir(i, j)
                            G[mi][mj] = A[i][j]
                # the axis line itself carries the corner colour
                for (i, j) in axis:
                    G[i][j] = acol
                ok = True
                for i in range(n):
                    for j in range(m):
                        mi, mj = mir(i, j)
                        if G[i][j] != G[mi][mj]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                rem = [(i, j) for i in range(n) for j in range(m) if G[i][j] == c]
                if rem:
                    remset = set(rem)
                    nb = []
                    for (i, j) in rem:
                        for p in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                            if 0 <= p[0] < n and 0 <= p[1] < m and p not in remset:
                                nb.append(G[p[0]][p[1]])
                    if nb:
                        fv = max(set(nb), key=nb.count)
                        for (i, j) in rem:
                            G[i][j] = fv
                if all(G[i][j] == A[i][j] for i in range(n) for j in range(m)):
                    continue
                cnt = sum(row.count(c) for row in A)
                key = (cnt, 0 if kind == "d" else 1, c)
                if best is None or key < best[0]:
                    best = (key, G, c, kind, acol)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    ops, sels = [], []
    sol = solve(A)
    if sol is not None:
        G = sol[0]
        target = [(i, j) for i in range(n) for j in range(m) if G[i][j] != A[i][j]]
        tset = set(target)
        seen = set()
        comps = []
        for cell in sorted(target):
            if cell in seen:
                continue
            stack = [cell]
            seen.add(cell)
            comp = []
            while stack:
                (i, j) = stack.pop()
                comp.append((i, j))
                for p in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if p in tset and p not in seen:
                        seen.add(p)
                        stack.append(p)
            comps.append(sorted(comp))
        # one blob at a time; inside a blob, one Color op per restored colour band
        for comp in comps:
            groups = {}
            for (i, j) in comp:
                groups.setdefault(int(G[i][j]), []).append((i, j))
            for col in sorted(groups, key=lambda c: min(groups[c])):
                ops.append(int(col))
                sels.append(sel_of(groups[col]))

    ops.append(34)
    sels.append([0, 0, n - 1, m - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 73251a56"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 73251a56"
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
                                f"for task 73251a56"
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
                    f"Failed to build a complete episode for task 73251a56 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"73251a56-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
