"""
ARC Task: 50846271 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, crossc, noisec = random.sample(cols, 3)
    return {"bgc": bgc, "crossc": crossc, "noisec": noisec,
            "dim_half": random.randint(1, 3)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, crossc, noisec, dim_half=None) -> dict:
    def unifint(lb, ub):
        lb, ub = int(lb), int(ub)
        if ub <= lb:
            return lb
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, lb + int(round(d * (ub - lb)))))

    def nbs(p):
        return {(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def mnbs(S):
        out = set()
        for p in S:
            out |= nbs(p)
        return out

    if dim_half is None:
        dim_half = random.randint(1, 3)
    dim = 2 * dim_half + 1
    hline = [(dim // 2, j) for j in range(dim)]
    vline = [(i, dim // 2) for i in range(dim)]
    cross = set(hline) | set(vline)
    ctr = (dim // 2, dim // 2)

    def cf1(d):
        return {(d // 2, 0), (d // 2, d - 1)} | set(random.sample(hline, random.randint(1, d)))

    def cf2(d):
        return {(0, d // 2), (d - 1, d // 2)} | set(random.sample(vline, random.randint(1, d)))

    def cf3(d):
        hh = [p for p in hline if p != ctr]
        vv = [p for p in vline if p != ctr]
        return (set(random.sample(hh, random.randint(1, d - 1))) |
                set(random.sample(vv, random.randint(1, d - 1))))

    def cf(d):
        return random.choice((cf1, cf2, cf3))(d)

    sh = lambda S, o: {(a + o[0], b + o[1]) for a, b in S}
    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    gi = np.full((h, w), bgc, dtype=int)
    namt = unifint(int(0.35 * h * w), int(0.65 * h * w))
    inds = {(i, j) for i in range(h) for j in range(w)}
    for (i, j) in random.sample(sorted(inds), namt):
        gi[i, j] = noisec
    initcross = random.choice((cf1, cf2))(dim)
    loci = random.randint(0, h - dim)
    locj = random.randint(0, w - dim)
    delt = sh(cross - initcross, (loci, locj))
    for (i, j) in sh(initcross, (loci, locj)):
        gi[i, j] = crossc
    for (i, j) in delt:
        gi[i, j] = noisec
    go = gi.copy()
    for (i, j) in delt:
        go[i, j] = 8
    plcd = sh(cross, (loci, locj))
    nb = mnbs(plcd)
    inds = (inds - plcd) - nb
    nb2 = mnbs(nb)
    inds -= nb2
    inds -= mnbs(nb2)
    noccs = unifint(1, int((h * w) / (10 * dim)))
    succ = tr = 0
    while succ < noccs and tr < 5 * noccs:
        tr += 1
        cands = [p for p in inds if p[0] <= h - dim and p[1] <= w - dim]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        # reject marker patterns that fit another plus position (unsolvable ambiguity)
        mk = cf(dim)
        for _ in range(20):
            if not any(mk <= sh(cross, (a, b)) for a in range(-dim + 1, dim)
                       for b in range(-dim + 1, dim) if (a, b) != (0, 0)):
                break
            mk = cf(dim)
        else:
            continue
        marked = sh(mk, loc)
        full = sh(cross, loc)
        if full.issubset(inds):
            d1 = mnbs(full)
            d2 = mnbs(d1)
            inds = (inds - full) - (d1 | d2 | mnbs(d2))
            succ += 1
            for (i, j) in full - marked:
                gi[i, j] = noisec
                go[i, j] = 8
            for (i, j) in marked:
                gi[i, j] = crossc
                go[i, j] = crossc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    from collections import Counter
    from maker.sel_helpers import sel_of
    A = np.asarray(I, dtype=int)
    hi, wi = A.shape
    cnt = Counter(A.flatten().tolist())
    crossc = min(cnt, key=lambda c: (cnt[c], c))          # marker colour = rarest colour
    cells = {(r, c) for r in range(hi) for c in range(wi) if A[r, c] == crossc}

    def comps_of(S):
        out, seen = [], set()
        for p in sorted(S):
            if p in seen:
                continue
            seen.add(p); st = [p]; comp = []
            while st:
                q = st.pop(); comp.append(q)
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        n = (q[0] + a, q[1] + b)
                        if n in S and n not in seen:
                            seen.add(n); st.append(n)
            out.append(comp)
        return out

    def extent(comp):
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        return max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    # --- cross size: read it off the demonstrations (completed crosses = 8 + marker colour) ---
    dim = 0
    for ex in (examples or []):
        try:
            EI = np.asarray(ex[0], dtype=int); EO = np.asarray(ex[1], dtype=int)
        except Exception:
            continue
        ec = Counter(EI.flatten().tolist())
        ecc = min(ec, key=lambda c: (ec[c], c))
        S = {(r, c) for r in range(EO.shape[0]) for c in range(EO.shape[1])
             if EO[r, c] == 8 or EO[r, c] == ecc}
        for comp in comps_of(S):
            dim = max(dim, extent(comp))

    # --- each cross is one unit: the centre whose plus explains its marker cells ---
    def fit(half):
        def cross_at(r, c):
            return ({(r, c + t) for t in range(-half, half + 1)} |
                    {(r + t, c) for t in range(-half, half + 1)})
        cand = []
        for r in range(half, hi - half):
            for c in range(half, wi - half):
                cr = cross_at(r, c)
                inbox = {(r + a, c + b) for a in range(-half, half + 1)
                         for b in range(-half, half + 1)} & cells
                if not inbox or not inbox <= cr:
                    continue
                both = (any(p[0] == r and p[1] != c for p in inbox) and
                        any(p[1] == c and p[0] != r for p in inbox))
                cand.append(((r, c), inbox, both))
        remaining, centers, used = set(cells), [], set()
        while remaining:
            best = None
            for (ctr, cov, both) in cand:
                if not cov <= remaining or (cross_at(*ctr) & used):
                    continue
                key = (len(cov), both, -ctr[0], -ctr[1])
                if best is None or key > best[0]:
                    best = (key, ctr, cov)
            if best is None:
                return None
            centers.append(best[1])
            remaining -= best[2]
            used |= cross_at(*best[1])
        return centers, cross_at

    res = fit(dim // 2) if dim in (3, 5, 7) else None
    if res is None:
        # no usable demonstration: the size explaining every marker with the fewest crosses
        best = None
        for hf in (1, 2, 3):
            rr = fit(hf)
            if rr is not None and (best is None or len(rr[0]) < len(best[0])):
                best = rr
        res = best
    centers, cross_at = res if res else ([], None)

    # --- complete each cross with 8 in one stroke: its plus cells not already marked ---
    ops, sels = [], []
    for (r, c) in sorted(centers):
        miss = sorted(p for p in cross_at(r, c) if A[p] != crossc)
        if miss:
            ops.append(8); sels.append(sel_of(miss))
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
                        f"num_examples+1 ({num_examples + 1}) for task 50846271"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 50846271"
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
                                f"for task 50846271"
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
                    f"Failed to build a complete episode for task 50846271 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"50846271-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
