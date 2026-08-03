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
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, crossc, noisec = random.sample(cols, 3)
    return {"bgc": bgc, "crossc": crossc, "noisec": noisec}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, crossc=None, noisec=None) -> dict:
    cols = remove(8, interval(0, 10, 1))
    if bgc is None or crossc is None or noisec is None:
        bgc, crossc, noisec = sample(cols, 3)
    cf1 = lambda d: {(d // 2, 0), (d // 2, d - 1)} | set(
        sample(totuple(connect((d // 2, 0), (d // 2, d - 1))), randint(1, d)))
    cf2 = lambda d: {(0, d // 2), (d - 1, d // 2)} | set(
        sample(totuple(connect((0, d // 2), (d - 1, d // 2))), randint(1, d)))
    cf3 = lambda d: set(sample(totuple(remove((d // 2, d // 2), connect((d // 2, 0), (d // 2, d - 1)))), randint(1, d - 1))) | set(
        sample(totuple(remove((d // 2, d // 2), connect((0, d // 2), (d - 1, d // 2)))), randint(1, d - 1)))
    cf = lambda d: choice((cf1, cf2, cf3))(d)

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    gi, go = None, None
    for _attempt in range(300):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        dimmax = max(1, min(3, (min(h, w) - 1) // 2))
        dim = unifint(diff_lb, diff_ub, (1, dimmax))
        dim = 2 * dim + 1
        cross = connect((dim // 2, 0), (dim // 2, dim - 1)) | connect((0, dim // 2), (dim - 1, dim // 2))
        gi = canvas(bgc, (h, w))
        namt = unifint(diff_lb, diff_ub, (int(0.35 * h * w), int(0.65 * h * w)))
        inds = asindices(gi)
        noise = sample(totuple(inds), namt)
        gi = fill(gi, noisec, noise)
        initcross = choice((cf1, cf2))(dim)
        loci = randint(0, h - dim)
        locj = randint(0, w - dim)
        delt = shift(cross - initcross, (loci, locj))
        gi = fill(gi, crossc, shift(initcross, (loci, locj)))
        gi = fill(gi, noisec, delt)
        go = fill(gi, 8, delt)
        plcd = shift(cross, (loci, locj))
        nbhs = mapply(neighbors, plcd)
        inds = (inds - plcd) - nbhs
        nbhs2 = mapply(neighbors, nbhs)
        inds = inds - nbhs2
        inds = inds - mapply(neighbors, nbhs2)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, int((h * w) / (10 * dim)))))
        succ = 0
        tr = 0
        maxtr = 5 * noccs
        while succ < noccs and tr < maxtr:
            tr += 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - dim and ij[1] <= w - dim)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            marked = shift(cf(dim), loc)
            full = shift(cross, loc)
            unmarked = full - marked
            inobj = recolor(noisec, unmarked) | recolor(crossc, marked)
            outobj = recolor(8, unmarked) | recolor(crossc, marked)
            outobji = toindices(outobj)
            if outobji.issubset(inds):
                dnbhs = mapply(neighbors, outobji)
                dnbhs2 = mapply(neighbors, dnbhs)
                inds = (inds - outobji) - (dnbhs | dnbhs2 | mapply(neighbors, dnbhs2))
                succ += 1
                gi = paint(gi, inobj)
                go = paint(go, outobj)
        # keep only pairs whose rule (complete every partial cross) is recoverable from the input alone
        try:
            ops, sels = derive_operations(gi, go)
        except Exception:
            continue
        sim = [list(row) for row in gi]
        for op, sel in zip(ops, sels):
            if op == 34:
                continue
            r0, c0, hh, ww = sel
            for r in range(r0, r0 + hh + 1):
                for c in range(c0, c0 + ww + 1):
                    sim[r][c] = op
        if tuple(tuple(row) for row in sim) == go:
            return {'input': gi, 'output': go}
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    A = np.asarray(I, dtype=int)
    B = np.asarray(O, dtype=int)
    hi, wi = A.shape
    ho, wo = B.shape

    # --- rule: the rarest colour draws partial plus/cross shapes; complete each one with 8 ---
    cnt = Counter(A.flatten().tolist())
    crossc = min(cnt, key=lambda c: cnt[c])
    orig = {(r, c) for r in range(hi) for c in range(wi) if A[r, c] == crossc}

    # 1. re-join collinear fragments of the same cross (gaps of at most 3 cells)
    S = set(orig)
    for _ in range(4):
        add = set()
        for (r, c) in S:
            for k in (2, 3, 4):
                if (r, c + k) in S:
                    add.add((r, c + 1))
                    add.add((r, c + k - 1))
                if (r + k, c) in S:
                    add.add((r + 1, c))
                    add.add((r + k - 1, c))
        add -= S
        if not add:
            break
        S |= add

    # 2. fragments -> arm length: the widest fragment spans one full cross
    comps = []
    seen = set()
    for p in sorted(S):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        comp = []
        while stack:
            q = stack.pop()
            comp.append(q)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (q[0] + dr, q[1] + dc)
                    if n in S and n not in seen:
                        seen.add(n)
                        stack.append(n)
        comps.append(comp)

    dim = 0
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        dim = max(dim, max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
    half = dim // 2

    # 3. cross centres: a cell with both a horizontal and a vertical neighbour,
    #    or the middle of a bare full-length arm
    centers = set()
    for (r, c) in S:
        if (((r, c - 1) in S or (r, c + 1) in S) and
                ((r - 1, c) in S or (r + 1, c) in S)):
            centers.add((r, c))
    for comp in comps:
        if len(comp) != dim:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if min(rs) == max(rs) or min(cs) == max(cs):
            centers.add(((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2))

    # 4. per cross: paint the arm cells that are still missing, one whole run at a time
    ops, sels = [], []
    painted = set()
    for (r, c) in sorted(centers):
        need_v = [rr for rr in range(max(0, r - half), min(hi, r + half + 1))
                  if (rr, c) not in orig and (rr, c) not in painted]
        runs = []
        for rr in need_v:
            if runs and rr == runs[-1][-1] + 1:
                runs[-1].append(rr)
            else:
                runs.append([rr])
        for run in runs:
            ops.append(8)
            sels.append([run[0], c, len(run) - 1, 0])
            for rr in run:
                painted.add((rr, c))

        need_h = [cc for cc in range(max(0, c - half), min(wi, c + half + 1))
                  if (r, cc) not in orig and (r, cc) not in painted]
        runs = []
        for cc in need_h:
            if runs and cc == runs[-1][-1] + 1:
                runs[-1].append(cc)
            else:
                runs.append([cc])
        for run in runs:
            ops.append(8)
            sels.append([r, run[0], 0, len(run) - 1])
            for cc in run:
                painted.add((r, cc))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
