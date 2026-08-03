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
from collections import Counter

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    pool = [c for c in cols if c != noisec]
    random.shuffle(pool)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rotname": r} for r in ROTS]
        examples += [{"rotname": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rotname": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "ccols_pool": pool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, ccols_pool, rotname=None) -> dict:
    if rotname is None:
        rotname = choice(tuple(ROTS))
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rotname]
    ub = min(30, max_h, max_w)
    lb = min(10, ub)
    while True:
        d = unifint(diff_lb, diff_ub, (lb, ub))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        nsl = max(2, min(nsl, len(ccols_pool), h - 1))
        slopes = [0] + sorted(sample(interval(1, h - 1, 1), nsl - 1))
        ccols = list(ccols_pool[:nsl])
        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))
        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset({ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0
        ndist = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        go = tuple(e for e in gi)
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = randint(1, min(5, h - 2))
            ow = randint(1, min(5, w - 2))
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O):
    # Rule: the picture is mirror-symmetric about one of its two diagonals; blobs of a
    # single intruder colour break that symmetry.  Erase the blobs, mirror the whole
    # grid, and lay the erased picture back on top: each hole takes its mirror partner.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    n = I.shape[0]
    full = [0, 0, I.shape[0] - 1, I.shape[1] - 1]
    ops, sels = [], []

    def mir(kind, i, j):
        return (j, i) if kind == 'd' else (n - 1 - j, n - 1 - i)

    cnt = Counter(I.flatten().tolist())

    # candidate (intruder colour, mirror axis): every pair of non-intruder mirror
    # partners must agree
    cands = []
    for kind in ('d', 'c'):
        for ncol in cnt:
            ok = True
            for i in range(n):
                for j in range(n):
                    mi, mj = mir(kind, i, j)
                    a = int(I[i, j]); b = int(I[mi, mj])
                    if a == ncol or b == ncol:
                        continue
                    if a != b:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cands.append((cnt[ncol], ncol, kind))
    cands.sort()

    def build(ncol, kind):
        axis = [((i, i) if kind == 'd' else (i, n - 1 - i)) for i in range(n)]
        clean = [int(I[r, c]) for (r, c) in axis if int(I[r, c]) != ncol]
        if not clean:
            return None, None
        ac = Counter(clean).most_common(1)[0][0]
        pred = I.copy()
        for i in range(n):
            for j in range(n):
                if int(I[i, j]) == ncol:
                    mi, mj = mir(kind, i, j)
                    pred[i, j] = ac if int(I[mi, mj]) == ncol else int(I[mi, mj])
        return ac, pred

    chosen = None
    for _, ncol, kind in cands:
        ac, pred = build(ncol, kind)
        if pred is not None and np.array_equal(pred, O):
            chosen = (ncol, kind, ac)
            break
    if chosen is None:
        _, ncol, kind = cands[0]
        ac, _p = build(ncol, kind)
        chosen = (ncol, kind, 0 if ac is None else ac)
    ncol, kind, axis_color = chosen

    # 1. punch out each intruder blob (one flood fill per connected blob).
    #    if the intruder colour already is 0 the holes are there already.
    if ncol != 0:
        seen = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if int(I[i, j]) == ncol and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < n and 0 <= cc < n and not seen[rr, cc] and int(I[rr, cc]) == ncol:
                                seen[rr, cc] = True
                                stack.append((rr, cc))
                    ops.append(10)
                    sels.append([i, j, 0, 0])

    # 2. keep the punched picture
    ops.append(29); sels.append(list(full))
    # 3. mirror the whole grid across its symmetry axis
    ops.append(24); sels.append(list(full))
    ops.append(27 if kind == 'd' else 26); sels.append(list(full))
    # 4. lay the punched picture back on top: holes keep the mirrored values
    ops.append(30); sels.append([0, 0, 0, 0])

    # 5. cells sitting on the axis mirror onto themselves - restore the axis line colour
    if axis_color != 0:
        for i in range(n):
            r, c = (i, i) if kind == 'd' else (i, n - 1 - i)
            if int(I[r, c]) == ncol:
                ops.append(int(axis_color))
                sels.append([r, c, 0, 0])

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
