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
from maker.sel_helpers import sel_of


# The only structural variant the generator has is the final rotation, which
# decides WHICH diagonal the grid is symmetric about (main diagonal for
# identity/rot180, anti-diagonal for rot90/rot270).  Plan it per instance so the
# examples cover both axes.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             noisec=None, rot=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if noisec is None:
        noisec = choice(cols)
    if rot is None:
        rot = choice((0, 1, 2, 3))
    m = min(max_h, max_w)
    lo = min(10, m)
    while True:
        d = unifint(diff_lb, diff_ub, (lo, m))
        h, w = d, d
        remcols = remove(noisec, cols)
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        slopes = [0] + sorted(sample(interval(1, h - 1, 1), nsl - 1))
        ccols = sample(remcols, nsl)
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
            oh = randint(1, max(1, min(5, h - 2)))
            ow = randint(1, max(1, min(5, w - 2)))
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The grid is a diagonally-symmetric pattern spoiled by rectangular blobs of one
    'noise' colour.  Everything below is measured from I: which diagonal is the axis of
    symmetry, which colour is the intruder, and what each intruding cell must become
    (the value of its mirror partner across that axis, or the axis colour itself for
    cells sitting on the axis).  O is never inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    if h != w:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])   # full-grid bbox == whole canvas
        return ops, sels
    n = h

    def mirror(A, axis):
        # 'main'  : reflect across the main diagonal      -> A[j, i]
        # 'anti'  : reflect across the anti-diagonal      -> A[n-1-j, n-1-i]
        return A.T if axis == 'main' else A[::-1, ::-1].T

    def axis_line(axis):
        if axis == 'main':
            return [(i, i) for i in range(n)], int(I[0, 0])
        return [(i, n - 1 - i) for i in range(n)], int(I[0, n - 1])

    # ---- infer axis + intruder colour from I alone -------------------------------
    # A candidate (axis, c) is admissible when: filling every cell from its mirror
    # partner (except where that partner is itself colour c) and restoring the axis
    # line to the colour it carries at the clean grid border makes the grid perfectly
    # symmetric about that axis.  Among admissible candidates the intruder is the one
    # occupying fewest cells of I.
    best = None
    for axis in ('main', 'anti'):
        M = mirror(I, axis)
        mism = int(np.count_nonzero(M != I))
        dcells, dcol = axis_line(axis)
        for c in sorted(set(I.flatten().tolist())):
            G = np.where(M != c, M, I)
            for (r, cc) in dcells:
                G[r, cc] = dcol
            if not np.array_equal(G, mirror(G, axis)):
                continue
            cnt = int(np.count_nonzero(I == c))
            changed = int(np.count_nonzero(G != I))
            key = (0 if changed > 0 else 1, cnt, mism, int(c))
            if best is None or key < best[0]:
                best = (key, G)
    if best is None:
        ops.append(34); sels.append([0, 0, n - 1, n - 1])
        return ops, sels
    G = best[1]

    # ---- cells the intruder occupies, grouped into its blobs ---------------------
    dirty = {(int(r), int(c)) for r, c in zip(*np.nonzero(G != I))}
    seen = set()
    blobs = []
    for p in sorted(dirty):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        comp = []
        while stack:
            (r, c) = stack.pop()
            comp.append((r, c))
            for (nr, nc) in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                q = (nr, nc)
                if q in dirty and q not in seen:
                    seen.add(q)
                    stack.append(q)
        blobs.append(sorted(comp))

    # ---- restore one blob at a time, colour by colour ----------------------------
    blobs.sort(key=lambda comp: comp[0])
    for comp in blobs:
        groups = {}
        for (r, c) in comp:
            groups.setdefault(int(G[r, c]), []).append((r, c))
        for col in sorted(groups, key=lambda k: groups[k][0]):
            ops.append(int(col))
            sels.append(sel_of(groups[col]))

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])   # full-grid bbox == whole canvas
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
