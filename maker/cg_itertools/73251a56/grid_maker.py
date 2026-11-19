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

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    noisec = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    # cover both axis orientations: main diagonal (identity/rot180), anti diagonal (rot90/rot270)
    if n_ex >= 2:
        ex = [random.choice(["identity", "rot180"]), random.choice(["rot90", "rot270"])]
        ex += [random.choice(ROTS) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [random.choice(ROTS) for _ in range(n_ex)]
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"noisec": noisec, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, rot=None, **kw) -> dict:
    from dsl import (interval, canvas, asindices, sfilter, fill, connect,
                     asobject, dmirror, paint, palette, toobject, ofcolor, backdrop,
                     identity, rot90, rot180, rot270)
    from utils import unifint
    from random import choice, sample, randint
    cols = interval(0, 10, 1)
    dmax = max(10, min(30, max_h, max_w))
    while True:
        d = unifint(diff_lb, diff_ub, (10, dmax))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, min(9, h // 2)))
        slopes = [0] + sorted(sample(interval(1, h - 1, 1), nsl - 1))
        ccols = sample(cols, nsl)
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
        ndist = unifint(diff_lb, diff_ub, (1, (h * w) // 15))
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        go = tuple(e for e in gi)
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = randint(1, 5)
            ow = randint(1, 5)
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break
    if rot is None:
        rot = choice(ROTS)
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    return {"input": rotf(gi), "output": rotf(go)}


def derive_operations(I, O=None, examples=None):
    # Reading (as in the golf answer g[y][x] | g[x][y]): the clean picture equals its own
    # transpose (about the main diagonal, or the anti-diagonal when the grid was turned),
    # so the answer is the grid OVERLAID with its own mirror image. Knock the noise colour
    # out to 0 (transparent holes), copy the holed grid, transpose the whole grid in place
    # (Rotate + FlipV), then paste the copy back on top: its holes are transparent, so
    # through every hole shows the mirror partner. Holes ON the axis map to themselves and
    # take the axis (corner) colour. Axis + noise colour are found from I alone.
    I = np.asarray(I, dtype=int)
    n, m = I.shape
    mirrors = {
        "main": (lambda r, c: (c, r), int(I[0, 0])),
        "anti": (lambda r, c: (n - 1 - c, n - 1 - r), int(I[0, n - 1])),
    }
    cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
    best = None
    for name, (f, dcol) in mirrors.items():
        for col in sorted(cnt, key=lambda v: cnt[v]):
            good = True
            for r in range(n):
                for c in range(n):
                    rr, cc = f(r, c)
                    a, b = I[r, c], I[rr, cc]
                    if (rr, cc) == (r, c):
                        if a != col and a != dcol:
                            good = False
                            break
                    elif a != col and b != col and a != b:
                        good = False
                        break
                if not good:
                    break
            if good:
                if best is None or cnt[col] < cnt[best[1]]:
                    best = (name, col)
                break
    name, noisec = best
    f, dcol = mirrors[name]
    full = [0, 0, n - 1, m - 1]  # whole square grid rectangle
    ops, sels = [], []

    noise = [(r, c) for r in range(n) for c in range(n) if I[r, c] == noisec]
    G = I.copy()
    # 1. all cells of the noise colour become transparent holes
    if noisec != 0 and noise:
        ops.append(0); sels.append(sel_of(noise))
        for p in noise:
            G[p] = 0
    T = G.T.copy() if name == "main" else G[::-1, ::-1].T.copy()
    if not np.array_equal(T, G):
        # 2. copy the holed grid (whole-grid rectangle)
        ops.append(29); sels.append(full)
        # 3. transpose the whole grid in place (square grid -> Rotate is safe)
        if name == "main":
            ops.append(24); sels.append(full)   # rot90 CCW then flipud = main transpose
        else:
            ops.append(25); sels.append(full)   # rot90 CW then flipud = anti transpose
        ops.append(27); sels.append(full)
        # 4. paste the holed original over its mirror image: holes show the partner
        ops.append(30); sels.append([0, 0, 0, 0])
        G = np.where(G != 0, G, T)
    # 5. holes on the axis are their own partner: give them the axis (corner) colour
    axis_holes = [(r, c) for (r, c) in noise if f(r, c) == (r, c)]
    if axis_holes and any(G[p] != dcol for p in axis_holes):
        ops.append(dcol); sels.append(sel_of(axis_holes))
    ops.append(34); sels.append(full)
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
