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


def sample_colors(num_examples=None) -> dict:
    # The noise colour is the one colour role the rule hinges on: fix it per episode so
    # every demonstration shows the same colour being treated as "missing".
    return {"noisec": random.randrange(10)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, noisec=None, **kwargs) -> dict:
    from dsl import (interval, remove, canvas, asindices, sfilter, fill, connect,
                     asobject, paint, dmirror, palette, toobject, ofcolor,
                     backdrop, identity, rot90, rot180, rot270)
    from utils import unifint
    from random import choice, randint, sample
    cols = interval(0, 10, 1)
    hi = max(10, min(max_h, max_w, 30))
    if noisec is None:
        noisec = choice(cols)
    remcols = remove(noisec, cols)
    while True:
        d = unifint(diff_lb, diff_ub, (10, hi))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, min(9, h // 2)))
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
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading taken from the golf entry  (g[r][c] = g[r][c] or g[c][r]):
    # the clean picture is symmetric about one diagonal. The noise colour marks MISSING
    # cells, and a cell is never missing together with its mirror partner. So the whole
    # grid is laid over its own mirror image, and each hole is filled with what its
    # partner shows. The maker treats each noise blob as its own unit; this treats the
    # whole grid as one unit:
    #   1. erase the noise colour (all cells of that colour -> 0 = "nothing"),
    #   2. CopyO the whole grid: the clipboard now holds every surviving clean cell,
    #   3. mirror the whole grid across the axis (Rotate + FlipH = (anti)transpose):
    #      each hole now holds its partner's clean colour,
    #   4. Paste the clean cells back at (0,0) on top (transparent over the holes),
    #   5. the axis is its own mirror, so repaint the axis line with its corner colour.
    # The noise colour is read from the demonstrations (the colour every example
    # repairs). The axis is then the diagonal that explains every asymmetry of I.
    I = np.asarray(I, dtype=int)
    n, m = I.shape
    mirrors = {
        "main": (lambda r, c: (c, r), int(I[0, 0])),
        "anti": (lambda r, c: (n - 1 - c, n - 1 - r), int(I[0, n - 1])),
    }

    def consistent(f, dcol, col):
        for r in range(n):
            for c in range(n):
                rr, cc = f(r, c)
                a, b = I[r, c], I[rr, cc]
                if (rr, cc) == (r, c):
                    if a != col and a != dcol:
                        return False
                elif a != col and b != col and a != b:
                    return False
        return True

    noisec = None
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            fixed = set(int(v) for v in np.unique(ei[ei != eo]))
            if fixed:
                cand = fixed if cand is None else cand & fixed
        if cand and len(cand) == 1:
            noisec = cand.pop()

    best = None
    if noisec is not None:
        for name, (f, dcol) in mirrors.items():
            if consistent(f, dcol, noisec):
                best = (name, noisec)
                break
    if best is None:
        cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
        for name, (f, dcol) in mirrors.items():
            for col in sorted(cnt, key=lambda v: cnt[v]):
                if consistent(f, dcol, col):
                    if best is None or cnt[col] < cnt[best[1]]:
                        best = (name, col)
                    break
    name, noisec = best
    f, dcol = mirrors[name]
    full = [0, 0, n - 1, m - 1]  # whole grid rectangle (whole-grid mirror / copy)
    axis = [(r, r) for r in range(n)] if name == "main" else [(r, n - 1 - r) for r in range(n)]
    noise = [(r, c) for r in range(n) for c in range(n) if I[r, c] == noisec]
    off_axis = [p for p in noise if f(*p) != p]
    on_axis_noise = any(I[r, c] == noisec for r, c in axis)

    ops, sels = [], []
    if off_axis:
        if noisec != 0:
            ops.append(0); sels.append(sel_of(noise))          # erase noise colour
        # clean cells must reclaim their places only if a mirrored hole lands on a
        # non-zero clean cell (otherwise the mirrored grid is already right there)
        need_paste = any(I[f(r, c)] != 0 for r, c in off_axis)
        if need_paste:
            ops.append(29); sels.append(full)                   # CopyO clean cells
        ops.append(25 if name == "main" else 24); sels.append(full)
        ops.append(26); sels.append(full)                       # -> (anti)transpose
        if need_paste:
            ops.append(30); sels.append([0, 0, 0, 0])           # paste clean cells back
        if on_axis_noise and dcol != 0:
            ops.append(dcol); sels.append(sel_of(axis))         # restore axis line
    elif on_axis_noise and dcol != noisec:
        ops.append(dcol); sels.append(sel_of(axis))             # noise only on the axis
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
