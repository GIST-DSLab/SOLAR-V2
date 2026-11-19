"""
ARC Task: 7df24a62 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(b, max(a, a + int(round((b - a) * d))))


def _dihedral(a):
    # the 8 ways a block can lie: 4 quarter turns, each with and without a mirror
    a = np.asarray(a)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, noisec=1, sqc=2, **kw) -> dict:
    # the canvas is trimmed by one cell on every side at the end, so +2
    h = _unifint(diff_lb, diff_ub, (12, max_h + 2))
    w = _unifint(diff_lb, diff_ub, (12, max_w + 2))
    oh = _unifint(diff_lb, diff_ub, (3, min(7, h // 3)))
    ow = _unifint(diff_lb, diff_ub, (3, min(7, w // 3)))
    inb = [(i, j) for i in range(1, oh - 1) for j in range(1, ow - 1)]
    obj = {random.choice(inb)}

    def shp(s):
        rs = [p[0] for p in s]
        cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(obj) != (oh - 2, ow - 2):
        obj.add(random.choice([p for p in inb if p not in obj]))
    pat = np.full((oh, ow), sqc, dtype=int)
    targ = np.full((oh, ow), bgc, dtype=int)
    for (i, j) in obj:
        pat[i, j] = noisec
        targ[i, j] = noisec
    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    gi[loci:loci + oh, locj:locj + ow] = pat
    box = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    nb = set(box)
    for (i, j) in box:
        nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
    inds = {(i, j) for i in range(1, h - 1) for j in range(1, w - 1)
            if gi[i, j] == bgc} - nb
    namt = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    for (i, j) in random.sample(sorted(inds), min(namt, len(inds))):
        gi[i, j] = noisec
    targs = _dihedral(targ)
    srcs = _dihedral(pat)
    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
    succ = tr = 0
    while succ < noccs and tr < 5 * noccs:
        tr += 1
        t = random.choice(targs)
        hh, ww = t.shape
        cands = [p for p in inds if 1 <= p[0] <= h - hh - 1 and 1 <= p[1] <= w - ww - 1]
        if not cands:
            continue
        a, b = random.choice(sorted(cands))
        tpi = {(a + i, b + j) for i in range(hh) for j in range(ww)}
        if tpi <= inds:
            succ += 1
            inds -= tpi
            gi[a:a + hh, b:b + ww] = t
    go = gi.copy()
    go[go == sqc] = bgc
    go[loci:loci + oh, locj:locj + ow] = pat
    res = go.copy()
    for t, s in zip(targs, srcs):
        hh, ww = t.shape
        for a in range(h - hh + 1):
            for b in range(w - ww + 1):
                if np.array_equal(go[a:a + hh, b:b + ww], t):
                    res[a:a + hh, b:b + ww] = s
    return {"input": gi[1:-1, 1:-1].tolist(), "output": res[1:-1, 1:-1].tolist()}


def derive_operations(I, O=None, examples=None):
    # How this reads the task: the box is a frame of colour sqc around a small
    # pattern of the scattered colour. Wherever that same pattern shows up again
    # on clean background, in any of its 8 orientations, the background around
    # it and between its cells becomes sqc. Each match is one unit and gets one
    # Color(sqc), applied to the background cells of that match's window.
    # The pattern's own cells are never touched, so nothing is copied, turned or
    # mirrored. Every value here is measured from I. O is never read.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    def maxdim(c):
        rs, cs = np.nonzero(I == c)
        return max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1)

    # box colour: the colour whose cells span the smallest extent
    sqc = int(min(vals, key=lambda c: (maxdim(c), c)))
    others = [(int(n), int(v)) for v, n in zip(vals, counts) if v != sqc]
    if others and len(vals) >= 3:
        # pad by one background cell: a match's frame may sit just off the grid
        P = np.full((H + 2, W + 2), bgc, dtype=int)
        P[1:-1, 1:-1] = I
        rs, cs = np.nonzero(P == sqc)
        model = P[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
        # what a match looks like in I: the pattern, and background where the frame goes
        targ = np.where(model == sqc, bgc, model)
        seen = set()
        G = I.copy()
        for t in _dihedral(targ):
            th, tw = t.shape
            for a in range(H + 2 - th + 1):
                for b in range(W + 2 - tw + 1):
                    if (a, b, th, tw) in seen:
                        continue
                    if not np.array_equal(P[a:a + th, b:b + tw], t):
                        continue
                    seen.add((a, b, th, tw))
                    # the background cells of this match's window, clipped to the
                    # grid; cells an earlier box already painted are left out
                    cells = [(r, c)
                             for r in range(max(a - 1, 0), min(a - 1 + th, H))
                             for c in range(max(b - 1, 0), min(b - 1 + tw, W))
                             if G[r, c] == bgc]
                    if cells:
                        ops.append(sqc)
                        sels.append(sel_of(cells))
                        for p in cells:
                            G[p] = sqc
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
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
                                f"for task 7df24a62"
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
                    f"Failed to build a complete episode for task 7df24a62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7df24a62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
