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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, sqc=None) -> dict:
    from random import choice, randint, sample

    def unifint(lb, ub, bounds):
        a, b = bounds
        b = max(a, b)
        return min(b, max(a, a + int(round((b - a) * random.uniform(lb, ub)))))

    if bgc is None or noisec is None or sqc is None:
        bgc, noisec, sqc = sample(range(10), 3)
    h = unifint(diff_lb, diff_ub, (12, max_h + 2))
    w = unifint(diff_lb, diff_ub, (12, max_w + 2))
    oh = unifint(diff_lb, diff_ub, (3, min(7, h // 3)))
    ow = unifint(diff_lb, diff_ub, (3, min(7, w // 3)))
    inbounds = [(i, j) for i in range(1, oh - 1) for j in range(1, ow - 1)]
    obj = {choice(inbounds)}

    def shp(s):
        rs = [p[0] for p in s]; cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
    while shp(obj) != (oh - 2, ow - 2):
        obj.add(choice([p for p in inbounds if p not in obj]))
    pat = np.full((oh, ow), sqc, dtype=int)
    targ = np.full((oh, ow), bgc, dtype=int)
    for (i, j) in obj:
        pat[i, j] = noisec
        targ[i, j] = noisec
    gi = np.full((h, w), bgc, dtype=int)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    gi[loci:loci + oh, locj:locj + ow] = pat
    boxc = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    near = set(boxc)
    for (i, j) in boxc:
        near |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
    inds = {(i, j) for i in range(1, h - 1) for j in range(1, w - 1) if gi[i, j] == bgc} - near
    namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    for (i, j) in sample(sorted(inds), namt):
        gi[i, j] = noisec
    tfs = [lambda a: a, lambda a: a.T, lambda a: np.rot90(a, 2).T,
           np.flipud, np.fliplr]
    targs = [f1(f2(targ)) for f1 in tfs for f2 in tfs]
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
    succ, tr = 0, 0
    while succ < noccs and tr < 5 * noccs:
        tr += 1
        t = choice(targs)
        hh, ww = t.shape
        cands = [p for p in inds if 1 <= p[0] <= h - hh - 1 and 1 <= p[1] <= w - ww - 1]
        if not cands:
            continue
        r, c = choice(sorted(cands))
        tpi = {(r + i, c + j) for i in range(hh) for j in range(ww)}
        if tpi <= inds:
            succ += 1
            inds -= tpi
            gi[r:r + hh, c:c + ww] = t
    go = _solve(gi, bgc, sqc)
    return {"input": gi[1:-1, 1:-1].tolist(), "output": go[1:-1, 1:-1].tolist()}


def _dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def _roles(G):
    # background = most common colour; box colour = the colour with the smallest extent
    cnt = Counter(G.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    best = None
    for col in cnt:
        if col == bgc:
            continue
        rr, cc = np.nonzero(G == col)
        ext = max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)
        if best is None or ext < best[0]:
            best = (ext, col)
    return bgc, best[1]


def _occurrence_rects(G, bgc, sqc):
    """Rectangles (in G coordinates, possibly over the edge) where the box's
    pattern recurs in any of the 8 orientations, box colour read as background."""
    H, W = G.shape
    rr, cc = np.nonzero(G == sqc)
    br, bc = rr.min(), cc.min()
    box = G[br:rr.max() + 1, bc:cc.max() + 1]
    P = np.full((H + 2, W + 2), bgc, dtype=int)
    P[1:-1, 1:-1] = G
    rects, seen = [], set()
    for B in _dihedral(box):
        T = np.where(B == sqc, bgc, B)
        th, tw = T.shape
        for r in range(H + 3 - th):
            for c in range(W + 3 - tw):
                if np.array_equal(P[r:r + th, c:c + tw], T):
                    key = (r - 1, c - 1, th, tw)
                    if key not in seen:
                        seen.add(key)
                        rects.append((key, B))
    return rects


def _solve(gi, bgc, sqc):
    go = gi.copy()
    H, W = gi.shape
    for (r, c, th, tw), B in _occurrence_rects(gi, bgc, sqc):
        for i in range(th):
            for j in range(tw):
                if 0 <= r + i < H and 0 <= c + j < W:
                    go[r + i, c + j] = B[i, j]
    return go


def derive_operations(I, O=None, examples=None):
    # O is never read: the box, its pattern and every place that pattern recurs
    # (any rotation/reflection, possibly running over the edge) are found in I.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc, sqc = _roles(I)
    G = I.copy()
    ops, sels = [], []
    for (r, c, th, tw), _B in _occurrence_rects(I, bgc, sqc):
        # one stroke per recurrence: the background inside its rectangle becomes
        # the box body; the pattern cells it frames are already in place
        cells = [(rr, cc) for rr in range(max(r, 0), min(r + th, H))
                 for cc in range(max(c, 0), min(c + tw, W)) if G[rr, cc] == bgc]
        if not cells:
            continue
        ops.append(int(sqc)); sels.append(sel_of(cells))
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
