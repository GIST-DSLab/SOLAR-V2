"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    for _attempt in range(60):
        rotf = random.choice((identity, rot90, rot180, rot270))
        glim_h, glim_w = (max_w, max_h) if rotf in (rot90, rot270) else (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (min(10, glim_h), glim_h))
        w = unifint(diff_lb, diff_ub, (min(10, glim_w), glim_w))
        if h < 6 or w < 6:
            continue
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
        go = canvas(bgc, (h, w))
        locs = set()
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci = (a + 1) + hp * a
                locj = (b + 1) + wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
        numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ = 0
        tr = 0
        maxtr = 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = random.randint(2, 6)
            pw = random.randint(2, 6)
            loci = random.randint(0, h - ph)
            locj = random.randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                        len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        if succ == 0:
            continue
        gi, go = rotf(gi), rotf(go)
        # keep only instances whose lattice tile is fully readable from the input
        Ia, Oa = np.asarray(gi, dtype=int), np.asarray(go, dtype=int)
        nc, vp, hp_, T = _analyse(Ia)
        if nc != noisec or (T < 0).any():
            continue
        rows, cols = np.indices(Oa.shape)
        if not np.array_equal(T[rows % vp, cols % hp_], Oa):
            continue
        return {'input': gi, 'output': go}
    raise ValueError("generation failed")


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                n += 1
                st = [(r, c)]; seen[r, c] = True
                while st:
                    a, b = st.pop()
                    for x, y in ((a+1, b), (a-1, b), (a, b+1), (a, b-1)):
                        if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                            seen[x, y] = True; st.append((x, y))
    return n


def _period(I, known, axis):
    n = I.shape[axis]
    for p in range(1, n):
        if axis == 0:
            a, b, ka, kb = I[p:], I[:-p], known[p:], known[:-p]
        else:
            a, b, ka, kb = I[:, p:], I[:, :-p], known[:, p:], known[:, :-p]
        both = ka & kb
        if both.any() and (a[both] == b[both]).all():
            return p
    return n


def _analyse(I):
    """Noise colour, lattice periods and the repeating tile, read from I alone."""
    h, w = I.shape
    best = None
    for col in sorted(set(I.flatten().tolist())):
        known = I != col
        if not known.any():
            continue
        vp, hp = _period(I, known, 0), _period(I, known, 1)
        rank = (vp * hp, _components(~known), int((~known).sum()))
        if best is None or rank < best[0]:
            best = (rank, col, vp, hp)
    _, nc, vp, hp = best
    known = I != nc
    T = -np.ones((vp, hp), dtype=int)
    for r in range(h):
        for c in range(w):
            if known[r, c] and T[r % vp, c % hp] < 0:
                T[r % vp, c % hp] = I[r, c]
    return nc, vp, hp, T


def derive_operations(I, O=None):
    """The grid is one tile repeated on a lattice; blots of a foreign colour
    cover parts of it.  Copy one intact lattice window, paste it over every
    damaged lattice window, then touch up what Paste cannot write (0s)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    nc, vp, hp, T = _analyse(I)
    noise = I == nc
    if not noise.any():
        return [34], [[0, 0, h - 1, w - 1]]

    def tile(r, c):
        return int(T[r % vp, c % hp])

    best = None
    # window = a x b lattice blocks; anchor phase (ar, ac)
    for a in range(1, max(1, h // vp) + 1):
        for b in range(1, max(1, w // hp) + 1):
            WH, WW = a * vp, b * hp
            for ar in range(vp):
                for ac in range(hp):
                    # intact source window on this lattice, fully inside the grid
                    src = None
                    for sr in range(ar, h - WH + 1, vp):
                        for sc in range(ac, w - WW + 1, hp):
                            if not noise[sr:sr + WH, sc:sc + WW].any():
                                src = (int(sr), int(sc)); break
                        if src: break
                    if src is None:
                        continue
                    blk = I[src[0]:src[0] + WH, src[1]:src[1] + WW]
                    # damaged lattice blocks a paste can repair (blot over a non-0 tile cell)
                    fixable = set()
                    for r, c in zip(*np.nonzero(noise)):
                        if r >= ar and c >= ac and tile(r, c) != 0:
                            fixable.add(((r - ar) // vp, (c - ac) // hp))
                    # greedy cover of damaged lattice blocks by a x b windows
                    left = set(fixable); places = []
                    while left:
                        i0, j0 = min(left)
                        cands = [(i0 - di, j0 - dj) for di in range(a) for dj in range(b)
                                 if i0 - di >= 0 and j0 - dj >= 0]
                        pi, pj = max(cands, key=lambda p: (sum(1 for (x, y) in left
                                     if p[0] <= x < p[0] + a and p[1] <= y < p[1] + b), -p[0], -p[1]))
                        places.append((int(ar + pi * vp), int(ac + pj * hp)))
                        left = {(x, y) for (x, y) in left
                                if not (pi <= x < pi + a and pj <= y < pj + b)}
                    G = I.copy()
                    for (pr, pc) in places:
                        sub = G[pr:pr + WH, pc:pc + WW]
                        bb = blk[:sub.shape[0], :sub.shape[1]]
                        sub[bb != 0] = bb[bb != 0]
                    rest_cols = {tile(r, c) for r, c in zip(*np.nonzero(G == nc))}
                    cost = (1 if places else 0) + len(places) + len(rest_cols)
                    key = (cost, WH * WW)
                    if best is None or key < best[0]:
                        best = (key, src, WH, WW, places)
    if best is None:
        places, src = [], None
    else:
        _, src, WH, WW, places = best

    G = I.copy()
    if places:
        # full rectangle: one intact lattice window, copied verbatim
        ops.append(28); sels.append([src[0], src[1], WH - 1, WW - 1])
        blk = I[src[0]:src[0] + WH, src[1]:src[1] + WW]
        for (pr, pc) in places:
            ops.append(30); sels.append([pr, pc, 0, 0])
            sub = G[pr:pr + WH, pc:pc + WW]
            bb = blk[:sub.shape[0], :sub.shape[1]]
            sub[bb != 0] = bb[bb != 0]
    # leftover blot cells (tile value 0, or no window reached them):
    # "blot cells whose lattice colour is k" -> Color k
    rest = {}
    for r, c in zip(*np.nonzero(G == nc)):
        rest.setdefault(tile(r, c), []).append((int(r), int(c)))
    for col in sorted(rest):
        ops.append(col); sels.append(sel_of(rest[col]))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
