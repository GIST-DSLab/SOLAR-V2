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

# the eight ways a box can be laid down, the ARCLE ops that carry each one, and
# the straight mirror left over once a diagonal one has had to be drawn instead
_TF = [
    lambda a: np.array(a),                 # 0 as it is
    lambda a: np.fliplr(a),                # 1 vmirror
    lambda a: np.flipud(a),                # 2 hmirror
    lambda a: np.rot90(a, 2),              # 3 rot180
    lambda a: np.rot90(a, 1),              # 4 quarter turn ccw
    lambda a: np.rot90(a, 3),              # 5 quarter turn cw
    lambda a: np.array(a).T,               # 6 dmirror = flipud(ccw)
    lambda a: np.rot90(np.array(a).T, 2),  # 7 cmirror = fliplr(ccw)
]
_TOPS = [[], [26], [27], [26, 27], [24], [25], [24, 27], [24, 26]]
_RESID = {6: [], 4: [27], 5: [26], 7: [26, 27]}   # left to do after a drawn dmirror
_APPLY = {26: lambda a: np.fliplr(a), 27: lambda a: np.flipud(a),
          24: lambda a: np.rot90(a, 1), 25: lambda a: np.rot90(a, 3)}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _find_box(I):
    """The box in I: a solid rectangle of one colour with a pattern inside it.

    Everything the rule needs is read off here and nowhere else — which colour
    frames the box, which colour draws the pattern, which colour the grid is
    made of, and how big the box is."""
    I = np.asarray(I, dtype=int)
    cnt = Counter(I.ravel().tolist())
    bgc = cnt.most_common(1)[0][0]
    best = None
    for col in cnt:
        if col == bgc:
            continue
        rs, cs = np.where(I == col)
        r0, c0 = int(rs.min()), int(cs.min())
        h, w = int(rs.max()) - r0 + 1, int(cs.max()) - c0 + 1
        if h < 3 or w < 3:
            continue
        blk = I[r0:r0 + h, c0:c0 + w]
        if not (np.all(blk[0] == col) and np.all(blk[-1] == col)
                and np.all(blk[:, 0] == col) and np.all(blk[:, -1] == col)):
            continue                      # not a solid frame of this colour
        inner = blk[1:-1, 1:-1]
        others = sorted(set(inner.ravel().tolist()) - {col})
        if len(others) != 1:
            continue                      # the pattern is drawn in one colour
        nz = (inner == others[0])
        if not (nz[0].any() and nz[-1].any() and nz[:, 0].any() and nz[:, -1].any()):
            continue                      # and it fills the box interior
        key = (max(h, w), h * w, r0, c0)
        if best is None or key < best[0]:
            best = (key, col, others[0], r0, c0, h, w)
    if best is None:
        return None
    _, sqc, noisec, r0, c0, h, w = best
    return bgc, sqc, noisec, r0, c0, h, w


def _occurrences(I, bgc, sqc, model):
    """Every place the box's bare pattern sits on clean grid, and how it lies.

    The grid is read with one row/column of background around it, the way a box
    may hang over the edge; a position comes back in the grid's own coordinates
    and may be negative."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    pad = np.full((H + 2, W + 2), bgc, dtype=int)
    pad[1:-1, 1:-1] = I
    res = {}
    for k in range(8):
        blk = _TF[k](model)
        tgt = np.where(blk == sqc, bgc, blk)   # the box with its frame taken away
        th, tw = tgt.shape
        for pr in range(H + 3 - th):
            for pc in range(W + 3 - tw):
                if (pr - 1, pc - 1) in res:
                    continue
                if np.array_equal(pad[pr:pr + th, pc:pc + tw], tgt):
                    res[(pr - 1, pc - 1)] = k
    return sorted(res.items())


def _complete(I):
    """I with a box drawn around every occurrence of its pattern."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    info = _find_box(I)
    if info is None:
        return I.copy(), None, []
    bgc, sqc, noisec, br, bc, oh, ow = info
    model = I[br:br + oh, bc:bc + ow].copy()
    occ = _occurrences(I, bgc, sqc, model)
    G = I.copy()
    for (r, c), k in occ:
        blk = _TF[k](model)
        th, tw = blk.shape
        for rr in range(max(r, 0), min(r + th, H)):
            for cc in range(max(c, 0), min(c + tw, W)):
                G[rr, cc] = blk[rr - r, cc - c]
    return G, info, occ


def sample_colors(num_examples=None) -> dict:
    # all three roles are fixed for the episode: which colour the grid is, which
    # draws the pattern, which frames the box
    cols = list(range(10))
    bgc, noisec, sqc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    hub = max(12, min(32, max_h + 2))     # a grid is trimmed by one all round
    wub = max(12, min(32, max_w + 2))
    for _attempt in range(60):
        h = _unifint(diff_lb, diff_ub, (min(12, hub), hub))
        w = _unifint(diff_lb, diff_ub, (min(12, wub), wub))
        oh = _unifint(diff_lb, diff_ub, (3, max(3, min(7, h // 3))))
        ow = _unifint(diff_lb, diff_ub, (3, max(3, min(7, w // 3))))
        if h < oh + 3 or w < ow + 3:
            continue

        interior = [(r, c) for r in range(1, oh - 1) for c in range(1, ow - 1)]
        obj = {random.choice(interior)}
        while True:
            rs = [p[0] for p in obj]
            cs = [p[1] for p in obj]
            if max(rs) - min(rs) == oh - 3 and max(cs) - min(cs) == ow - 3:
                break
            obj.add(random.choice([p for p in interior if p not in obj]))
        model = np.full((oh, ow), sqc, dtype=int)
        for (r, c) in obj:
            model[r, c] = noisec
        targ = np.where(model == sqc, bgc, model)

        gi = np.full((h, w), bgc, dtype=int)
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        gi[loci:loci + oh, locj:locj + ow] = model

        blocked = set()
        for r in range(loci, loci + oh):
            for c in range(locj, locj + ow):
                blocked.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    blocked.add((r + dr, c + dc))
        inds = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                if (r, c) not in blocked}
        if len(inds) < 4 * oh * ow:
            continue
        namt = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
        for p in random.sample(sorted(inds), min(namt, len(inds))):
            gi[p] = noisec

        noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
        succ, tr, maxtr = 0, 0, 5 * noccs
        while succ < noccs and tr < maxtr:
            tr += 1
            t = _TF[random.randrange(8)](targ)
            th, tw = t.shape
            cands = [ij for ij in sorted(inds)
                     if 1 <= ij[0] <= h - th - 1 and 1 <= ij[1] <= w - tw - 1]
            if not cands:
                continue
            r0, c0 = random.choice(cands)
            fp = {(r0 + i, c0 + j) for i in range(th) for j in range(tw)}
            if fp <= inds:
                succ += 1
                inds -= fp
                gi[r0:r0 + th, c0:c0 + tw] = t
        if succ == 0:
            continue

        I = gi[1:-1, 1:-1]
        O, info, occ = _complete(I)
        if info is None or not occ:
            continue
        if info[:3] != (bgc, sqc, noisec) or info[5:] != (oh, ow):
            continue
        H2, W2 = I.shape
        if any(r < 0 or c < 0 or r + _TF[k](model).shape[0] > H2
               or c + _TF[k](model).shape[1] > W2 for (r, c), k in occ):
            continue                      # keep every drawn box whole
        ops, _sels = derive_operations(I, O)
        if not (set(ops) & {24, 25, 26, 27}):
            continue                      # the episode must show a box turned over
        return {"input": I.tolist(), "output": O.tolist()}
    raise ValueError("could not lay out an instance")


def derive_operations(I, O):
    # O is never read: the box, the pattern, the places it recurs and the way
    # each one lies are all measured from I.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    info = _find_box(I)
    if info is not None:
        bgc, sqc, noisec, br, bc, oh, ow = info
        model = I[br:br + oh, bc:bc + ow].copy()
        G = I.copy()
        copied = False
        for (r, c), k in _occurrences(I, bgc, sqc, model):
            blk = _TF[k](model)
            th, tw = blk.shape
            inside = (r >= 0 and c >= 0 and r + th <= H and c + tw <= W)
            if inside and np.array_equal(G[r:r + th, c:c + tw], blk):
                continue                  # a box already drawn here covers this one
            # ARCLE turns a region a quarter only when that region is square
            turnable = (k < 4 or oh == ow)
            if inside and turnable and bool((model != 0).all()):
                if not copied:            # the box, exactly its own rectangle
                    ops.append(28); sels.append([br, bc, oh - 1, ow - 1])
                    copied = True
                ops.append(30); sels.append([r, c, 0, 0])   # stamp it over the pattern
                cur = model.copy()
                G[r:r + oh, c:c + ow] = cur
                resid = _TOPS[k]
            else:
                if not inside:
                    u, resid = k, []      # a box over the edge cannot be turned in place
                elif turnable:
                    u, resid = 0, _TOPS[k]
                else:
                    u, resid = 6, _RESID[k]   # draw the diagonal, mirror the rest
                U = _TF[u](model)
                uh, uw = U.shape
                box = [(rr, cc) for rr in range(max(r, 0), min(r + uh, H))
                       for cc in range(max(c, 0), min(c + uw, W))]
                fill = [p for p in box if G[p] != sqc]
                if fill:                  # the body of the box, pattern included
                    ops.append(int(sqc)); sels.append(sel_of(fill))
                    for p in fill:
                        G[p] = sqc
                nz = [p for p in box
                      if U[p[0] - r, p[1] - c] == noisec and G[p] != noisec]
                if nz:                    # the pattern back on top of it
                    ops.append(int(noisec)); sels.append(sel_of(nz))
                    for p in nz:
                        G[p] = noisec
                cur = U
            for op in resid:              # turn it the way this copy lies
                nxt = _APPLY[op](cur)
                if np.array_equal(nxt, cur):
                    continue              # this orientation is already reached
                # exactly the box's own rectangle, its background included
                ops.append(op)
                sels.append([r, c, cur.shape[0] - 1, cur.shape[1] - 1])
                cur = nxt
                G[r:r + cur.shape[0], c:c + cur.shape[1]] = cur

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
