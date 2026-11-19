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

# the 8 orientations of a square box, and the ARCLE ops that produce each one
_TF = [
    lambda a: np.array(a),          # identity
    lambda a: np.fliplr(a),         # vmirror
    lambda a: np.flipud(a),         # hmirror
    lambda a: np.rot90(a, 2),       # rot180
    lambda a: np.rot90(a, 1),       # rot90 CCW
    lambda a: np.rot90(a, 3),       # rot90 CW
    lambda a: np.array(a).T,        # dmirror  (transpose)     = flipud(rot90CCW)
    lambda a: np.rot90(a, 2).T,     # cmirror  (anti-transpose)= flipud(rot90CW)
]
_TOPS = [[], [26], [27], [26, 27], [24], [25], [24, 27], [25, 27]]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # box colour and pattern colour stay non-zero so Copy/Paste can carry the box
    sqc = random.choice([c for c in cols if c != bgc and c != 0])
    noisec = random.choice([c for c in cols if c not in (bgc, sqc, 0)])
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    hub = min(32, max_h + 2)          # grids are trimmed by 1 on every side
    hlb = min(12, hub)
    wub = min(32, max_w + 2)
    wlb = min(12, wub)
    h = _unifint(diff_lb, diff_ub, (hlb, hub))
    w = _unifint(diff_lb, diff_ub, (wlb, wub))
    odub = max(4, min(7, h // 3, w // 3))
    od = _unifint(diff_lb, diff_ub, (4, odub))     # square box -> every orientation is square
    if h < od + 3 or w < od + 3:
        raise ValueError("grid too small for the box")

    interior = [(r, c) for r in range(1, od - 1) for c in range(1, od - 1)]
    obj = {random.choice(interior)}
    while True:
        rs = [p[0] for p in obj]
        cs = [p[1] for p in obj]
        if max(rs) - min(rs) == od - 3 and max(cs) - min(cs) == od - 3:
            break
        obj.add(random.choice([p for p in interior if p not in obj]))

    pat = np.full((od, od), sqc, dtype=int)        # source: box colour + pattern
    targ = np.full((od, od), bgc, dtype=int)       # target: background + pattern
    for (r, c) in obj:
        pat[r, c] = noisec
        targ[r, c] = noisec

    gi = np.full((h, w), bgc, dtype=int)
    loci = random.randint(1, h - od - 1)
    locj = random.randint(1, w - od - 1)
    gi[loci:loci + od, locj:locj + od] = pat

    blocked = set()
    for r in range(loci, loci + od):
        for c in range(locj, locj + od):
            blocked.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                blocked.add((r + dr, c + dc))
    inds = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if gi[r, c] == bgc and (r, c) not in blocked:
                inds.add((r, c))
    if len(inds) < 4 * od * od:
        raise ValueError("not enough free space")

    namt = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    for (r, c) in random.sample(sorted(inds), min(namt, len(inds))):
        gi[r, c] = noisec

    targs = [_TF[k](targ) for k in range(8)]
    sours = [_TF[k](pat) for k in range(8)]

    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od * od * 4))))
    succ, tr, maxtr = 0, 0, 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        k = random.randrange(8)
        cands = [ij for ij in sorted(inds)
                 if 1 <= ij[0] <= h - od - 1 and 1 <= ij[1] <= w - od - 1]
        if not cands:
            break
        r0, c0 = random.choice(cands)
        fp = {(r0 + i, c0 + j) for i in range(od) for j in range(od)}
        if fp <= inds:
            succ += 1
            inds -= fp
            gi[r0:r0 + od, c0:c0 + od] = targs[k]

    # every place (any orientation) where the bare pattern sits on clean background
    occ = {}
    for k in range(8):
        t = targs[k]
        for r in range(h - od + 1):
            for c in range(w - od + 1):
                if (r, c) in occ:
                    continue
                if np.array_equal(gi[r:r + od, c:c + od], t):
                    occ[(r, c)] = k
    if not occ:
        raise ValueError("no occurrence of the pattern")

    used = set()
    for (r, c), k in sorted(occ.items()):
        # keep every drawn box whole and unambiguous after the trim
        if not (1 <= r and 1 <= c and r + od <= h - 1 and c + od <= w - 1):
            raise ValueError("occurrence would be clipped by the trim")
        fp = {(r + i, c + j) for i in range(od) for j in range(od)}
        if fp & used:
            raise ValueError("overlapping occurrences")
        used |= fp

    go = gi.copy()
    for (r, c), k in sorted(occ.items()):
        go[r:r + od, c:c + od] = sours[k]

    return {"input": gi[1:-1, 1:-1].tolist(), "output": go[1:-1, 1:-1].tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    # box colour = the colour with the most compact bounding box (bgc and the
    # pattern colour are both scattered over the whole grid)
    sqc, best = None, None
    for col in sorted(set(I.flatten().tolist())):
        rs, cs = np.where(I == col)
        ext = max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1)
        if best is None or ext < best:
            best, sqc = ext, col
    rs, cs = np.where(I == sqc)
    br, bc = int(rs.min()), int(cs.min())
    od = int(max(rs.max() - br + 1, cs.max() - bc + 1))
    od = min(od, ho - br, wo - bc)
    P = I[br:br + od, bc:bc + od]            # the box, as it stands in the input

    # each copy of the box in O, together with the orientation it is shown in
    boxes = []
    for r in range(ho - od + 1):
        for c in range(wo - od + 1):
            win = O[r:r + od, c:c + od]
            if np.array_equal(win, I[r:r + od, c:c + od]):
                continue                      # nothing new here (e.g. the original box)
            for k in range(8):
                if np.array_equal(win, _TF[k](P)):
                    boxes.append((r, c, k))
                    break

    G = I.copy()
    copied = False
    for (r, c, k) in boxes:
        want = _TF[k](P)
        if np.array_equal(G[r:r + od, c:c + od], want):
            continue
        if not copied:
            # CopyI the original box; selection is exactly its full od x od rectangle
            ops.append(28)
            sels.append([br, bc, od - 1, od - 1])
            copied = True
        if (P == 0).any():
            # Paste cannot carry 0s: lay a base of the box colour over the whole
            # destination rectangle first, then draw the pattern on top of it
            base = np.full((od, od), sqc, dtype=int)
            if not np.array_equal(G[r:r + od, c:c + od], base):
                ops.append(int(sqc))
                sels.append([r, c, od - 1, od - 1])   # exactly the destination rectangle
                G[r:r + od, c:c + od] = base
        # stamp the box over the pattern that matched here
        reg = G[r:r + od, c:c + od].copy()
        m = P != 0
        reg[m] = P[m]
        if not np.array_equal(reg, G[r:r + od, c:c + od]):
            ops.append(30)
            sels.append([r, c, 0, 0])
            G[r:r + od, c:c + od] = reg
        # mirror / rotate the stamped box into the orientation this copy appears in
        for op in _TOPS[k]:
            reg = G[r:r + od, c:c + od]
            if op == 26:
                nreg = np.fliplr(reg)
            elif op == 27:
                nreg = np.flipud(reg)
            elif op == 24:
                nreg = np.rot90(reg, 1)
            else:
                nreg = np.rot90(reg, 3)
            if np.array_equal(nreg, reg):
                continue                       # this orientation is already reached
            ops.append(op)
            sels.append([r, c, od - 1, od - 1])  # exactly the destination rectangle (square)
            G[r:r + od, c:c + od] = nreg

    if not np.array_equal(G, O):               # safety net; never used on valid instances
        for rr in range(ho):
            for cc in range(wo):
                if G[rr, cc] != O[rr, cc]:
                    ops.append(int(O[rr, cc]))
                    sels.append([rr, cc, 0, 0])
                    G[rr, cc] = O[rr, cc]

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
