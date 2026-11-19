"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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


# ──────────────────────────────────────────────────────────────────────────
#  Task a64e4611 — a noise field has a "corridor" wiped into it: a vertical
#  background band running to the bottom edge plus horizontal background
#  bands running to the left/right edge.  The output paints colour 3 over the
#  INTERIOR of that corridor: every corridor cell whose whole 3×3 neighbour-
#  hood is still corridor (the grid border does not count as a boundary).
#  Everything below is measured from the INPUT only.
# ──────────────────────────────────────────────────────────────────────────


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _corridor_fill(I):
    """Measure the wiped background corridor in I, return its eroded interior."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # 1) background colour: the noise field contains no solid 3x3 monochrome
    #    block (the generator breaks them all), so solid blocks only ever sit
    #    inside the wiped corridor.
    bgc, best = int(I[0, 0]), -1
    for col in np.unique(I):
        M = (I == col)
        n = 0
        for r in range(h - 2):
            for c in range(w - 2):
                if M[r:r + 3, c:c + 3].all():
                    n += 1
        if n > best:
            best, bgc = n, int(col)
    M = (I == bgc)

    # 2) the vertical trunk: the largest solid background rectangle (width >= 3)
    #    anchored on the bottom edge.
    up = np.zeros(w, dtype=int)
    for c in range(w):
        r = h - 1
        while r >= 0 and M[r, c]:
            up[c] += 1
            r -= 1
    trunk = None
    for c1 in range(w):
        mn = h + 1
        for c2 in range(c1, w):
            mn = min(mn, int(up[c2]))
            if mn == 0:
                break
            if c2 - c1 + 1 >= 3:
                area = mn * (c2 - c1 + 1)
                if trunk is None or area > trunk[0]:
                    trunk = (area, c1, c2, mn)

    C = np.zeros((h, w), dtype=bool)
    if trunk is not None:
        _, c1, c2, H = trunk
        r0 = h - H
        C[r0:h, c1:c2 + 1] = True
        # 3) horizontal arms: runs (>= 3 rows) of trunk rows that are solid
        #    background from the trunk all the way out to a side edge.
        for lo, hi, side in ((0, c1, 'L'), (c2 + 1, w, 'R')):
            if hi - lo < 1:
                continue
            rows = [r for r in range(r0, h) if M[r, lo:hi].all()]
            i = 0
            while i < len(rows):
                j = i
                while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                    j += 1
                if j - i + 1 >= 3:
                    a, b = rows[i], rows[j]
                    if side == 'L':
                        C[a:b + 1, 0:c1 + 1] = True
                    else:
                        C[a:b + 1, c1:w] = True
                i = j + 1

    # 4) interior = corridor eroded by one cell (grid border is not a boundary)
    F = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not C[r, c]:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if not C[rr, cc]:
                        ok = False
            F[r, c] = ok
    return F


def _bands(F):
    """Decompose the fill region into its maximal rectangular bands."""
    h, w = F.shape
    cands = []
    for r1 in range(h):
        if not F[r1].any():
            continue
        colok = np.ones(w, dtype=bool)
        for r2 in range(r1, h):
            colok = colok & F[r2]
            if not colok.any():
                break
            c = 0
            while c < w:
                if colok[c]:
                    e = c
                    while e + 1 < w and colok[e + 1]:
                        e += 1
                    up_ext = r1 > 0 and F[r1 - 1, c:e + 1].all()
                    dn_ext = r2 < h - 1 and F[r2 + 1, c:e + 1].all()
                    if not up_ext and not dn_ext:
                        cands.append((-(r2 - r1 + 1) * (e - c + 1), r1, r2, c, e))
                    c = e + 1
                else:
                    c += 1
    cands.sort()
    covered = np.zeros((h, w), dtype=bool)
    total = int(F.sum())
    out = []
    for (_, r1, r2, c1, c2) in cands:
        if covered[r1:r2 + 1, c1:c2 + 1].all():
            continue
        out.append((r1, r2, c1, c2))
        covered[r1:r2 + 1, c1:c2 + 1] = True
        if int(covered.sum()) == total:
            break
    return out


# ──────────────────────────────────────────────────────────────────────────
_VARIANTS = [
    {"mode": "L",   "spi_top": True},    # left arm,  trunk touches top edge
    {"mode": "R",   "spi_top": False},   # right arm, trunk starts mid-grid
    {"mode": "LR",  "spi_top": False},   # both arms on the same rows
    {"mode": "OPP", "spi_top": True},    # two arms, opposite sides, different rows
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        examples = [dict(v) for v in _VARIANTS]
        examples += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


def _build(diff_lb, diff_ub, max_h, max_w, bgc, noisec, mode, spi_top):
    h = _unifint(diff_lb, diff_ub, (min(18, max(10, max_h)), max_h))
    w = _unifint(diff_lb, diff_ub, (min(18, max(10, max_w)), max_w))
    if min(h, w) < 12:
        return None
    lb, ub = int(0.4 * h * w), int(0.5 * h * w)
    nbgc = _unifint(diff_lb, diff_ub, (lb, ub))

    gi = np.full((h, w), noisec, dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w)]
    for (r, c) in random.sample(cells, nbgc):
        gi[r, c] = bgc

    addn, addb = set(), set()
    for r in range(h - 2):
        for c in range(w - 2):
            blk = gi[r:r + 3, c:c + 3]
            if (blk == bgc).all():
                addn.add((r + random.randint(0, 2), c + random.randint(0, 2)))
            elif (blk == noisec).all():
                addb.add((r + random.randint(0, 2), c + random.randint(0, 2)))
    for (r, c) in addn:
        gi[r, c] = noisec
    for (r, c) in addb:
        gi[r, c] = bgc

    dim_ub = min(8, min(h, w) - 7)
    if dim_ub < 3:
        return None
    dim = random.randint(random.randint(3, dim_ub), dim_ub)
    hi_locj = min(h, w) - dim - 4
    if hi_locj < 3:
        return None
    locj = random.randint(3, hi_locj)
    spi = 0 if spi_top else random.randint(3, h // 2)

    go = gi.copy()
    gi[spi:h, locj:locj + dim] = bgc
    go[spi:h, locj:locj + dim] = bgc
    go[(spi + 1 if spi > 0 else spi):h, locj + 1:locj + dim - 1] = 3

    if mode == "L":
        rounds = [(-1,)]
    elif mode == "R":
        rounds = [(1,)]
    elif mode == "LR":
        rounds = [(-1, 1)]
    else:
        s = random.choice((-1, 1))
        rounds = [(s,), (-s,)]

    for sg_set in rounds:
        if spi + 3 <= h - 6:
            startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        else:
            startloc = spi
        if h - startloc - 3 < 3:
            return None
        hh = random.randint(3, min(8, h - startloc - 3))
        for sgn in sg_set:
            sl = slice(0, locj + 1) if sgn == -1 else slice(locj, w)
            for ii in range(startloc, startloc + hh):
                gi[ii, sl] = bgc
                seg = go[ii, sl]
                seg[seg != 3] = bgc
        for sgn in sg_set:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                if sgn == -1:
                    go[ii, 0:locj + dim - 1] = 3
                else:
                    go[ii, locj + 1:w] = 3

    ok = np.array_equal(_corridor_fill(gi), go == 3)
    return gi, go, ok


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, mode=None, spi_top=None) -> dict:
    if mode is None:
        mode = random.choice(["L", "R", "LR", "OPP"])
    if spi_top is None:
        spi_top = random.choice([True, False])
    last = None
    for _ in range(500):
        out = _build(diff_lb, diff_ub, max_h, max_w, bgc, noisec, mode, spi_top)
        if out is None:
            continue
        gi, go, ok = out
        last = {"input": gi.tolist(), "output": go.tolist()}
        if ok:                      # keep only unambiguous corridors
            return last
    return last


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Rule measured from the input: find the wiped background corridor
    # (bottom-anchored trunk + edge-reaching arms) and paint its interior.
    F = _corridor_fill(I)

    for (r1, r2, c1, c2) in _bands(F):
        cells = [(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]
        ops.append(3)                       # Color3 over one whole corridor band
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])       # full-grid rectangle: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
