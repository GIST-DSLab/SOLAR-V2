"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of

# ── the four concentric "rings" of the 5x5 motif ─────────────────────────────
_R1 = ((0, 0), (0, 4), (4, 0), (4, 4))          # outer corners
_R2 = ((2, 0), (0, 2), (4, 2), (2, 4))          # outer edge midpoints
_R3 = ((1, 1), (3, 1), (1, 3), (3, 3))          # inner corners
_R4 = ((2, 2),)                                 # centre
_RINGS_IN_OUT = [                               # inner -> outer, for op emission
    [(1, 1), (1, 3), (3, 1), (3, 3)],
    [(0, 2), (2, 0), (2, 4), (4, 2)],
    [(0, 0), (0, 4), (4, 0), (4, 4)],
]
_CORE = [(r, c) for r in range(5) for c in range(5)]
_PERIM5 = [(r, c) for (r, c) in _CORE if r in (0, 4) or c in (0, 4)]
_INNER5 = [(r, c) for r in range(1, 4) for c in range(1, 4) if r in (1, 3) or c in (1, 3)]
_ODD5 = [(r, c) for (r, c) in _CORE if (r + c) % 2 == 1]


def _is_motif(I, bgc, i, j):
    """Exactly the verifier's acceptance test for a 5x5 motif whose ulcorner is (i, j)."""
    hi, wi = I.shape
    if i - 1 < 0 or j - 1 < 0 or i + 5 > hi - 1 or j + 5 > wi - 1:
        return False
    for c in range(j - 1, j + 6):                      # 7x7 halo must be pure background
        if I[i - 1, c] != bgc or I[i + 5, c] != bgc:
            return False
    for r in range(i - 1, i + 6):
        if I[r, j - 1] != bgc or I[r, j + 5] != bgc:
            return False
    if I[i + 2, j + 2] == bgc:                          # centre lit
        return False
    if not any(I[i + r, j + c] != bgc for r, c in _PERIM5):     # outer band used
        return False
    if not any(I[i + r, j + c] != bgc for r, c in _INNER5):     # inner band used
        return False
    if any(I[i + r, j + c] != bgc for r, c in _ODD5):           # odd parity always empty
        return False
    pts = [(r, c) for r, c in _CORE if I[i + r, j + c] != bgc]
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return max(rs) - min(rs) == 4 and max(cs) - min(cs) == 4     # spans the full 5x5


def _detect(I, bgc):
    hi, wi = I.shape
    return [(i, j) for i in range(1, hi - 5) for j in range(1, wi - 5)
            if _is_motif(I, bgc, i, j)]


def _closure(I, bgc):
    """Symmetric completion of every detected motif; ok=False on colour conflicts."""
    O = I.copy()
    assigned = {}
    ok = True
    for (i, j) in _detect(I, bgc):
        for (r, c) in _CORE:
            v = I[i + r, j + c]
            if v == bgc:
                continue
            for (a, b) in [(r, c), (c, r), (4 - c, 4 - r), (4 - r, c), (r, 4 - c)]:
                key = (i + a, j + b)
                if key in assigned and assigned[key] != v:
                    ok = False
                assigned[key] = v
                O[key] = v
    return O, ok


def _detectable(block, bgc):
    if block[2, 2] == bgc:
        return False
    if not any(block[r, c] != bgc for r, c in _PERIM5):
        return False
    if not any(block[r, c] != bgc for r, c in _INNER5):
        return False
    pts = [(r, c) for r, c in _CORE if block[r, c] != bgc]
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return max(rs) - min(rs) == 4 and max(cs) - min(cs) == 4


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def _one(diff_lb, diff_ub, max_h, max_w, bgc, ccols, force=False):
    rings = [_R4, _R3, _R2, _R1]
    bx = backdrop(frozenset(_R1))
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    ccols = list(ccols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            go = fill(go, ringcols[0], plcdrings[0])
            idx = randint(1, 3)
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            go = fill(go, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
            remringcols = ringcols[1:idx] + ringcols[idx + 1:]
            numrs = unifint(diff_lb, diff_ub, (1, 2))
            if idx != 1:
                # the inner band must carry at least one cell or the motif is
                # not readable at all (the verifier ignores it entirely)
                numrs = 2
            locs = sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            ncells = [4 - unifint(diff_lb, diff_ub, (0, 3)) for _ in remrings]
            if force:
                ncells[0] = min(ncells[0], 3)
            tofillgi = merge(frozenset(
                recolor(col, frozenset(sample(totuple(remring), n)))
                for remring, col, n in zip(remrings, remringcols, ncells)))
            tofillgo = merge(frozenset(
                recolor(col, remring) for remring, col in zip(remrings, remringcols)))
            if min(shape(tofillgi)) == 5:
                cgi = paint(gi, tofillgi)
                blk = np.array(crop(cgi, loc, (5, 5)))
                if _detectable(blk, bgc):
                    succ += 1
                    gi = cgi
                    go = paint(go, tofillgo)
    return gi, go, succ


def _fallback(max_h, max_w, bgc, ccols):
    """Minimal guaranteed-valid instance: one motif with a half-drawn outer ring."""
    h, w = max(7, min(9, max_h)), max(7, min(9, max_w))
    ccols = list(ccols)
    ci, cm, co = (choice(ccols) for _ in range(3))
    i = randint(1, h - 6)
    j = randint(1, w - 6)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, cm, shift(_R4, (i, j)))
    gi = fill(gi, ci, shift(_R3, (i, j)))
    go = fill(gi, co, shift(_R1, (i, j)))
    gi = fill(gi, co, shift(frozenset({(0, 0), (4, 4)}), (i, j)))
    return {'input': gi, 'output': go}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    best = None
    for attempt in range(80):
        gi, go, succ = _one(diff_lb, diff_ub, max_h, max_w, bgc, ccols,
                            force=attempt >= 5)
        if succ == 0:
            continue
        I, G = np.array(gi), np.array(go)
        if Counter(I.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        O, ok = _closure(I, bgc)
        if not ok or not (O == G).all():
            continue                       # ambiguous / undetectable -> resample
        if not (I == G).all():
            return {'input': gi, 'output': go}
        best = {'input': gi, 'output': go}
    if best is not None:
        return best
    return _fallback(max_h, max_w, bgc, ccols)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    # one motif at a time; inside a motif, complete its concentric bands
    # from the centre outwards.  Every selection is read off I alone.
    for (i, j) in _detect(I, bgc):
        for ring in _RINGS_IN_OUT:
            present = [(r, c) for r, c in ring if I[i + r, j + c] != bgc]
            if len(present) in (0, 4):          # band unused, or already whole
                continue
            col = Counter(int(I[i + r, j + c]) for r, c in present).most_common(1)[0][0]
            missing = [(i + r, j + c) for (r, c) in ring if (r, c) not in present]
            ops.append(col)
            sels.append(sel_of(missing))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
