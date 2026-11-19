"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, fgc, c1, c2 = random.sample(cols, 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    od = unifint(diff_lb, diff_ub, (2, min(4, min(h, w))))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = randint(0, h - od)
    locj = randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {'input': gi, 'output': go}


# --- the square symmetries ARCLE can perform, cheapest chain first ------------
#     26 = FlipH (fliplr), 27 = FlipV (flipud), 25 = Rotate270 (CW), 24 = Rotate90 (CCW)
_SYMS = [
    ([],       lambda a: a),
    ([26],     lambda a: np.fliplr(a)),
    ([27],     lambda a: np.flipud(a)),
    ([25],     lambda a: np.rot90(a, 3)),
    ([24],     lambda a: np.rot90(a, 1)),
    ([26, 27], lambda a: np.rot90(a, 2)),
    ([25, 26], lambda a: np.fliplr(np.rot90(a, 3))),   # transpose
    ([24, 26], lambda a: np.fliplr(np.rot90(a, 1))),   # anti-transpose
]


def _upscale(a, f):
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)


def _turn_of(src, dst):
    """shortest chain of ARCLE ops turning square array src into dst, else None"""
    for sops, fn in _SYMS:
        if np.array_equal(fn(src), dst):
            return sops
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    G = I.copy()

    def paint_cells(cells, col):
        """Color<col> over the cells that do not already hold it."""
        cells = [(r, c) for r, c in cells if G[r, c] != col]
        if cells:
            ops.append(int(col))
            sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = col

    def turn_square(r0, c0, S, op):
        """ops 24-27 on the whole S x S square at (r0, c0) — a rigid reflection or
        quarter turn of that region, background included (bbox == the region)."""
        ops.append(int(op))
        sels.append(sel_of([(r0 + i, c0 + j) for i in range(S) for j in range(S)]))
        blk = G[r0:r0 + S, c0:c0 + S]
        G[r0:r0 + S, c0:c0 + S] = {26: np.fliplr, 27: np.flipud,
                                   25: lambda a: np.rot90(a, 3),
                                   24: lambda a: np.rot90(a, 1)}[op](blk)

    diff = (I != O)
    if diff.any():
        # the cells the rule adds always wear the template's body colour
        fgc = int(Counter(O[diff].tolist()).most_common(1)[0][0])
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

        # the template is the only thing in I wearing the body colour, and its
        # body always reaches all four sides, so its bbox is the template square
        fr, fc = np.where(I == fgc)
        tr, tc = int(fr.min()), int(fc.min())
        od = max(int(fr.max()) - tr + 1, int(fc.max()) - tc + 1)
        tr, tc = min(tr, h - od), min(tc, w - od)
        key = I[tr:tr + od, tc:tc + od]

        # every square of O that is the template scaled 1..4 and reflected/turned
        found = {}
        for fac in range(1, 5):
            S = od * fac
            if S > h or S > w:
                continue
            up = _upscale(key, fac)
            for sops, fn in _SYMS:
                pat = fn(up)
                head = pat[0, 0]
                for rr in range(h - S + 1):
                    for cc in range(w - S + 1):
                        if O[rr, cc] != head:
                            continue
                        if not diff[rr:rr + S, cc:cc + S].any():
                            continue        # nothing to do here (e.g. the template)
                        if (I[rr:rr + S, cc:cc + S] == fgc).any():
                            continue        # a copy carries no body cell in I
                        if np.array_equal(O[rr:rr + S, cc:cc + S], pat):
                            found.setdefault((rr, cc, S), fac)
        sites, taken = [], np.zeros((h, w), dtype=bool)
        for (rr, cc, S) in sorted(found, key=lambda t: (-t[2], t[0], t[1])):
            if taken[rr:rr + S, cc:cc + S].any():
                continue
            taken[rr:rr + S, cc:cc + S] = True
            sites.append((rr, cc, S, found[(rr, cc, S)]))
        sites.sort(key=lambda t: (t[0], t[1]))

        # each copy: stamp the template over its square, then turn that square
        # until its two corner markers sit where the copy shows them
        for (rr, cc, S, fac) in sites:
            canon = _upscale(key, fac)              # the template, at this scale
            pat = O[rr:rr + S, cc:cc + S]           # the copy, in its orientation
            turn = _turn_of(canon, pat)
            body = [(rr + i, cc + j) for i in range(S) for j in range(S)
                    if canon[i, j] == fgc]
            if turn is None:                        # no square symmetry fits: fill in
                paint_cells([(rr + i, cc + j) for i in range(S) for j in range(S)
                             if pat[i, j] == fgc], fgc)
                continue
            for col in [bgc] + sorted({int(v) for v in canon.flatten()} - {bgc, fgc}):
                paint_cells([(rr + i, cc + j) for i in range(S) for j in range(S)
                             if canon[i, j] == col], col)     # ground, then markers
            paint_cells(body, fgc)                            # then the body
            for op in turn:
                turn_square(rr, cc, S, op)

        # anything the scan could not account for, filled in per colour
        rest = (G != O)
        if rest.any():
            for col in sorted({int(v) for v in O[rest]}):
                paint_cells([(int(r), int(c))
                             for r, c in zip(*np.where(rest & (O == col)))], col)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
