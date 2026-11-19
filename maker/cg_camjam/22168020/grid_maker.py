"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour matters to the rule. Each object's colour is
    # sampled per instance, because the rule ("fill each row/column span of a
    # colour") works the same for any colour.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _connect(a, b):
    ai, aj = a
    bi, bj = b
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    if ai == bi:
        return {(ai, j) for j in range(sj, ej + 1)}
    if aj == bj:
        return {(i, aj) for i in range(si, ei + 1)}
    if bi - ai == bj - aj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(sj, ej + 1))}
    if bi - ai == aj - bj:
        return {(i, j) for i, j in zip(range(si, ei + 1), range(ej, sj - 1, -1))}
    return set()


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    h = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_w))))
    num = unifint(diff_lb, diff_ub, (1, max(1, min(9, (h * w) // 10))))
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    mfs = [
        lambda p: p,                           # identity
        lambda p: (p[1], p[0]),                # dmirror
        lambda p: (-p[1], -p[0]),              # cmirror (normalised below)
        lambda p: (-p[0], p[1]),               # hmirror (normalised below)
        lambda p: (p[0], -p[1]),               # vmirror (normalised below)
    ]

    succ, tr, maxtr = 0, 0, 6 * num
    while tr < maxtr and succ < num:
        tr += 1
        d = unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if not inds:
            continue
        loci, locj = random.choice(sorted(inds))
        io1 = _connect((loci, locj), (loci + d - 1, locj + d - 1))
        io2 = _connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        iol = sorted(io)
        oo = set()
        for a in iol:
            for b in iol:
                seg = _connect(a, b)
                if seg and len({p[0] for p in seg}) == 1:
                    oo |= seg
        mf = random.choice(mfs)
        # apply the mirror, then shift back so the upper-left corner stays put
        mio = [mf(p) for p in io]
        moo = [mf(p) for p in oo]
        allp = mio + moo
        r0 = min(p[0] for p in allp)
        c0 = min(p[1] for p in allp)
        ul_r = min(p[0] for p in io | oo)
        ul_c = min(p[1] for p in io | oo)
        io_m = {(p[0] - r0 + ul_r, p[1] - c0 + ul_c) for p in mio}
        oo_m = {(p[0] - r0 + ul_r, p[1] - c0 + ul_c) for p in moo}
        if not remcols:
            break
        col = random.choice(remcols)
        if oo_m.issubset(inds):
            for (i, j) in io_m:
                gi[i][j] = col
            for (i, j) in oo_m:
                go[i][j] = col
            succ += 1
            inds -= oo_m
            remcols = [c for c in remcols if c != col]
    if succ == 0:
        raise ValueError("no object placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # The background is the colour the generator fills the canvas with. It is
    # the dominant colour because the objects are thin.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    grid = I.copy()

    # Reading (the scanline view): every line through an object holds two
    # cells of its colour, and the gap between them gets filled. Here the unit
    # is "the span of one object on one line", filled as a single op from the
    # first cell to the last. A shape can open horizontally (row spans) or,
    # after dmirror/cmirror, vertically (column spans). Each colour is one
    # object, so we fill one object completely before starting the next.
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})
    for col in colors:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        rows = sorted({r for r, _ in cells})
        cols_ = sorted({c for _, c in cells})
        # row spans, top to bottom
        for r in rows:
            cs = [c for rr, c in cells if rr == r]
            if len(cs) < 2:
                continue
            seg = [(r, c) for c in range(min(cs), max(cs) + 1)]
            if all(grid[p] == col for p in seg):
                continue
            ops.append(col); sels.append(sel_of(seg))
            for p in seg:
                grid[p] = col
        # column spans, left to right
        for c in cols_:
            rs = [r for r, cc in cells if cc == c]
            if len(rs) < 2:
                continue
            seg = [(r, c) for r in range(min(rs), max(rs) + 1)]
            if all(grid[p] == col for p in seg):
                continue
            ops.append(col); sels.append(sel_of(seg))
            for p in seg:
                grid[p] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
