"""
ARC Task: 4093f84a (RE-ARC) — LLM-generated grid_maker
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
    barc = random.choice(cols)
    bgc = random.choice([c for c in cols if c != barc])
    dotc = random.choice([c for c in cols if c not in (barc, bgc)])
    return {"bgc": bgc, "barc": barc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + round((b - a) * lb); hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    loci1, loci2 = sorted(random.sample(range(2, h - 2), 2))
    gi = [[bgc] * w for _ in range(h)]
    for ii in range(loci1, loci2 + 1):
        gi[ii] = [barc] * w
    go = [row[:] for row in gi]
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    for l1 in random.sample(range(w), num1):
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        for r in random.sample(range(0, loci1), k):
            gi[r][l1] = dotc
        for r in range(loci1 - k, loci1):
            go[r][l1] = barc
    for l2 in random.sample(range(w), num2):
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        for r in random.sample(range(loci2 + 1, h), k):
            gi[r][l2] = dotc
        for r in range(loci2 + 1, loci2 + 1 + k):
            go[r][l2] = barc
    if random.choice((True, False)):
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading (gravity sort per line): the band is a floor; every scattered dot
    is a grain that falls straight toward the band and piles up against it,
    becoming band material.  So: mark every dot as band material (one Color
    op on all dot-coloured cells), then, side by side, let every grain that is
    not yet resting on the band/pile drop one cell per step (one Move of all
    falling grains), restoring the cells they just vacated to background.
    Everything is measured from I; O is only used for the submit bbox.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: lives ONLY in complete, contiguous lines
    barc, vertical = None, False
    for c in colors:
        rs = [r for r in range(hi) if (Ia[r] == c).any()]
        if rs and len(rs) == rs[-1] - rs[0] + 1 and all((Ia[r] == c).all() for r in rs):
            barc, vertical = c, False
            break
    if barc is None:
        for c in colors:
            cs = [j for j in range(wi) if (Ia[:, j] == c).any()]
            if cs and len(cs) == cs[-1] - cs[0] + 1 and all((Ia[:, j] == c).all() for j in cs):
                barc, vertical = c, True
                break
    A = Ia.T if vertical else Ia            # canonical frame: band made of rows
    h, w = A.shape
    band = [r for r in range(h) if (A[r] == barc).all()]
    r1, r2 = band[0], band[-1]
    cnt = Counter(Ia.flatten().tolist())
    rest = [c for c in colors if c != barc]
    dotc = min(rest, key=lambda c: cnt[c])
    bgc = [c for c in rest if c != dotc][0]

    def real(r, c):
        return (c, r) if vertical else (r, c)
    # direction toward the band: side 0 = before the band, side 1 = after it
    OP_TOWARD = {0: (22 if vertical else 21), 1: (23 if vertical else 20)}

    ops, sels = [], []

    if barc == 0:
        # A Move only carries non-zero cells, so grains of band colour 0 cannot
        # be slid.  Paint each line's pile directly: a bar of k band cells
        # grown from the band edge, and the dots it did not swallow -> bgc.
        def seg(a_, b_, c_):
            return sel_of([real(r, c_) for r in range(a_, b_ + 1)])
        for c in range(w):
            for side in (0, 1):
                if side == 0:
                    dots = [r for r in range(r1) if A[r, c] == dotc]
                    k = len(dots)
                    if not k:
                        continue
                    ops.append(0); sels.append(seg(r1 - k, r1 - 1, c))
                    left = [r for r in dots if r < r1 - k]
                else:
                    dots = [r for r in range(r2 + 1, h) if A[r, c] == dotc]
                    k = len(dots)
                    if not k:
                        continue
                    ops.append(0); sels.append(seg(r2 + 1, r2 + k, c))
                    left = [r for r in dots if r > r2 + k]
                if left:
                    ops.append(int(bgc)); sels.append(sel_of([real(r, c) for r in left]))
        ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
        return ops, sels

    # 1. every dot becomes band material (all cells of the dot colour)
    dot_cells = [real(r, c) for r in range(h) for c in range(w) if A[r, c] == dotc]
    ops.append(int(barc)); sels.append(sel_of(dot_cells))

    # 2. gravity, one side of the band at a time: every grain not yet resting on
    #    the band (or on the pile against it) drops one cell per step.
    G = A.copy()
    G[A == dotc] = barc
    for side in (0, 1):
        while True:
            falling = []
            for c in range(w):
                if side == 0:
                    r = r1 - 1
                    while r >= 0 and G[r, c] == barc:
                        r -= 1
                    falling += [(rr, c) for rr in range(0, r) if G[rr, c] == barc]
                else:
                    r = r2 + 1
                    while r < h and G[r, c] == barc:
                        r += 1
                    falling += [(rr, c) for rr in range(r + 1, h) if G[rr, c] == barc]
            if not falling:
                break
            dr = 1 if side == 0 else -1
            ops.append(OP_TOWARD[side])
            sels.append(sel_of([real(r, c) for r, c in falling]))
            fs = set(falling)
            moved = {(r + dr, c) for r, c in falling}
            holes = sorted(fs - moved)
            for r, c in holes:
                G[r, c] = 0
            for r, c in moved:
                G[r, c] = barc
            if bgc != 0 and holes:
                # the cells the falling grains just left read 0 -> background
                ops.append(int(bgc))
                sels.append(sel_of([real(r, c) for r, c in holes]))
                for r, c in holes:
                    G[r, c] = bgc

    ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
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
                                f"for task 4093f84a"
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
                    f"Failed to build a complete episode for task 4093f84a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4093f84a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
