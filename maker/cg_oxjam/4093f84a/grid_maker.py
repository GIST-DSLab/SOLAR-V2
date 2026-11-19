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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, barc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    lim = min(max_h, max_w) if transpose else None
    hmax = lim if transpose else max_h
    wmax = lim if transpose else max_w
    h = unifint(diff_lb, diff_ub, (7, hmax))
    w = unifint(diff_lb, diff_ub, (7, wmax))
    loci1, loci2 = sorted(sample(interval(2, h - 2, 1), 2))
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = sample(opts, num1)
    locs2 = sample(opts, num2)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = sample(interval(loci2 + 1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (the golf entry's): the band is a floor and the dots are loose
    grains under gravity pulling toward it.  On each side, every dot that has
    a free cell (or a falling dot) between it and the band drops one cell, all
    together, step after step, until everything rests in a stack on the band.
    The dots are band material, so they take the band colour.
      * band colour != 0: first paint every dot the band colour (one Color on
        all dot cells), then let them fall.
      * band colour == 0 (cannot ride a Move): each line's grains become a bar
        of the same count on the band; unreached grains dissolve to bgc.
    Each fall step is a Move of the group of dots that can still fall: grabbed
    once, continued with empty selections while the group is unchanged,
    re-grabbed when some dot has landed.  Cells the grains left behind read 0
    and are repaired to bgc once the side has settled.  Measured from I only.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: the colour living only in complete contiguous lines
    barc, vertical = None, False
    for c in colors:
        rs = [r for r in range(hi) if bool((Ia[r] == c).any())]
        if rs and len(rs) == rs[-1] - rs[0] + 1 and all(bool((Ia[r] == c).all()) for r in rs):
            barc, vertical = c, False
            break
    if barc is None:
        for c in colors:
            cs = [j for j in range(wi) if bool((Ia[:, j] == c).any())]
            if cs and len(cs) == cs[-1] - cs[0] + 1 and all(bool((Ia[:, j] == c).all()) for j in cs):
                barc, vertical = c, True
                break

    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    # dots: the minority of the two remaining colours; the other is background
    cnt = Counter(Ia.flatten().tolist())
    rest = [c for c in colors if c != barc]
    dotc = min(rest, key=lambda c: cnt[c])
    bgc = [c for c in rest if c != dotc][0] if len(rest) > 1 else None

    def real(r, c):
        return (c, r) if vertical else (r, c)

    # canonical direction toward the band on each side, and its ARCLE op
    if vertical:
        sides = [(+1, 22), (-1, 23)]   # left side -> MoveR, right side -> MoveL
    else:
        sides = [(+1, 21), (-1, 20)]   # top side -> MoveD, bottom side -> MoveU

    if barc == 0:
        # A 0-coloured grain cannot be carried by an ARCLE Move (Move only lifts
        # nonzero cells), so the fall cannot be shown as a translation here.
        # Each line's grains instead turn into a bar of the same count grown
        # from the band, and grains the bar does not reach dissolve to bgc.
        ops, sels = [], []
        for c in range(w):
            for d in (+1, -1):
                if d == +1:
                    dots = [r for r in range(0, r1) if A[r, c] == dotc]
                    bar = [r1 - 1 - i for i in range(len(dots))]
                else:
                    dots = [r for r in range(r2 + 1, h) if A[r, c] == dotc]
                    bar = [r2 + 1 + i for i in range(len(dots))]
                if not dots:
                    continue
                ops.append(int(barc)); sels.append(sel_of([real(r, c) for r in bar]))
                left = [r for r in dots if r not in bar]
                if left:
                    ops.append(int(bgc)); sels.append(sel_of([real(r, c) for r in left]))
        ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
        return ops, sels

    G = A.copy()                       # python mirror of the working grid
    ops, sels = [], []
    all_dots = [(r, c) for r in range(h) for c in range(w) if A[r, c] == dotc]
    # the grains become band material: one Color on every dot cell
    ops.append(int(barc)); sels.append(sel_of([real(r, c) for r, c in all_dots]))
    for r, c in all_dots:
        G[r, c] = barc
    movc = barc

    for d, mop in sides:
        if d == +1:
            dots = {(r, c) for (r, c) in all_dots if r < r1}
            floor = r1
        else:
            dots = {(r, c) for (r, c) in all_dots if r > r2}
            floor = r2
        if not dots:
            continue
        side_rows = range(0, r1) if d == +1 else range(r2 + 1, h)
        grabbed = None
        snap = None
        while True:
            # which dots can fall this step (synchronously, nearest the band first)
            moving = set()
            for (r, c) in sorted(dots, key=lambda p: -p[0] * d):
                nr = r + d
                if nr == floor:
                    continue
                if (nr, c) not in dots or (nr, c) in moving:
                    moving.add((r, c))
            if not moving:
                break
            if grabbed is not None and grabbed == moving:
                # same falling group: continue the grab (empty selection)
                ops.append(mop); sels.append(sel_of([]))
            else:
                # some grains landed: grab the group still falling
                ops.append(mop); sels.append(sel_of([real(r, c) for r, c in moving]))
                snap = G.copy()
                for (r, c) in moving:
                    snap[r, c] = 0
            newpos = {(r + d, c) for (r, c) in moving}
            G = snap.copy()
            for (r, c) in newpos:
                G[r, c] = movc
            dots = (dots - moving) | newpos
            grabbed = newpos
        # cells the grains left behind on this side read 0 now
        holes = [(r, c) for r in side_rows for c in range(w) if G[r, c] == 0]
        if bgc is not None and bgc != 0 and holes:
            ops.append(int(bgc)); sels.append(sel_of([real(r, c) for r, c in holes]))
            for (r, c) in holes:
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
