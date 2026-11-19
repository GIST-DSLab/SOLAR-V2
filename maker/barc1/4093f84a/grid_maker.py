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


# Discrete structural variant: the generator's final `dmirror` coin-flip decides whether
# the band is horizontal (rows) or vertical (columns).
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, barc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])
    cap_h = max(7, max_w if mirrored else max_h)
    cap_w = max(7, max_h if mirrored else max_w)
    h = unifint(diff_lb, diff_ub, (7, cap_h))
    w = unifint(diff_lb, diff_ub, (7, cap_w))
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
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Reading: a solid band of `barc` lines is a magnet. Every scattered `dotc`
    pixel is pulled straight toward the band (perpendicular to it) until it
    collides with the band or with pixels already resting against it, then
    turns `barc`. Pixels stack on contact. Every displacement is a real Move
    chain (one grab + empty-selection steps).
    - barc != 0: nearest pixel first, each slides until it touches the stack.
    - barc == 0: the outermost pixel slides until it hits the next one, they
      stick and slide on together, until the train meets the band (a vacated
      cell reads 0 == barc, so this keeps every slide visibly necessary).
    Colour change: dotc != 0 -> pixels move as dotc and each finished stack is
    FloodFilled to barc; dotc == 0 (ARCLE cannot move 0) -> recolour first.
    Everything is measured from I; O only gives the submit bbox.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: the colour that lives ONLY in complete, contiguous lines
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

    A = Ia.T if vertical else Ia          # canonical frame: band made of rows
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    cnt = Counter(Ia.flatten().tolist())
    rest = [c for c in colors if c != barc]
    dotc = min(rest, key=lambda c: cnt[c])
    bgc = [c for c in rest if c != dotc][0]

    def real(r, c):
        return (c, r) if vertical else (r, c)

    # canonical "down" (toward band from above) / "up" (toward band from below)
    OP_DOWN = 22 if vertical else 21
    OP_UP = 23 if vertical else 20

    G = Ia.copy()                          # mirror of the working grid
    all_dots = [real(r, c) for c in range(w) for r in range(h) if A[r, c] == dotc]

    ops, sels = [], []
    mover = dotc
    if dotc == 0:                          # 0 can't be moved: recolour first
        ops.append(int(barc)); sels.append(sel_of(all_dots))
        for (r, c) in all_dots:
            G[r, c] = barc
        mover = barc

    def slide(cells, op, n):
        """grab `cells` once, then n-1 empty-selection steps"""
        ops.append(op); sels.append(sel_of(cells))
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))

    for c in range(w):
        for side in (0, 1):
            if side == 0:     # above the band: pulled down (canonical)
                dots = sorted([r for r in range(0, r1) if A[r, c] == dotc], reverse=True)
                front, step, op = r1 - 1, -1, OP_DOWN
            else:             # below the band: pulled up (canonical)
                dots = sorted([r for r in range(r2 + 1, h) if A[r, c] == dotc])
                front, step, op = r2 + 1, 1, OP_UP
            if not dots:
                continue
            vacated = set()

            def do_slide(rows, n):
                cells = [real(r, c) for r in rows]
                slide(cells, op, n)
                new_rows = [r - step * n for r in rows]
                for p in cells:
                    G[p] = 0
                for r in new_rows:
                    G[real(r, c)] = mover
                vacated.update(set(cells) - {real(r, c) for r in new_rows})
                return new_rows

            if barc != 0:
                # nearest pixel first: each slides until it touches the band
                # or the pixels already stacked against it
                for r in dots:
                    n = abs(r - front)
                    if n > 0:
                        do_slide([r], n)
                    front += step
            else:
                # band colour 0: outermost pixel slides until it hits the next
                # pixel, they stick and slide on together, until the band
                order = dots[::-1]                       # farthest first
                train = [order[0]]
                band_row = front - step
                for nxt in order[1:] + [band_row]:
                    inner = train[-1]
                    n = abs(inner - nxt) - 1
                    if n > 0:
                        train = do_slide(train, n)
                    if nxt != band_row:
                        train = train + [nxt]
            # repair the pixels' starting cells left empty (bgc != 0)
            if bgc != 0:
                hole = sorted(p for p in vacated if G[p] == 0)
                if hole:
                    ops.append(int(bgc)); sels.append(sel_of(hole))
                    for p in hole:
                        G[p] = bgc

    if dotc != 0:
        # every pulled pixel turns band colour: one FloodFill per stack (a
        # connected run of dotc now resting against the band), seeded at the
        # stack cell touching the band.
        seen = set()
        for c in range(w):
            for r in (r1 - 1, r2 + 1):
                if not (0 <= r < h):
                    continue
                p = real(r, c)
                if G[p] != dotc or p in seen:
                    continue
                stack, todo = {p}, [p]
                while todo:
                    y, x = todo.pop()
                    for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= q[0] < hi and 0 <= q[1] < wi and q not in stack and G[q] == dotc:
                            stack.add(q); todo.append(q)
                seen |= stack
                ops.append(10 + int(barc)); sels.append([p[0], p[1], 0, 0])
                for q in stack:
                    G[q] = barc

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
