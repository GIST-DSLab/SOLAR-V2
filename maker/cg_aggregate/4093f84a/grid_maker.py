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
    # the dots become band pieces and are slid with Move ops; ARCLE cannot
    # move a 0-coloured object, so the band colour is non-zero
    barc = random.choice(range(1, 10))
    bgc, dotc = random.sample([c for c in cols if c != barc], 2)
    return {"bgc": bgc, "barc": barc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, dotc=None) -> dict:
    if bgc is None or barc is None or dotc is None:
        barc = random.choice(range(1, 10))
        bgc, dotc = random.sample([c for c in range(10) if c != barc], 2)
    h = unifint(diff_lb, diff_ub, (7, max_h))
    w = unifint(diff_lb, diff_ub, (7, max_w))
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
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (gravity, as in the golf entry): a solid band of `barc` lines splits
    the grid. Every dot is a piece of band material: first all dot-coloured cells
    take the band colour, then on each side every piece in a line perpendicular
    to the band FALLS toward the band until it rests on the band or on the piece
    already stacked there. Each piece slides with Move ops (one grab, then
    empty-selection steps), nearest piece first so it only glides over
    background; once a line has settled, the 0-holes its pieces left
    behind are repaired to bgc.
    Everything is measured from I (and the demos for the dot colour); O only gives
    the submit bbox.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: lives only in complete contiguous lines
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

    rest = [c for c in colors if c != barc]
    dotc = None
    if examples:
        # the dot colour is the one the demos make vanish entirely
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            gone = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            cand = gone if cand is None else cand & gone
        if cand:
            m = [c for c in rest if c in cand]
            if len(m) == 1:
                dotc = m[0]
    cnt = Counter(Ia.flatten().tolist())
    if dotc is None:
        dotc = min(rest, key=lambda c: cnt[c])
    others = [c for c in rest if c != dotc]
    bgc = others[0] if others else None

    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    def real(r, c):
        return (c, r) if vertical else (r, c)

    # canonical "down" (toward band from above) / "up" (from below)
    op_toward_from_above = 22 if vertical else 21
    op_toward_from_below = 23 if vertical else 20

    ops, sels = [], []
    # every dot is band material: all dot-coloured cells take the band colour
    # (if the band colour is 0 a 0-piece cannot be moved, so the dots slide in
    #  their own colour and take the band colour after landing)
    piece = barc if barc != 0 else dotc
    dot_cells = [real(r, c) for r in range(h) for c in range(w) if A[r, c] == dotc]
    landed = set()
    G = A.copy()          # tracks the visible grid (canonical frame)
    if barc != 0:
        ops.append(int(barc)); sels.append(sel_of(dot_cells))
        G[A == dotc] = barc
    for c in range(w):
        for side in (0, 1):
            if side == 0:
                dots = sorted([r for r in range(0, r1) if A[r, c] == dotc], reverse=True)
                targets = [r1 - 1 - i for i in range(len(dots))]
                mop = op_toward_from_above
            else:
                dots = sorted([r for r in range(r2 + 1, h) if A[r, c] == dotc])
                targets = [r2 + 1 + i for i in range(len(dots))]
                mop = op_toward_from_below
            for r, t in zip(dots, targets):           # nearest piece falls first
                n = abs(t - r)
                if n == 0:
                    continue
                ops.append(mop); sels.append(sel_of([real(r, c)]))   # grab the piece
                for _ in range(n - 1):
                    ops.append(mop); sels.append(sel_of([]))         # keep sliding
                G[r, c] = 0          # ARCLE leaves the start cell at 0
                G[t, c] = piece
                landed.add((t, c))
            if bgc is not None and bgc != 0:
                # once this line's stack has settled, the holes it left (cells
                # now reading 0 on this side of this line) go back to background
                span = range(0, r1) if side == 0 else range(r2 + 1, h)
                holes = [real(r, c) for r in span if G[r, c] == 0]
                if holes:
                    ops.append(int(bgc)); sels.append(sel_of(holes))
    if barc == 0:
        # the settled stacks take the band colour: flood each connected stack
        # (seeded on a piece that fell there, when there is one)
        seen = set()
        for r in range(h):
            for c in range(w):
                if G[r, c] != dotc or (r, c) in seen:
                    continue
                comp, todo = [], [(r, c)]
                seen.add((r, c))
                while todo:
                    a, b = todo.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen and G[na, nb] == dotc:
                            seen.add((na, nb)); todo.append((na, nb))
                fell = [p for p in comp if p in landed]
                sr, sc = real(*min(fell if fell else comp))
                ops.append(10 + int(barc)); sels.append([sr, sc, 0, 0])
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
