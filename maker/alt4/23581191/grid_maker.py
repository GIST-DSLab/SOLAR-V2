"""
ARC Task: 23581191 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter, deque
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colours ----
VARIANTS = [{"density": "sparse"}, {"density": "dense"}]


def sample_colors(num_examples=None) -> dict:
    # generator's colopts = interval(0,10,1) without 2 (2 is the crossing colour)
    colopts = [c for c in range(10) if c != 2]
    bgcol = random.choice(colopts)
    rem = [c for c in colopts if c != bgcol]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgcol": bgcol, "acol": acol, "bcol": bcol, "instance_plan": plan}


# --------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgcol=None, acol=None, bcol=None, density=None) -> dict:
    if density is None:
        density = random.choice(VARIANTS)["density"]
    hub = max(3, min(30, int(max_h)))
    wub = max(3, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, wub))
    c = canvas(bgcol, (h, w))
    inds = totuple(asindices(c))

    # sparse instances keep A-only / B-only stripes visible (learnability);
    # dense ones reproduce the original card_bounds
    if density == "sparse":
        cub = max(1, min(h, w) // 2)
    else:
        cub = max(1, (h * w) // 4)
    na = unifint(diff_lb, diff_ub, (1, cub))
    nb = unifint(diff_lb, diff_ub, (1, cub))
    na = max(1, min(na, len(inds) - 1))
    a = tuple(sample(inds, na))
    rest = tuple(difference(inds, a))
    nb = max(1, min(nb, len(rest)))
    b = tuple(sample(rest, nb))

    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)

    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------ trajectory -----
def derive_operations(I, O, examples=None):
    """
    Rule: every dot of colour A lights up its whole row and its whole column;
    same for colour B.  Background cells lit by A only become A, by B only
    become B, by both become the crossing colour (2).  Dots are untouched.

    Account: background REGIONS that are lit uniformly are flooded (FloodFill),
    the rest is drawn LINE BY LINE - one operation per row/column the rule is
    actually about, plus the crossings of an A line with a B line.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])                 # canvas colour
    others = sorted([int(c) for c in cnt if int(c) != bgc], key=lambda c: -cnt[c])
    A = others[0] if others else bgc
    B = others[1] if len(others) > 1 else A

    # crossing colour is a task convention -> read it off the demonstrations
    cross = 2
    if examples:
        for pair in examples:
            try:
                ei = np.asarray(pair[0], dtype=int)
                eo = np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            new = set(eo.flatten().tolist()) - set(ei.flatten().tolist())
            if len(new) == 1:
                cross = int(new.pop())
                break

    rowsA, colsA, rowsB, colsB = set(), set(), set(), set()
    for r in range(h):
        for c in range(w):
            v = int(I[r, c])
            if v == A:
                rowsA.add(r); colsA.add(c)
            elif v == B:
                rowsB.add(r); colsB.add(c)

    target = {}
    for r in range(h):
        for c in range(w):
            if int(I[r, c]) != bgc:
                continue
            inA = (r in rowsA) or (c in colsA)
            inB = (r in rowsB) or (c in colsB)
            if inA and inB:
                t = cross
            elif inA:
                t = A
            elif inB:
                t = B
            else:
                t = bgc
            target[(r, c)] = t

    cur = I.copy()
    ops, sels = [], []
    pending = set(p for p, t in target.items() if t != bgc)

    def emit(cells, color):
        sel = [p for p in cells if p in pending and int(cur[p]) != color]
        if not sel:
            return
        ops.append(int(color))
        sels.append(sel_of(sel))
        for p in sel:
            cur[p] = color
            pending.discard(p)

    # --- 1. whole background regions that are lit uniformly: one FloodFill ---
    seen = set()
    for r0 in range(h):
        for c0 in range(w):
            if (r0, c0) in seen or int(I[r0, c0]) != bgc:
                continue
            comp = []
            dq = deque([(r0, c0)])
            seen.add((r0, c0))
            while dq:
                r, c = dq.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                            and int(I[nr, nc]) == bgc:
                        seen.add((nr, nc))
                        dq.append((nr, nc))
            tset = set(target[p] for p in comp)
            if len(comp) >= 3 and len(tset) == 1:
                t = tset.pop()
                if t != bgc:
                    ops.append(10 + int(t))
                    sels.append(sel_of([comp[0]]))       # FloodFill: 1 seed cell
                    for p in comp:
                        cur[p] = t
                for p in comp:
                    pending.discard(p)

    # --- 2. the A lines, then the B lines (each line = one operation) --------
    for colr, rws, cls in ((A, rowsA, colsA), (B, rowsB, colsB)):
        for r in sorted(rws):
            emit([(r, c) for c in range(w) if target.get((r, c)) == colr], colr)
        for c in sorted(cls):
            emit([(r, c) for r in range(h) if target.get((r, c)) == colr], colr)

    # --- 3. where an A line meets a B line -> crossing colour ----------------
    for r in sorted(rowsA & rowsB):                      # rows lit by both
        emit([(r, c) for c in range(w) if target.get((r, c)) == cross], cross)
    for c in sorted(colsA & colsB):                      # columns lit by both
        emit([(r, c) for r in range(h) if target.get((r, c)) == cross], cross)
    for r in sorted(rowsA - rowsB):                      # A row x B columns
        emit([(r, c) for c in sorted(colsB) if target.get((r, c)) == cross], cross)
    for r in sorted(rowsB - rowsA):                      # B row x A columns
        emit([(r, c) for c in sorted(colsA) if target.get((r, c)) == cross], cross)

    # --- safety net (should stay empty) --------------------------------------
    if pending:
        leftovers = sorted(pending)
        for colr in sorted(set(target[p] for p in leftovers)):
            emit([p for p in leftovers if target[p] == colr], colr)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == whole grid, submitted as-is
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
                        f"num_examples+1 ({num_examples + 1}) for task 23581191"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23581191"
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
                                f"for task 23581191"
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
                    f"Failed to build a complete episode for task 23581191 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23581191-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
