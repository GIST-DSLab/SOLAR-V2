"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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

# Comparing the code-golf reading with the maker's derive_operations: they agree.
# The golf entry looks for the pair of offsets that lays the two pieces into one
# window so that every cell belongs to exactly one piece (no overlap, no gap).
# That is the same quantity the maker holds fixed: the relative offset that
# interlocks piece A and piece B into a solid rectangle. Both treat each
# coloured piece as one unit that is translated whole, and the answer is just
# that window. So the operations stay the same: grab each piece once, slide it
# into its slot with unit Moves, then crop to the assembled rectangle.


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on the two pieces' shapes, not their colours.
    # The background colour is fixed for the whole episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if a >= b:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    if bgc is None:
        bgc = random.choice(range(10))
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    others = [c for c in range(10) if c != bgc]

    def dn(p):
        r, c = p
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    def is_rect(S):
        rs = [r for r, _ in S]
        cs = [c for _, c in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    def norm(S):
        r0 = min(r for r, _ in S)
        c0 = min(c for _, c in S)
        return {(r - r0, c - c0) for r, c in S}

    for _ in range(1000):
        hcap = max(2, min(8, max_h // 3))
        hi = unifint(diff_lb, diff_ub, (2, hcap))
        h = unifint(diff_lb, diff_ub, (min(3 * hi, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(3 * hi, max_w), max_w))
        if hi > h or hi > w:
            continue
        ca, cb = random.sample(others, 2)
        inds = [(r, c) for r in range(hi) for c in range(hi)]
        ok = False
        for _t in range(200):
            pa = random.choice(inds)
            rem = [p for p in inds if p != pa]
            pb = random.choice(rem)
            rem = set(rem) - {pb}
            A, B = {pa}, {pb}
            while rem:
                ac = {q for p in A for q in dn(p)} & rem
                bc = {q for p in B for q in dn(p)} & rem
                opts = []
                if ac:
                    opts.append(0)
                if bc:
                    opts.append(1)
                if not opts:
                    break
                if random.choice(opts) == 0:
                    loc = random.choice(sorted(ac)); A.add(loc)
                else:
                    loc = random.choice(sorted(bc)); B.add(loc)
                rem.discard(loc)
            if rem:
                continue
            if not is_rect(A) and not is_rect(B):
                ok = True
                break
        if not ok:
            continue
        go = [[ca if (r, c) in A else cb for c in range(hi)] for r in range(hi)]
        An, Bn = norm(A), norm(B)
        ha = max(r for r, _ in An) + 1; wa = max(c for _, c in An) + 1
        hb = max(r for r, _ in Bn) + 1; wb = max(c for _, c in Bn) + 1
        placed = None
        for _k in range(50):
            ai = random.randint(0, h - ha); aj = random.randint(0, w - wa)
            pA = {(r + ai, c + aj) for r, c in An}
            halo = {(r + dr, c + dc) for r, c in pA for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            bi = random.randint(0, h - hb); bj = random.randint(0, w - wb)
            pB = {(r + bi, c + bj) for r, c in Bn}
            if pB & halo:
                continue
            placed = (pA, pB)
            break
        if placed is None:
            continue
        gi = [[bgc] * w for _ in range(h)]
        for r, c in placed[0]:
            gi[r][c] = ca
        for r, c in placed[1]:
            gi[r][c] = cb
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    """Two puzzle pieces lie on a background and interlock into a solid
    rectangle. Everything is measured from I, plus the background colour, which
    the episode's example inputs confirm. O is never inspected."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background colour: the one every input grid in this episode is painted with
    grids = [I]
    if examples:
        for pair in examples:
            try:
                grids.append(np.asarray(pair[0], dtype=int))
            except Exception:
                pass
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]
    if not bool((I == bgc).any()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the two pieces: same-colour connected components on that background
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append((col, sorted(cells)))

    if len(comps) < 2:
        return [34], [[0, 0, h - 1, w - 1]]

    comps.sort(key=lambda t: -len(t[1]))
    comps = comps[:2]
    comps.sort(key=lambda t: t[1][0])
    (colA, cellsA), (colB, cellsB) = comps

    ulA = (min(r for r, _ in cellsA), min(c for _, c in cellsA))
    ulB = (min(r for r, _ in cellsB), min(c for _, c in cellsB))
    nA = {(r - ulA[0], c - ulA[1]) for r, c in cellsA}
    nB = {(r - ulB[0], c - ulB[1]) for r, c in cellsB}
    span = 2 + max(max(r for r, _ in nA), max(r for r, _ in nB),
                   max(c for _, c in nA), max(c for _, c in nB))

    # interlocking offset: the union is a solid rectangle with no overlap
    sol = None
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            sb = {(r + dr, c + dc) for r, c in nB}
            if sb & nA:
                continue
            u = nA | sb
            r0 = min(r for r, _ in u); c0 = min(c for _, c in u)
            r1 = max(r for r, _ in u); c1 = max(c for _, c in u)
            if len(u) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            sol = (dr - r0, dc - c0, r1 - r0 + 1, c1 - c0 + 1, -r0, -c0)
            break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]

    pB = (sol[0], sol[1])
    hr, wr = sol[2], sol[3]
    pA = (sol[4], sol[5])
    srcA, srcB = set(cellsA), set(cellsB)

    # where to build the rectangle: prefer leaving one piece in place, avoid
    # moving a colour-0 piece, and pick an order where no piece overwrites the other
    best = None
    best_any = None
    for R0 in range(0, h - hr + 1):
        for C0 in range(0, w - wr + 1):
            dA = (R0 + pA[0] - ulA[0], C0 + pA[1] - ulA[1])
            dB = (R0 + pB[0] - ulB[0], C0 + pB[1] - ulB[1])
            movers = []
            if dA != (0, 0):
                movers.append('A')
            if dB != (0, 0):
                movers.append('B')
            cost = (1000 * sum(1 for m in movers if (colA if m == 'A' else colB) == 0)
                    + 100 * len(movers)
                    + abs(dA[0]) + abs(dA[1]) + abs(dB[0]) + abs(dB[1]))
            if len(movers) < 2:
                order = movers
            else:
                dstA = {(r + R0 + pA[0], c + C0 + pA[1]) for r, c in nA}
                dstB = {(r + R0 + pB[0], c + C0 + pB[1]) for r, c in nB}
                if not (dstA & srcB):
                    order = ['A', 'B']
                elif not (dstB & srcA):
                    order = ['B', 'A']
                else:
                    order = None
            if best_any is None or cost < best_any[0]:
                best_any = (cost, R0, C0, movers, dA, dB)
            if order is not None and (best is None or cost < best[0]):
                best = (cost, R0, C0, order, dA, dB)
    if best is None:
        best = best_any
    _, R0, C0, order, dA, dB = best

    # carry each piece to its slot: one grab, then empty-selection Moves
    ops, sels = [], []
    info = {'A': (colA, cellsA, nA, dA, pA), 'B': (colB, cellsB, nB, dB, pB)}
    for m in order:
        col, cells, n, d, p = info[m]
        if col != 0:
            steps = []
            steps += [(20 if d[0] < 0 else 21, (-1 if d[0] < 0 else 1, 0))] * abs(d[0])
            steps += [(23 if d[1] < 0 else 22, (0, -1 if d[1] < 0 else 1))] * abs(d[1])
            cur = sorted(cells)
            first = True
            for op, (sr, sc) in steps:
                ops.append(op)
                sels.append(sel_of(cur) if first else sel_of([]))
                first = False
                cur = [(r + sr, c + sc) for r, c in cur]
            # the vacated footprint is either covered by the other piece or lies
            # outside the rectangle that gets cropped, so no repair is needed
        else:
            # ARCLE's object buffer drops colour-0 cells, so a colour-0 piece
            # cannot be carried by Move; paint it in its slot instead
            dst = [(r + R0 + p[0], c + C0 + p[1]) for r, c in sorted(n)]
            ops.append(0)
            sels.append(sel_of(dst))

    # keep only the assembled rectangle (the selection is exactly that full rect)
    ops.append(33)
    sels.append([R0, C0, hr - 1, wr - 1])
    ops.append(34)
    sels.append([0, 0, hr - 1, wr - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
