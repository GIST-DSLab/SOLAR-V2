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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    bgc, ca, cb = random.sample(range(10), 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None) -> dict:
    def unifint(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    if bgc is None:
        bgc = random.randrange(10)
    if ca is None:
        ca = random.choice([c for c in range(10) if c != bgc])
    if cb is None:
        cb = random.choice([c for c in range(10) if c not in (bgc, ca)])

    def is_rect(S):
        rs = [r for r, _ in S]; cs = [c for _, c in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    def dn(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    while True:
        hi_cap = max(2, min(8, max_h // 3))
        wi_cap = max(2, min(8, max_w // 3))
        hi = unifint(2, hi_cap)
        wi = unifint(2, wi_cap)
        h = unifint(min(3 * hi, max_h), max_h)
        w = unifint(min(3 * wi, max_w), max_w)
        if hi > h or hi > w:
            continue
        inds = [(i, j) for i in range(hi) for j in range(hi)]
        while True:
            pa = random.choice(inds)
            rem = [p for p in inds if p != pa]
            pb = random.choice(rem)
            rem = [p for p in rem if p != pb]
            A, B = {pa}, {pb}
            remset = set(rem)
            while remset:
                ac = {q for p in A for q in dn(p)} & remset
                bc = {q for p in B for q in dn(p)} & remset
                opts = []
                if ac:
                    opts.append(0)
                if bc:
                    opts.append(1)
                if random.choice(opts) == 0:
                    loc = random.choice(sorted(ac)); A.add(loc)
                else:
                    loc = random.choice(sorted(bc)); B.add(loc)
                remset.discard(loc)
            if not (is_rect(A) and is_rect(B)):
                break
        go = [[ca if (i, j) in A else cb for j in range(hi)] for i in range(hi)]

        def norm(S):
            r0 = min(r for r, _ in S); c0 = min(c for _, c in S)
            return {(r - r0, c - c0) for r, c in S}
        An, Bn = norm(A), norm(B)
        ha = max(r for r, _ in An) + 1; wa = max(c for _, c in An) + 1
        hb = max(r for r, _ in Bn) + 1; wb = max(c for _, c in Bn) + 1
        ok = False
        for _ in range(10):
            ai = random.randint(0, h - ha)
            aj = random.randint(0, w - wa)
            pA = {(r + ai, c + aj) for r, c in An}
            cands = [(i, j) for i in range(h - hb + 1) for j in range(w - wb + 1)
                     if (i, j) not in pA]
            if not cands:
                continue
            bi, bj = random.choice(cands)
            pB = {(r + bi, c + bj) for r, c in Bn}
            if not (pA & pB):
                ok = True
                break
        if ok:
            break
    gi = [[bgc] * w for _ in range(h)]
    for r, c in pA:
        gi[r][c] = ca
    for r, c in pB:
        gi[r][c] = cb
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Golf reading: the piece with MORE cells is the frame; the square it
    belongs to has side sqrt(total piece cells); the other piece fills the
    frame's holes. Frame stays put whenever its square fits on the grid; the
    other piece is carried in by Move (painted if it is colour 0, which ARCLE
    cannot carry). Then crop to the square. O is never inspected."""
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the colour every input of this episode is painted with
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

    # the two pieces (4-connected, single colour)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            comps.append((col, sorted(cells)))
    if len(comps) < 2:
        return [34], [[0, 0, h - 1, w - 1]]
    comps.sort(key=lambda t: (-len(t[1]), t[1][0]))
    (colF, cellsF), (colP, cellsP) = comps[0], comps[1]   # F = frame (larger), P = filler

    ulF = (min(r for r, _ in cellsF), min(c for _, c in cellsF))
    ulP = (min(r for r, _ in cellsP), min(c for _, c in cellsP))
    nF = {(r - ulF[0], c - ulF[1]) for r, c in cellsF}
    nP = {(r - ulP[0], c - ulP[1]) for r, c in cellsP}
    hF = max(r for r, _ in nF) + 1; wF = max(c for _, c in nF) + 1

    # side of the assembled square = sqrt(total piece cells)
    tot = len(nF) + len(nP)
    side = int(round(tot ** 0.5))
    if side * side != tot:
        return [34], [[0, 0, h - 1, w - 1]]

    # where the frame sits inside the square so that its holes are exactly the filler's shape
    sol = None
    for fr in range(0, side - hF + 1):
        for fc in range(0, side - wF + 1):
            holes = {(r, c) for r in range(side) for c in range(side)} - \
                    {(r + fr, c + fc) for r, c in nF}
            hr0 = min(r for r, _ in holes); hc0 = min(c for _, c in holes)
            if {(r - hr0, c - hc0) for r, c in holes} == nP:
                sol = ((fr, fc), (hr0, hc0))
                break
        if sol:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    offF, offP = sol

    # where to build the square: frame stationary if it fits, fewest movers,
    # never Move a colour-0 piece, and an order where no piece lands on the other's source
    best = None
    for R0 in range(0, h - side + 1):
        for C0 in range(0, w - side + 1):
            dF = (R0 + offF[0] - ulF[0], C0 + offF[1] - ulF[1])
            dP = (R0 + offP[0] - ulP[0], C0 + offP[1] - ulP[1])
            info = {'F': (colF, cellsF, nF, dF, offF), 'P': (colP, cellsP, nP, dP, offP)}
            movers = [k for k in ('F', 'P') if info[k][3] != (0, 0)]
            dst = {k: {(r + R0 + info[k][4][0], c + C0 + info[k][4][1]) for r, c in info[k][2]}
                   for k in ('F', 'P')}
            src = {'F': set(cellsF), 'P': set(cellsP)}
            if len(movers) < 2:
                order = movers
            elif not (dst['F'] & src['P']):
                order = ['F', 'P']
            elif not (dst['P'] & src['F']):
                order = ['P', 'F']
            else:
                continue
            cost = (10000 * sum(1 for m in movers if info[m][0] == 0)
                    + 1000 * len(movers) + 500 * ('F' in movers)
                    + abs(dF[0]) + abs(dF[1]) + abs(dP[0]) + abs(dP[1]))
            if best is None or cost < best[0]:
                best = (cost, R0, C0, order, info)
    if best is None:
        return [34], [[0, 0, h - 1, w - 1]]
    _, R0, C0, order, info = best

    ops, sels = [], []
    for m in order:
        col, cells, n, d, off = info[m]
        if col != 0:
            steps = [(20 if d[0] < 0 else 21)] * abs(d[0]) + [(23 if d[1] < 0 else 22)] * abs(d[1])
            for i, op in enumerate(steps):
                ops.append(op)
                # first step grabs the piece; later empty selections keep carrying it
                sels.append(sel_of(sorted(cells)) if i == 0 else sel_of([]))
            # the vacated footprint lies outside the square we crop to
        else:
            # colour 0 cannot be carried by Move: paint the frame's holes with it
            ops.append(0)
            sels.append(sel_of(sorted((r + R0 + off[0], c + C0 + off[1]) for r, c in n)))

    # crop to the assembled square (selection is exactly that full rectangle)
    ops.append(33); sels.append([R0, C0, side - 1, side - 1])
    ops.append(34); sels.append([0, 0, side - 1, side - 1])
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
