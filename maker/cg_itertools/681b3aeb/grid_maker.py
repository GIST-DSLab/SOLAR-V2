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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _dneigh(r, c):
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ca=1, cb=2) -> dict:
    hi_cap = max(2, min(8, max_h // 3, max_w // 3))
    while True:
        hi = _unifint(diff_lb, diff_ub, (2, hi_cap))
        h = _unifint(diff_lb, diff_ub, (min(3 * hi, max_h), max_h))
        w = _unifint(diff_lb, diff_ub, (min(3 * hi, max_w), max_w))
        inds = [(i, j) for i in range(hi) for j in range(hi)]
        while True:
            pa = random.choice(inds)
            rem = [x for x in inds if x != pa]
            pb = random.choice(rem)
            rem = [x for x in rem if x != pb]
            A, B = {pa}, {pb}
            remset = set(rem)
            for _ in range(len(rem)):
                ac = [x for x in remset if any(n in A for n in _dneigh(*x))]
                bc = [x for x in remset if any(n in B for n in _dneigh(*x))]
                opts = []
                if ac:
                    opts.append(0)
                if bc:
                    opts.append(1)
                if not opts:
                    break
                if random.choice(opts) == 0:
                    loc = random.choice(ac); A.add(loc)
                else:
                    loc = random.choice(bc); B.add(loc)
                remset.discard(loc)
            if remset:
                continue

            def isrect(S):
                rs = [r for r, _ in S]; cs = [c for _, c in S]
                return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if not isrect(A) and not isrect(B):
                break
        go = [[ca if (i, j) in A else cb for j in range(hi)] for i in range(hi)]

        def norm(S):
            r0 = min(r for r, _ in S); c0 = min(c for _, c in S)
            return {(r - r0, c - c0) for r, c in S}
        An, Bn = norm(A), norm(B)
        ha = max(r for r, _ in An) + 1; wa = max(c for _, c in An) + 1
        hb = max(r for r, _ in Bn) + 1; wb = max(c for _, c in Bn) + 1
        if ha > h or wa > w or hb > h or wb > w:
            continue
        ok = False
        for _ in range(10):
            li = random.randint(0, h - ha); lj = random.randint(0, w - wa)
            pA = {(r + li, c + lj) for r, c in An}
            bi = random.randint(0, h - hb); bj = random.randint(0, w - wb)
            pB = {(r + bi, c + bj) for r, c in Bn}
            if pA & pB:
                continue
            # keep the pieces from touching (so each stays its own component)
            near = {(r + dr, c + dc) for r, c in pA for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if near & pB:
                continue
            ok = True
            break
        if not ok:
            continue
        gi = [[bgc] * w for _ in range(h)]
        for r, c in pA:
            gi[r][c] = ca
        for r, c in pB:
            gi[r][c] = cb
        return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading: the solved rectangle is framed by ONE piece; every cell of that
    rectangle the frame piece does not cover is simply the other piece's colour.
    So: cut the canvas down to the frame piece's rectangle (moving the frame into
    its slot if the rectangle cannot be cut around it in place), then fill the
    frame's holes with the other piece's colour.  O is never inspected."""
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # ---- background: the colour every example input is painted with
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

    # ---- the two pieces (8-connected single-colour components)
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
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True; stack.append((ny, nx))
            comps.append((col, sorted(cells)))
    if len(comps) < 2:
        return [34], [[0, 0, h - 1, w - 1]]
    comps.sort(key=lambda t: -len(t[1]))
    comps = comps[:2]

    # ---- frame piece = the larger one (as the golf reading: the rarer colour
    # is the filler), unless its colour is 0, which ARCLE cannot carry/crop
    (colF, cellsF), (colH, cellsH) = comps
    if colF == 0:
        (colF, cellsF), (colH, cellsH) = (colH, cellsH), (colF, cellsF)

    ulF = (min(r for r, _ in cellsF), min(c for _, c in cellsF))
    ulH = (min(r for r, _ in cellsH), min(c for _, c in cellsH))
    nF = {(r - ulF[0], c - ulF[1]) for r, c in cellsF}
    nH = {(r - ulH[0], c - ulH[1]) for r, c in cellsH}
    hF = max(r for r, _ in nF) + 1
    wF = max(c for _, c in nF) + 1
    span = 2 + max(hF, wF, max(r for r, _ in nH) + 1, max(c for _, c in nH) + 1)

    # ---- the rectangle the frame belongs to: where the other piece's shape
    # fits exactly into the frame's gaps to make a solid rectangle
    sol = None
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            sh = {(r + dr, c + dc) for r, c in nH}
            if sh & nF:
                continue
            u = nF | sh
            r0 = min(r for r, _ in u); c0 = min(c for _, c in u)
            r1 = max(r for r, _ in u); c1 = max(c for _, c in u)
            if len(u) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            sol = (-r0, -c0, r1 - r0 + 1, c1 - c0 + 1)
            break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    pF = (sol[0], sol[1])          # frame's offset inside the rectangle
    hr, wr = sol[2], sol[3]

    ops, sels = [], []
    R0, C0 = ulF[0] - pF[0], ulF[1] - pF[1]
    if R0 >= 0 and C0 >= 0 and R0 + hr <= h and C0 + wr <= w:
        # the frame's rectangle lies on the grid: cut it out in place
        # (selection is exactly that full rectangle)
        ops.append(33); sels.append([R0, C0, hr - 1, wr - 1])
        canvas = I[R0:R0 + hr, C0:C0 + wr].copy()
    else:
        # cut out the frame's own bounding box (full rectangle) ...
        ops.append(33); sels.append([ulF[0], ulF[1], hF - 1, wF - 1])
        canvas = np.zeros((hr, wr), dtype=int)
        canvas[:hF, :wF] = I[ulF[0]:ulF[0] + hF, ulF[1]:ulF[1] + wF]
        if (hF, wF) != (hr, wr):
            # ... grow the canvas to the rectangle's size (full rectangle) ...
            ops.append(33); sels.append([0, 0, hr - 1, wr - 1])
        # ... and slide the frame into its slot
        cur = sorted(nF)
        steps = [(21, (1, 0))] * pF[0] + [(22, (0, 1))] * pF[1]
        first = True
        for op, (sr, sc) in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc) for r, c in cur]
        if steps:
            for r, c in nF:
                canvas[r, c] = 0
            for r, c in cur:
                canvas[r, c] = colF

    # ---- every rectangle cell the frame does not cover takes the filler colour
    frame_cells = {(r + pF[0], c + pF[1]) for r, c in nF}
    hole = [(r, c) for r in range(hr) for c in range(wr) if (r, c) not in frame_cells]
    if hole and any(canvas[r, c] != colH for r, c in hole):
        ops.append(colH); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, hr - 1, wr - 1])
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
