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
def sample_colors(num_examples=None) -> dict:
    # the generator samples three distinct colours: background, piece A, piece B
    cols = list(range(10))
    bgc, ca, cb = random.sample(cols, 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ca, cb) -> dict:
    while True:
        hi_ub = max(2, min(8, max_h // 3, max_w // 3))
        hi = unifint(diff_lb, diff_ub, (2, hi_ub))
        wi = unifint(diff_lb, diff_ub, (2, hi_ub))
        h = unifint(diff_lb, diff_ub, (3 * hi, max(3 * hi, max_h)))
        w = unifint(diff_lb, diff_ub, (3 * wi, max(3 * wi, max_w)))
        if h > max_h or w > max_w or h < hi or w < hi:
            continue

        c = canvas(-1, (hi, hi))

        # grow two interlocking pieces that together tile the hi x hi square,
        # at least one of them not a plain rectangle
        conda, condb = True, True
        A, B = None, None
        for _ in range(60):
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for _k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                if len(opts) == 0:
                    break
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            if len(A) + len(B) != hi * hi:
                conda, condb = True, True
                continue
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
            if not (conda and condb):
                break
        if A is None or B is None or (conda and condb):
            continue

        # the assembly must be UNIQUE, otherwise the pair is not solvable
        ar = min(r for r, _ in A)
        ac = min(cc for _, cc in A)
        br = min(r for r, _ in B)
        bc = min(cc for _, cc in B)
        nA = {(r - ar, cc - ac) for r, cc in A}
        nB = {(r - br, cc - bc) for r, cc in B}
        sols = set()
        for dr in range(-hi - 1, hi + 2):
            for dc in range(-hi - 1, hi + 2):
                sb = {(r + dr, cc + dc) for r, cc in nB}
                if sb & nA:
                    continue
                u = nA | sb
                r0 = min(r for r, _ in u)
                c0 = min(cc for _, cc in u)
                r1 = max(r for r, _ in u)
                c1 = max(cc for _, cc in u)
                if len(u) != (r1 - r0 + 1) * (c1 - c0 + 1):
                    continue
                sols.add(tuple(sorted(
                    [(r - r0, cc - c0, ca) for r, cc in nA] +
                    [(r - r0, cc - c0, cb) for r, cc in sb])))
        if len(sols) != 1:
            continue

        # place piece A so that the square it belongs to fits inside the canvas,
        # place piece B anywhere free
        hb_, wb_ = height(B), width(B)
        R0 = randint(0, h - hi)
        C0 = randint(0, w - hi)
        plcda = frozenset((r + R0, cc + C0) for r, cc in A)
        ok = False
        plcdb = None
        for _ in range(200):
            rb = randint(0, h - hb_)
            cbb = randint(0, w - wb_)
            plcdb = frozenset((r - br + rb, cc - bc + cbb) for r, cc in B)
            if plcdb & plcda:
                continue
            ok = True
            break
        if not ok:
            continue

        gi = canvas(bgc, (h, w))
        gi = fill(gi, ca, plcda)
        gi = fill(gi, cb, plcdb)
        go = fill(c, ca, frozenset(A))
        go = fill(go, cb, frozenset(B))
        return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    """Two puzzle pieces lie on a background.  Like blitting both sprites into
    a fresh canvas and cropping: carry BOTH pieces to the top-left corner of the
    grid, interlocked into a solid rectangle, then crop the canvas to it.
    Everything is measured from I (bgc confirmed by the example inputs)."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

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

    # the two pieces: single-colour 8-connected components
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
    pieces = []
    for col, cells in comps:
        ul = (min(r for r, _ in cells), min(c for _, c in cells))
        n = {(r - ul[0], c - ul[1]) for r, c in cells}
        pieces.append({'col': col, 'cells': cells, 'ul': ul, 'n': n})
    nA, nB = pieces[0]['n'], pieces[1]['n']
    span = 2 + max(max(max(r, c) for r, c in nA), max(max(r, c) for r, c in nB))

    # the interlocking: offset of B relative to A making a gap-free rectangle
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
            sol = ((-r0, -c0), (dr - r0, dc - c0), r1 - r0 + 1, c1 - c0 + 1)
            break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    pieces[0]['p'], pieces[1]['p'], hr, wr = sol

    # the fresh canvas is the grid's top-left corner; if the pieces there would
    # land on each other's source, shift the canvas origin minimally
    anchors = sorted(((R0, C0) for R0 in range(h - hr + 1) for C0 in range(w - wr + 1)),
                     key=lambda t: (t[0] + t[1], t[0]))
    plan = None
    for R0, C0 in anchors:
        for pc in pieces:
            pc['dst'] = {(r + R0 + pc['p'][0], c + C0 + pc['p'][1]) for r, c in pc['n']}
        for order in ((0, 1), (1, 0)):
            first, second = pieces[order[0]], pieces[order[1]]
            # a piece carried later must not have its source covered by an
            # earlier placement (a colour-0 piece is painted, never grabbed)
            if second['col'] != 0 and (first['dst'] & set(second['cells'])):
                continue
            plan = (R0, C0, order)
            break
        if plan is not None:
            break
    if plan is None:
        return [34], [[0, 0, h - 1, w - 1]]
    R0, C0, order = plan
    for pc in pieces:
        pc['dst'] = {(r + R0 + pc['p'][0], c + C0 + pc['p'][1]) for r, c in pc['n']}

    ops, sels = [], []
    for k in order:
        pc = pieces[k]
        d = (R0 + pc['p'][0] - pc['ul'][0], C0 + pc['p'][1] - pc['ul'][1])
        if d == (0, 0):
            continue
        if pc['col'] != 0:
            steps = [(20 if d[0] < 0 else 21)] * abs(d[0]) + [(23 if d[1] < 0 else 22)] * abs(d[1])
            for i, op in enumerate(steps):
                ops.append(op)
                # first step grabs the piece; later steps keep the same grab
                sels.append(sel_of(pc['cells']) if i == 0 else sel_of([]))
            # vacated footprint (left at 0) lies outside the canvas or is
            # covered by the other piece, so the crop below makes it moot
        else:
            # ARCLE cannot carry colour-0 cells: paint this piece at its slot
            ops.append(0)
            sels.append(sel_of(sorted(pc['dst'])))

    # crop the canvas to the assembled rectangle (full rect selection)
    ops.append(33); sels.append([R0, C0, hr - 1, wr - 1])
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
