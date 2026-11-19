"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is a fixed episode-wide role; the figure colours are
    # free (the rule depends on the figure's shape about its centre, not colour).
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = int(round(a + (b - a) * lb)); hi = int(round(a + (b - a) * ub))
        return random.randint(min(lo, hi), max(lo, hi))
    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = random.randint(0, h - d); locj = random.randint(0, w - d)
    remcols = [c for c in range(10) if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numcols)
    inds = {(i, j) for i in range(d) for j in range(d)}
    if d % 2 == 0:
        q = {ij for ij in inds if ij[0] < d // 2 and ij[1] < d // 2}
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = {ij for ij in inds if ij[0] < d // 2 and ij[1] <= d // 2}
        m = d // 2
        cp = {(m, m), (m-1, m-1), (m-1, m+1), (m+1, m-1), (m+1, m+1)}
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        for i in range(k, d - k):
            for j in range(k, d - k):
                if i in (k, d-k-1) or j in (k, d-k-1):
                    rings.add((i, j))
    qin = sorted(q - rings); qout = sorted(rings & q)

    def mk():
        n = unifint(diff_lb, diff_ub, (1, len(q)))
        cells = set(random.sample(sorted(q), n)) | {random.choice(qin)} | {random.choice(qout)}
        return {(random.choice(ccols), ij) for ij in cells}

    def hm(o): return {(c, (d-1-i, j)) for c, (i, j) in o}
    def vm(o): return {(c, (i, d-1-j)) for c, (i, j) in o}
    tailobj = mk()
    while hm(tailobj) == tailobj and vm(tailobj) == tailobj:
        tailobj = mk()

    def paint(g, o):
        g = g.copy()
        for c, (i, j) in o:
            g[i, j] = c
        return g
    def rot90(g): return np.rot90(g, 3)
    subg = np.full((d, d), bgc, dtype=int)
    for k in range(4):
        subg = paint(subg, tailobj); subg = rot90(subg)
    fc = random.choice(ccols)
    fxobj = {(fc, ij) for ij in cp}
    subg = paint(subg, fxobj)
    subgo = subg.copy()
    subgi = subg.copy()
    for ij in rings:
        subgi[ij] = bgc
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for _ in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    k = random.choice([0, 1, 2, 3])
    subgi = np.rot90(subgi, k); subgo = np.rot90(subgo, k)
    gi = np.full((h, w), bgc, dtype=int); go = gi.copy()
    gi[loci:loci+d, locj:locj+d] = subgi
    go[loci:loci+d, locj:locj+d] = subgo
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """Pinwheel completion, following the code-golf reading.

    Where it agrees with the maker: both find the centre from the figure's
    solid centre mark and draw quarter-turn copies one at a time, using only I.
    Where it differs: the golf entry works on one square WINDOW centred on that
    mark. It turns the window AS IT STANDS NOW, which already holds the copies
    drawn so far, and it fills only the window's BLANK (background) cells. It
    never paints over a cell that already has a colour. Each step is grouped by
    colour. The selection is the set of blank cells the turned window lands on,
    and it is built from I and the grid so far. O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the figure never spans the canvas, so a uniform row/col exists
    bgc = None
    for r in range(hi):
        if np.all(I[r] == I[r, 0]):
            bgc = int(I[r, 0]); break
    if bgc is None:
        for c in range(wi):
            if np.all(I[:, c] == I[0, c]):
                bgc = int(I[0, c]); break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    fg = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if not fg:
        return [34], [[0, 0, hi - 1, wi - 1]]
    r0 = min(p[0] for p in fg); r1 = max(p[0] for p in fg)
    c0 = min(p[1] for p in fg); c1 = max(p[1] for p in fg)

    def rot1(r, c, cr2, cc2):
        # one quarter turn about pivot (cr2/2, cc2/2) in doubled coordinates
        s = (cr2 + cc2) // 2; t = (cc2 - cr2) // 2
        return (s - c, t + r)

    # pivot = centre of the solid centre mark (2x2 for even side, cell + its
    # diagonals for odd side); the window around it must hold all of I's figure
    # and every turn must land figure cells on background or on the same colour
    best = None
    for cr2 in range(2 * (hi - 1) + 1):
        for cc2 in range(2 * (wi - 1) + 1):
            if (cr2 - cc2) % 2:
                continue
            ext = max(cr2 - 2 * r0, 2 * r1 - cr2, cc2 - 2 * c0, 2 * c1 - cc2)
            if cr2 - ext < 0 or cr2 + ext > 2 * (hi - 1) or cc2 - ext < 0 or cc2 + ext > 2 * (wi - 1):
                continue
            a, b = cr2 // 2, cc2 // 2
            if cr2 % 2:
                mark = [(a, b), (a, b + 1), (a + 1, b), (a + 1, b + 1)]
            else:
                mark = [(a, b), (a-1, b-1), (a-1, b+1), (a+1, b-1), (a+1, b+1)]
            if any(not (0 <= p[0] < hi and 0 <= p[1] < wi) for p in mark):
                continue
            mv = {int(I[p]) for p in mark}
            if len(mv) != 1 or next(iter(mv)) == bgc:
                continue
            ok = True; agree = 0
            for (r, c, v) in fg:
                p = (r, c)
                for _ in range(3):
                    p = rot1(p[0], p[1], cr2, cc2)
                    u = int(I[p])
                    if u == v:
                        agree += 1
                    elif u != bgc:
                        ok = False; break
                if not ok:
                    break
            if not ok:
                continue
            side_hi = 7 if cr2 % 2 == 0 else 6
            core = 0
            for side in range(side_hi, 0, -1):
                if side % 2 != (0 if cr2 % 2 else 1):
                    continue
                tr = (cr2 - (side - 1)) // 2; tc = (cc2 - (side - 1)) // 2
                if tr < 0 or tc < 0 or tr + side > hi or tc + side > wi:
                    continue
                P = I[tr:tr + side, tc:tc + side]
                if np.array_equal(P, np.rot90(P)) and int((P != bgc).sum()) >= 2:
                    core = side; break
            key = (-core, abs(cr2 - (r0 + r1)) + abs(cc2 - (c0 + c1)), -agree, ext)
            if best is None or key < best[0]:
                best = (key, (cr2, cc2, ext))
    if best is None:
        return [34], [[0, 0, hi - 1, wi - 1]]
    cr2, cc2, ext = best[1]
    wr0 = (cr2 - ext) // 2; wc0 = (cc2 - ext) // 2; side = ext + 1
    window = [(r, c) for r in range(wr0, wr0 + side) for c in range(wc0, wc0 + side)]

    cur = I.copy()
    ops, sels = [], []
    for _ in range(3):
        # turn the window as it currently stands; drop its figure only into
        # the window's still-blank cells, one colour group at a time
        turned = {}
        for (r, c) in window:
            v = int(cur[r, c])
            if v != bgc:
                turned[rot1(r, c, cr2, cc2)] = v
        groups = {}
        for p, v in turned.items():
            if int(cur[p]) == bgc:
                groups.setdefault(v, []).append(p)
        for v in sorted(groups):
            cells = sorted(groups[v])
            ops.append(int(v)); sels.append(sel_of(cells))
            for p in cells:
                cur[p] = v
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-canvas bbox: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
