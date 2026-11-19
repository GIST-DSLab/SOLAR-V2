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
    # Only the background colour is a role here. The rule (complete the pinwheel
    # about its solid centre mark) does not depend on which figure colours appear.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = int(round(random.uniform(lo, hi)))
    return max(a, min(b, v))


def _mirror_obj(obj, axis):
    rows = [ij[0] for _, ij in obj]
    cols = [ij[1] for _, ij in obj]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    if axis == "h":
        return {(v, (r0 + r1 - i, j)) for v, (i, j) in obj}
    return {(v, (i, c0 + c1 - j)) for v, (i, j) in obj}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    d = _unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = random.randint(0, h - d)
    locj = random.randint(0, w - d)
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numcols)

    inds = [(i, j) for i in range(d) for j in range(d)]
    if d % 2 == 0:
        q = [(i, j) for (i, j) in inds if i < d // 2 and j < d // 2]
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = [(i, j) for (i, j) in inds if i < d // 2 and j <= d // 2]
        m = d // 2
        cp = {(m, m), (m-1, m-1), (m-1, m+1), (m+1, m-1), (m+1, m+1)}
    nrings = _unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        a, b = k, d - k - 1
        for t in range(a, b + 1):
            rings |= {(a, t), (b, t), (t, a), (t, b)}
    qset = set(q)
    qin = sorted(qset - rings)
    qout = sorted(rings & qset)

    def make_tail():
        n = _unifint(diff_lb, diff_ub, (1, len(q)))
        cells = set(random.sample(q, n))
        if qin:
            cells.add(random.choice(qin))
        if qout:
            cells.add(random.choice(qout))
        return {(random.choice(ccols), ij) for ij in cells}

    tailobj = make_tail()
    tries = 0
    while _mirror_obj(tailobj, "h") == tailobj and _mirror_obj(tailobj, "v") == tailobj:
        tailobj = make_tail()
        tries += 1
        if tries > 200:
            raise ValueError("symmetric tail")

    def paint(g, obj):
        for v, (i, j) in obj:
            if 0 <= i < g.shape[0] and 0 <= j < g.shape[1]:
                g[i, j] = v
        return g

    def rot90(g):
        return np.rot90(g, -1).copy()  # clockwise like DSL rot90

    subg = np.full((d, d), bgc, dtype=int)
    for _ in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fc = random.choice(ccols)
    fxobj = {(fc, ij) for ij in cp}
    subg = paint(subg, fxobj)
    subgo = subg.copy()
    subgi = subg.copy()
    for (i, j) in rings:
        subgi[i, j] = bgc
    nsplits = _unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for _ in range(nsplits)]
    for idx, cel in enumerate(list(tailobj)):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    k = random.choice([0, 1, 2, 3])
    for _ in range(k):
        subgi = rot90(subgi)
        subgo = rot90(subgo)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + d, locj:locj + d] = subgi
    go[loci:loci + d, locj:locj + d] = subgo
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Finish the pinwheel by drawing the three quarter-turn copies of the figure.

    The code-golf entry reads the task the same way the maker does: the output
    is the union of the figure turned 0/90/180/270 degrees about the centre of
    its solid centre mark. So there is nothing to change. The pivot comes from I
    alone (it carries the solid mark, keeps the finished figure on the canvas,
    and maps every foreground cell onto background or onto its own colour).
    Each quarter-turn copy is then drawn one at a time, one colour group at a
    time. O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

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

    def rot_images(r, c, cr2, cc2):
        s = (cr2 + cc2) // 2
        t = (cc2 - cr2) // 2
        return [(s - c, t + r), (cr2 - r, cc2 - c), (c - t, s - r)]

    best = None
    for cr2 in range(2 * (hi - 1) + 1):
        for cc2 in range(2 * (wi - 1) + 1):
            if (cr2 - cc2) % 2:
                continue
            ext = max(cr2 - 2 * r0, 2 * r1 - cr2, cc2 - 2 * c0, 2 * c1 - cc2)
            if ext < 0:
                continue
            if cr2 - ext < 0 or cr2 + ext > 2 * (hi - 1):
                continue
            if cc2 - ext < 0 or cc2 + ext > 2 * (wi - 1):
                continue
            if cr2 % 2:
                mark = [(cr2 // 2, cc2 // 2), (cr2 // 2, cc2 // 2 + 1),
                        (cr2 // 2 + 1, cc2 // 2), (cr2 // 2 + 1, cc2 // 2 + 1)]
            else:
                pr, pc = cr2 // 2, cc2 // 2
                mark = [(pr, pc), (pr-1, pc-1), (pr-1, pc+1), (pr+1, pc-1), (pr+1, pc+1)]
            if any(not (0 <= p[0] < hi and 0 <= p[1] < wi) for p in mark):
                continue
            mv = {int(I[p]) for p in mark}
            if len(mv) != 1 or mv.pop() == bgc:
                continue
            agree = 0
            ok = True
            for (r, c, v) in fg:
                for (nr, nc) in rot_images(r, c, cr2, cc2):
                    if not (0 <= nr < hi and 0 <= nc < wi):
                        ok = False; break
                    u = int(I[nr, nc])
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
                tr = (cr2 - (side - 1)) // 2
                tc = (cc2 - (side - 1)) // 2
                if tr < 0 or tc < 0 or tr + side > hi or tc + side > wi:
                    continue
                P = I[tr:tr + side, tc:tc + side]
                if np.array_equal(P, np.rot90(P)) and int((P != bgc).sum()) >= 2:
                    core = side; break
            key = (-core, abs(cr2 - (r0 + r1)) + abs(cc2 - (c0 + c1)), -agree, ext)
            if best is None or key < best[0]:
                best = (key, (cr2, cc2))
    if best is None:
        return [34], [[0, 0, hi - 1, wi - 1]]
    cr2, cc2 = best[1]

    # Quarter-turn copies, one at a time (90, 180, 270), each painted colour
    # group by colour group; cells already holding that colour are skipped so
    # every op visibly changes the grid.
    cur = I.copy()
    ops, sels = [], []
    for k in range(3):
        copy_cells = {}
        for (r, c, v) in fg:
            nr, nc = rot_images(r, c, cr2, cc2)[k]
            if 0 <= nr < hi and 0 <= nc < wi:
                copy_cells[(nr, nc)] = v
        groups = {}
        for p, v in copy_cells.items():
            if int(cur[p]) != v:
                groups.setdefault(v, []).append(p)
        for v in sorted(groups):
            cells = sorted(groups[v])
            ops.append(int(v))
            sels.append(sel_of(cells))
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
