"""
ARC Task: 72ca375d (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    trgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "trgc": trgc}


def _nbrs(p):
    r, c = p
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def _norm(s):
    r0 = min(p[0] for p in s); c0 = min(p[1] for p in s)
    return {(r - r0, c - c0) for r, c in s}


def _vmir(s):
    c0 = min(p[1] for p in s); c1 = max(p[1] for p in s)
    return {(r, c0 + c1 - c) for r, c in s}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + round(lb * (b - a)); hi = a + round(ub * (b - a))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, trgc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    if trgc is None:
        trgc = random.choice(remcols)
    h = _unif(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unif(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = _unif(diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = _unif(diff_lb, diff_ub, (2, 8))
    srcobjwh = _unif(diff_lb, diff_ub, (1, 4))
    bnds = {(i, j) for i in range(srcobjh) for j in range(srcobjwh)}
    sp = (random.randint(0, srcobjh - 1), srcobjwh - 1)
    srcobj = {sp}
    bnds.discard(sp)
    ncellsd = _unif(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = random.choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = _unif(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for _ in range(ncells - 1):
        nb = set().union(*[_nbrs(p) for p in srcobj])
        cand = sorted((bnds - srcobj) & nb)
        if not cand:
            break
        srcobj.add(random.choice(cand))
    srcobj = _norm(srcobj)
    wd = max(p[1] for p in srcobj) + 1
    srcobj = srcobj | {(r, c + wd) for r, c in _vmir(srcobj)}
    sh = max(p[0] for p in srcobj) + 1
    sw = max(p[1] for p in srcobj) + 1
    if sh > h or sw > w:
        raise ValueError("too big")
    go = np.full((sh, sw), bgc, dtype=int)
    for r, c in srcobj:
        go[r, c] = trgc
    gi = np.full((h, w), bgc, dtype=int)
    li = random.randint(0, h - sh); lj = random.randint(0, w - sw)
    shftd = {(r + li, c + lj) for r, c in srcobj}
    for r, c in shftd:
        gi[r, c] = trgc
    indss = {(i, j) for i in range(h) for j in range(w)}
    # keep the symmetric object's whole window (plus a 1-cell ring) free of
    # distractors so the output window is exactly the object on background
    box = {(i, j) for i in range(li - 1, li + sh + 1) for j in range(lj - 1, lj + sw + 1)}
    indss = indss - box
    maxtrials = 4 * nobjs
    tr = 0; succ = 0
    remcands = {(i, j) for i in range(8) for j in range(8)} - srcobj
    while succ < nobjs and tr <= maxtrials:
        if not indss:
            break
        while True:
            newobj = set(srcobj)
            numpert = 64 - _unif(diff_lb, diff_ub, (1, 63))
            for _ in range(numpert):
                isadd = random.choice((True, False))
                if isadd and len(newobj) < 64:
                    nb = set().union(*[_nbrs(p) for p in newobj])
                    cnd = sorted((remcands - newobj) & nb)
                    if not cnd:
                        break
                    newobj.add(random.choice(cnd))
                if not isadd and len(newobj) > 2:
                    newobj.remove(random.choice(sorted(newobj)))
            newobj = _norm(newobj)
            # keep largest 8-connected piece
            left = set(newobj); best = set()
            while left:
                st = [left.pop()]; comp = set(st)
                while st:
                    p = st.pop()
                    for q in _nbrs(p):
                        if q in left:
                            left.remove(q); comp.add(q); st.append(q)
                if len(comp) > len(best):
                    best = comp
            newobj = _norm(best)
            if newobj != _vmir(newobj):
                break
        col = random.choice(remcols)
        loccands = [(i, j) for (i, j) in indss
                    if all((r + i, c + j) in indss for r, c in newobj)]
        if not loccands:
            tr += 1
            continue
        i, j = random.choice(loccands)
        placed = {(r + i, c + j) for r, c in newobj}
        for r, c in placed:
            gi[r, c] = col
        succ += 1
        indss = (indss - placed) - set().union(*[_nbrs(p) for p in placed])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Rule: one object in the scene is left-right mirror-symmetric; the answer is
    # that object's bounding window cut out of I.  Golf reading = "the colour whose
    # cells' bbox equals its own mirror"; maker reading = "the 8-connected object
    # equal to its own mirror".  Both lead to the same single action (crop to that
    # window); they differ only in the unit.  The generator may paint distractors in
    # the target colour, so the object (component) is the primary unit and the
    # colour class is the fallback.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    bgc = None
    if examples:
        cnt = Counter()
        for ex_i, _ in examples:
            a = np.asarray(ex_i, dtype=int)
            cnt[Counter(a.flatten().tolist()).most_common(1)[0][0]] += 1
        bgc = cnt.most_common(1)[0][0]
    if bgc is None or bgc not in set(I.flatten().tolist()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def sym(cells):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            m[r - r0, c - c0] = True
        return np.array_equal(m, np.fliplr(m)), (r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]; st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                cr, cc = st.pop(); cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True; st.append((nr, nc))
            comps.append(cells)

    win = None
    best = None
    for cells in comps:
        ok, box = sym(cells)
        if ok:
            key = (len(cells), box[2] * box[3])
            if best is None or key > best:
                best, win = key, box
    if win is None:
        # colour-class reading (golf): all cells of one colour, mirror-symmetric
        for col in sorted(set(I.flatten().tolist()) - {bgc}):
            cells = [tuple(p) for p in np.argwhere(I == col)]
            ok, box = sym(cells)
            if ok:
                win = box
                break
    if win is None:
        win = (0, 0, hi, wi)
    r0, c0, bh, bw = win
    ops, sels = [], []
    if (r0, c0, bh, bw) != (0, 0, hi, wi):
        # full rectangle intended: the symmetric object's window, background included
        ops.append(33); sels.append([r0, c0, bh - 1, bw - 1])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
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
                                f"for task 72ca375d"
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
                    f"Failed to build a complete episode for task 72ca375d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72ca375d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
