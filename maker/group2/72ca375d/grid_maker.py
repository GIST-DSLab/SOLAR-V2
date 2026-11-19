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
    bgc = random.choice(range(10))
    trgc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "trgc": trgc}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    return min(b, max(a, round(a + (b - a) * d)))


def _nbrs(p):
    i, j = p
    return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _mapn(s):
    out = set()
    for p in s:
        out |= _nbrs(p)
    return out


def _norm(s):
    mi = min(i for i, _ in s)
    mj = min(j for _, j in s)
    return {(i - mi, j - mj) for i, j in s}


def _vmir(s):
    d = min(j for _, j in s) + max(j for _, j in s)
    return {(i, d - j) for i, j in s}


def _shape(s):
    return (max(i for i, _ in s) - min(i for i, _ in s) + 1,
            max(j for _, j in s) - min(j for _, j in s) + 1)


def _compress(g):
    g = np.asarray(g)
    rows = [r for r in range(g.shape[0]) if len(set(g[r].tolist())) > 1]
    cols = [c for c in range(g.shape[1]) if len(set(g[:, c].tolist())) > 1]
    return g[np.ix_(rows, cols)]


def _largest_comp(g, val):
    h, w = g.shape
    seen = set()
    best = set()
    for r in range(h):
        for c in range(w):
            if g[r, c] != val or (r, c) in seen:
                continue
            st = [(r, c)]
            seen.add((r, c))
            comp = set()
            while st:
                p = st.pop()
                comp.add(p)
                for q in _nbrs(p):
                    if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and g[q] == val:
                        seen.add(q)
                        st.append(q)
            if len(comp) > len(best):
                best = comp
    return best


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, trgc=1, **kw) -> dict:
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = _unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = _unifint(diff_lb, diff_ub, (2, 8))
    srcobjwh = _unifint(diff_lb, diff_ub, (1, 4))
    bnds = {(i, j) for i in range(srcobjh) for j in range(srcobjwh)}
    spi = random.randint(0, srcobjh - 1)
    sp = (spi, srcobjwh - 1)
    srcobj = {sp}
    bnds.discard(sp)
    ncellsd = _unifint(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = random.choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = _unifint(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for _ in range(ncells - 1):
        cands = sorted((bnds - srcobj) & _mapn(srcobj))
        if not cands:
            break
        srcobj.add(random.choice(cands))
    srcobj = _norm(srcobj)
    wd = _shape(srcobj)[1]
    srcobj = srcobj | {(i, j + wd) for i, j in _vmir(srcobj)}
    srcobjh, srcobjw = _shape(srcobj)
    remcols = [c for c in cols if c != bgc]
    go = np.full((srcobjh, srcobjw), bgc, dtype=int)
    for i, j in srcobj:
        go[i, j] = trgc
    loci = random.randint(0, h - srcobjh)
    locj = random.randint(0, w - srcobjw)
    gi = np.full((h, w), bgc, dtype=int)
    shftd = {(i + loci, j + locj) for i, j in srcobj}
    for i, j in shftd:
        gi[i, j] = trgc
    indss = {(i, j) for i in range(h) for j in range(w)}
    indss = (indss - shftd) - _mapn(shftd)
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    remcands = {(i, j) for i in range(8) for j in range(8)} - srcobj
    while succ < nobjs and tr <= maxtrials:
        if len(indss) == 0:
            break
        while True:
            newobj = set(srcobj)
            numperti = _unifint(diff_lb, diff_ub, (1, 63))
            numpert = 64 - numperti
            for _ in range(numpert):
                isadd = random.choice((True, False))
                if isadd and len(newobj) < 64:
                    cndds = sorted((remcands - newobj) & _mapn(newobj))
                    if len(cndds) == 0:
                        break
                    newobj.add(random.choice(cndds))
                if not isadd and len(newobj) > 2:
                    newobj.discard(random.choice(sorted(newobj)))
            newobj = _norm(newobj)
            a, b = _shape(newobj)
            cc = np.full((a + 2, b + 2), -1, dtype=int)
            for i, j in newobj:
                cc[i + 1, j + 1] = -2
            cc2 = _compress(cc)
            comp = _largest_comp(cc2, -2)
            if not comp:
                continue
            newobj = _norm(comp)
            if newobj != _vmir(newobj):
                break
        col = random.choice(remcols)
        loccands = [(i, j) for (i, j) in indss
                    if all((a + i, b + j) in indss for a, b in newobj)]
        if len(loccands) == 0:
            tr += 1
            continue
        li, lj = random.choice(sorted(loccands))
        newobj = {(a + li, b + lj) for a, b in newobj}
        for i, j in newobj:
            gi[i, j] = col
        succ += 1
        indss = (indss - newobj) - _mapn(newobj)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected same-colour objects; the target is the one whose shape
    # equals its own left-right mirror (measured from I only).
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for (a, b) in cells:
                mask[a - r0, b - c0] = True
            if not np.array_equal(mask, np.fliplr(mask)):
                continue
            key = (len(cells), mask.size)
            if best is None or key > best[0]:
                best = (key, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    if best is None:
        r0, c0, bh, bw = 0, 0, hi, wi
    else:
        _, r0, c0, bh, bw = best

    ops, sels = [], []
    # Stage 1 (row unit): keep only the horizontal band of rows spanned by the
    # symmetric object — a full-width rectangle, rows r0..r0+bh-1.
    if bh != hi:
        ops.append(33)
        sels.append([r0, 0, bh - 1, wi - 1])
    # Stage 2 (column unit): from that band keep only the object's columns —
    # the full-height rectangle of the band, columns c0..c0+bw-1.
    if bw != wi:
        ops.append(33)
        sels.append([0, c0, bh - 1, bw - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
