"""
ARC Task: 97a05b5b (RE-ARC) — LLM-generated grid_maker
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
import numpy as np

# ---------------------------------------------------------------- helpers ----
_TF = [
    ("identity", lambda a: a),
    ("rot180",   lambda a: np.rot90(a, 2)),
    ("rot90",    lambda a: np.rot90(a, 3)),   # DSL rot90 == clockwise
    ("rot270",   lambda a: np.rot90(a, 1)),   # DSL rot270 == counterclockwise
    ("hmirror",  lambda a: a[::-1, :]),
    ("vmirror",  lambda a: a[:, ::-1]),
    ("cmirror",  lambda a: a[::-1, ::-1].T),  # anti-transpose
    ("dmirror",  lambda a: a.T),              # transpose
]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(max(0.0, min(1.0, diff_lb)), max(0.0, min(1.0, diff_ub)))
    return min(max(a, int(round(a + (b - a) * d))), b)


def _components(mask):
    """4-connected components of a boolean mask -> list of cell lists."""
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            stack.append((na, nb))
                out.append(cells)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _occ(sub, pat):
    """All top-left positions where pat matches sub exactly."""
    sh, sw = sub.shape
    ph, pw = pat.shape
    hits = []
    if ph > sh or pw > sw:
        return hits
    for i in range(sh - ph + 1):
        for j in range(sw - pw + 1):
            if np.array_equal(sub[i:i + ph, j:j + pw], pat):
                hits.append((i, j))
    return hits


def _analyse(I):
    """Recover the rule from the input alone.

    Non-background components split into (a) solid two-colour rectangles = the
    "keys", and (b) the perforated square.  A key's square-coloured cells spell a
    shape that appears inside the square as a background-coloured notch, in one
    of the 8 dihedral orientations; the key rectangle is stamped over that notch.
    """
    A = np.asarray(I, dtype=int)
    bg = Counter(A.flatten().tolist()).most_common(1)[0][0]
    keys, rest = [], []
    for cells in _components(A != bg):
        r0, c0, r1, c1 = _bbox(cells)
        solid = len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
        ncol = len({A[r, c] for r, c in cells})
        (keys if (solid and ncol == 2) else rest).append(cells)
    if not rest:
        return None
    rest_cells = [p for cells in rest for p in cells]
    R0, C0, R1, C1 = _bbox(rest_cells)
    sqc = Counter([int(A[r, c]) for r, c in rest_cells]).most_common(1)[0][0]
    sub = A[R0:R1 + 1, C0:C1 + 1].copy()
    out = sub.copy()
    placements, clean = [], True
    for cells in keys:
        kr0, kc0, kr1, kc1 = _bbox(cells)
        kg = A[kr0:kr1 + 1, kc0:kc1 + 1]
        others = sorted(set(kg.flatten().tolist()) - {sqc})
        if len(others) != 1:
            clean = False
            continue
        col = others[0]
        pat = np.where(kg == sqc, bg, sqc)      # the stencil as it looks in the square
        found, seen_grids = None, {}
        for _, f in _TF:
            p = np.ascontiguousarray(f(pat))
            sig = (p.shape, p.tobytes())
            if sig in seen_grids:
                continue
            hits = _occ(sub, p)
            seen_grids[sig] = hits
            if hits and found is None:
                found = (p, np.ascontiguousarray(f(kg)), hits)
        n_matching = sum(1 for v in seen_grids.values() if v)
        if found is None:
            clean = False
            continue
        p, k, hits = found
        if n_matching != 1 or len(hits) != 1:
            clean = False                        # ambiguous for the reference rule
        r, c = hits[0]
        out[r:r + k.shape[0], c:c + k.shape[1]] = k
        placements.append((r, c, p, col, bg, sqc))
    return {"bg": bg, "sqc": sqc, "origin": (R0, C0), "sub": sub,
            "out": out, "placements": placements, "clean": clean,
            "nkeys": len(keys)}


# --------------------------------------------------------- sample_colors ----
def sample_colors(num_examples=None) -> dict:
    # bgc and sqc are the two structural roles (canvas / square).  The key
    # colours are irrelevant to the rule (shape matching only), so they stay free.
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


# --------------------------------------------------------------- generate ----
def _idx_mirror(kind, obj):
    ui = min(i for i, _ in obj); uj = min(j for _, j in obj)
    li = max(i for i, _ in obj); lj = max(j for _, j in obj)
    if kind == 'h':
        d = ui + li
        return {(d - i, j) for i, j in obj}
    if kind == 'v':
        d = uj + lj
        return {(i, d - j) for i, j in obj}
    if kind == 'd':
        return {(j - uj + ui, i - ui + uj) for i, j in obj}
    o = _idx_mirror('v', obj)                      # cmirror = v(d(v(obj)))
    o = _idx_mirror('d', o)
    return _idx_mirror('v', o)


def _normalize(obj):
    ui = min(i for i, _ in obj); uj = min(j for _, j in obj)
    return frozenset((i - ui, j - uj) for i, j in obj)


def _grid_tf(kind, g):
    if kind == 'i':
        return g
    if kind == 'h':
        return g[::-1, :]
    if kind == 'v':
        return g[:, ::-1]
    if kind == 'd':
        return g.T
    return g[::-1, ::-1].T


def _build(diff_lb, diff_ub, max_h, max_w, bgc, sqc):
    h = _unifint(diff_lb, diff_ub, (min(15, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(15, max_w), max_w))
    if h < 12 or w < 12:
        return None
    sgh = random.randint(h // 3, h // 3 * 2)
    sgw = random.randint(w // 3, w // 3 * 2)
    if sgh < 4 or sgw < 4:
        return None
    oh = random.randint(2, sgh // 2)
    ow = random.randint(2, sgw // 2)
    nobjs = _unifint(diff_lb, diff_ub, (1, 8))

    # ---- shape pool: connected blobs made symmetric, never a solid rectangle,
    #      all distinct from one another under every dihedral transform.
    cands = [(i, j) for i in range(oh) for j in range(ow)]
    objs, forbidden = [], set()
    tr, maxtr = 0, 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {random.choice(cands)}
        ncells = random.randint(1, oh * ow - 1)
        for _ in range(ncells - 1):
            nbrs = set()
            for i, j in obj:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di or dj:
                            nbrs.add((i + di, j + dj))
            pool = list((set(cands) - obj) & nbrs)
            if not pool:
                break
            obj.add(random.choice(pool))
        obj = obj | _idx_mirror(random.choice('dvch'), obj)
        oi, oj = min(i for i, _ in obj), min(j for _, j in obj)
        li, lj = max(i for i, _ in obj), max(j for _, j in obj)
        if len(obj) == (li - oi + 1) * (lj - oj + 1):
            continue                                # solid rectangle -> no notch
        objn = _normalize(obj)
        if objn not in forbidden:
            objs.append(objn)
        for f1 in 'ivdch':
            for f2 in 'ivdch':
                o = objn
                if f2 != 'i':
                    o = _idx_mirror(f2, o)
                if f1 != 'i':
                    o = _idx_mirror(f1, o)
                forbidden.add(_normalize(o))
    if not objs:
        return None

    loci = random.randint(0, h - sgh)
    locj = random.randint(0, w - sgw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + sgh, locj:locj + sgw] = sqc
    go = np.full((sgh, sgw), sqc, dtype=int)

    gi_free = np.ones((h, w), bool)
    gi_free[max(0, loci - 1):loci + sgh + 1, max(0, locj - 1):locj + sgw + 1] = False
    go_free = np.ones((sgh, sgw), bool)
    remcols = [c for c in range(10) if c not in (bgc, sqc)]

    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr and objs:
        tr += 1
        obj = random.choice(objs)
        col = random.choice(remcols)
        oi = max(i for i, _ in obj) + 1
        oj = max(j for _, j in obj) + 1
        subgi = np.full((oi, oj), col, dtype=int)
        for i, j in obj:
            subgi[i, j] = sqc
        subgo = np.ascontiguousarray(
            _grid_tf(random.choice('ivdch'), _grid_tf(random.choice('ivdch'), subgi)))
        ohi, owi = subgi.shape
        oho, owo = subgo.shape
        go_cands = [(i, j)
                    for i in range(sgh - oho + 1) for j in range(sgw - owo + 1)
                    if go_free[max(0, i - 1):i + oho + 1, max(0, j - 1):j + owo + 1].all()]
        if not go_cands:
            continue
        # the generator's own placement window for the keys (left strip)
        gi_cands = [(i, j)
                    for i in range(h - ohi + 1) for j in range(min(owi, w - owi) + 1)
                    if gi_free[i:i + ohi, j:j + owi].all()]
        if not gi_cands:
            continue
        gr, gc = random.choice(go_cands)
        ir, ic = random.choice(gi_cands)
        gi[ir:ir + ohi, ic:ic + owi] = subgi                 # the key, outside
        go[gr:gr + oho, gc:gc + owo] = subgo                 # the stamped result
        region = gi[loci + gr:loci + gr + oho, locj + gc:locj + gc + owo]
        region[subgo == sqc] = bgc                           # punch the notch
        gi_free[max(0, ir - 1):ir + ohi + 1, max(0, ic - 1):ic + owi + 1] = False
        go_free[max(0, gr - 1):gr + oho + 1, max(0, gc - 1):gc + owo + 1] = False
        objs.remove(obj)
        remcols.remove(col)
        succ += 1
    if succ < 1:
        return None

    # Keep only instances the reference rule resolves unambiguously.
    info = _analyse(gi)
    if info is None or not info["clean"]:
        return None
    if info["bg"] != bgc or info["sqc"] != sqc or info["nkeys"] != succ:
        return None
    if info["origin"] != (loci, locj) or info["sub"].shape != (sgh, sgw):
        return None
    if not np.array_equal(info["out"], go):
        return None
    return {"input": gi.tolist(), "output": go.tolist()}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(range(10), 2)
    for _ in range(400):
        res = _build(diff_lb, diff_ub, max_h, max_w, bgc, sqc)
        if res is not None:
            return res
    for _ in range(400):                       # easier settings as a fallback
        res = _build(0.0, 0.3, max_h, max_w, bgc, sqc)
        if res is not None:
            return res
    raise RuntimeError("generation failed")


# ------------------------------------------------------- derive_operations ----
def derive_operations(I, O):
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            uniq = sorted({(int(r), int(c)) for r, c in cells})
            return {"cells": [[r, c] for r, c in uniq]}

    A = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    info = _analyse(A)
    R0, C0 = info["origin"]
    sqc = info["sqc"]

    # Work in input coordinates while the key rectangles are still on screen, so
    # each notch is filled next to the key that explains it; crop last.
    for (r, c, pat, col, bg, _sqc) in sorted(info["placements"],
                                             key=lambda p: (p[0], p[1])):
        ph, pw = pat.shape
        hole = [(R0 + r + i, C0 + c + j)
                for i in range(ph) for j in range(pw) if pat[i, j] == bg]
        body = [(R0 + r + i, C0 + c + j)
                for i in range(ph) for j in range(pw) if pat[i, j] != bg]
        if hole:
            ops.append(int(sqc)); sels.append(sel_of(hole))   # heal this notch
        if body:
            ops.append(int(col)); sels.append(sel_of(body))   # stamp the key body

    # Full-rectangle selection on purpose: crop the canvas down to the square.
    ops.append(33); sels.append([int(R0), int(C0), ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
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
                                f"for task 97a05b5b"
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
                    f"Failed to build a complete episode for task 97a05b5b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97a05b5b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
