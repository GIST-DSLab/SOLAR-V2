"""
ARC Task: 0e206a2e (RE-ARC) — LLM-generated grid_maker
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
    bgc, acol, bcol, ccol, dcol = random.sample(range(10), 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": dcol}


def _dn(p):
    r, c = p
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _nb(p):
    r, c = p
    return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _shape(cells):
    rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def _diag(cells):
    return len({x - y for x, y in cells}) == 1 or len({x + y for x, y in cells}) == 1


def _tf(name, g):
    H, W = len(g), len(g[0])
    if name == "id":
        return [row[:] for row in g]
    if name == "dm":
        return [[g[c][r] for c in range(H)] for r in range(W)]
    if name == "cm":
        return [[g[H - 1 - c][W - 1 - r] for c in range(H)] for r in range(W)]
    if name == "vm":
        return [row[::-1] for row in g]
    if name == "hm":
        return [row[:] for row in g[::-1]]
    if name == "r90":
        return [[g[H - 1 - c][r] for c in range(H)] for r in range(W)]
    if name == "r180":
        return [row[::-1] for row in g[::-1]]
    return [[g[c][W - 1 - r] for c in range(H)] for r in range(W)]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None,
             ccol=None, Dcol=None, **kw) -> dict:
    def unifint(lo, hi):
        a = lo + diff_lb * (hi - lo); b = lo + diff_ub * (hi - lo)
        return max(lo, min(hi, int(round(random.uniform(a, b)))))
    if bgc is None:
        bgc, acol, bcol, ccol, Dcol = random.sample(range(10), 5)
    cap = min(max_h, max_w)          # dihedral transforms may swap h and w
    h = unifint(10, max(10, cap))
    w = unifint(10, max(10, cap))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    nsrcs = unifint(1, min(h, w) // 5)
    srcs, abclist = [], []
    tr = succ = 0
    while tr < 5 * nsrcs and succ < nsrcs:
        tr += 1
        objsize = unifint(5, 20)
        bb = {(i, j) for i in range(7) for j in range(7)}
        sp = random.choice(sorted(bb))
        shp = {sp}
        def grow():
            cand = set()
            for p in shp:
                cand |= _dn(p)
            cand = sorted((cand & bb) - shp)
            shp.add(random.choice(cand))
        for _ in range(objsize - 1):
            grow()
        while 1 in _shape(shp):
            grow()
        while _diag(shp):
            grow()
        mr = min(p[0] for p in shp); mc = min(p[1] for p in shp)
        shp = [(r - mr, c - mc) for r, c in shp]
        random.shuffle(shp)
        ok = False
        for _ in range(500):
            a, b, c = shp[:3]
            if not (1 in _shape({a, b, c}) or _diag({a, b, c})):
                ok = True
                break
            random.shuffle(shp)
        if not ok:
            continue
        a, b, c = shp[:3]
        key = sorted(_shape({a, b, c}))
        if key in abclist:
            continue
        D = shp[3:]
        markers = [(acol, a), (bcol, b), (ccol, c)]
        obj = markers + [(Dcol, p) for p in D]
        opts = [(i, j) for (i, j) in inds if all((i + r, j + c2) in inds for r, c2 in shp)]
        if not opts:
            continue
        li, lj = random.choice(opts)
        succ += 1
        placed = set()
        for col, (r, c2) in obj:
            gi[li + r][lj + c2] = col
            go[li + r][lj + c2] = -1
            placed.add((li + r, lj + c2))
        rm = set(placed)
        for p in placed:
            rm |= _nb(p)
        inds -= rm
        srcs.append((obj, markers))
        abclist.append(key)
    num = unifint(1, (h * w) // 30)
    tr = succ = 0
    while succ < num and tr < 10 * num:
        fn = random.choice(["id", "dm", "cm", "vm", "hm", "r90", "r180", "r270"])
        gi = _tf(fn, gi); go = _tf(fn, go)
        H, W = len(go), len(go[0])
        nonbg = {(i, j) for i in range(H) for j in range(W) if go[i][j] != bgc}
        bad = set()
        for p in nonbg:
            bad |= _nb(p)
        full = {(i, j) for i in range(H) for j in range(W) if go[i][j] == bgc} - bad
        obj, markers = random.choice(srcs)
        if not full:
            break
        li, lj = random.choice(sorted(full))
        if all((li + r, lj + c2) in full for _, (r, c2) in obj):
            succ += 1
            for col, (r, c2) in markers:
                gi[li + r][lj + c2] = col
            for col, (r, c2) in obj:
                go[li + r][lj + c2] = col
        tr += 1
    go = [[bgc if v == -1 else v for v in row] for row in go]
    # reject ambiguous grids (spurious marker triples formed across placements)
    ops, sels = derive_operations(gi, go)
    g = np.array(gi)
    for o, s in zip(ops, sels):
        if o < 10:
            for r, c in s["cells"]:
                g[r, c] = o
    if not (g == np.array(go)).all() or len(set(np.array(gi).flatten().tolist())) < 5:
        raise ValueError("ambiguous instance; retry")
    return {"input": gi, "output": go}


# The 8 dihedral maps of a relative offset (dr, dc): rotations and reflections.
_DIHEDRAL = [
    lambda r, c: (r, c), lambda r, c: (c, -r), lambda r, c: (-r, -c), lambda r, c: (-c, r),
    lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, r), lambda r, c: (-c, -r),
]


def derive_operations(I, O, examples=None):
    """
    Reading (template as a rigid unit, as in the golf entry): each 4-colour
    component is a TEMPLATE (body colour + three single-cell markers).  The
    template is lifted off its spot (erased to background) and laid down --
    rotated/reflected as a whole -- onto every bare marker triple matching a
    dihedral image of its own triple; its body is drawn around those markers.
    Work is grouped template by template: erase it, then draw each copy of it.
    Everything is measured from I; O is never read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    ops, sels = [], []

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; cells = []
                while st:
                    a, b = st.pop(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True; st.append((na, nb))
                comps.append(sorted(cells))

    templates = []
    for cells in comps:
        cols = [int(I[p]) for p in cells]
        if len(set(cols)) != 4:
            continue
        body_col = Counter(cols).most_common(1)[0][0]
        markers = [(int(I[p]), p) for p in cells if int(I[p]) != body_col]
        body = [p for p in cells if int(I[p]) == body_col]
        if len(markers) == 3 and body:
            templates.append((cells, body_col, markers, body))

    # I with every template lifted off: only bare marker triples remain
    bare = I.copy()
    for cells, _, _, _ in templates:
        for p in cells:
            bare[p] = bgc

    g = I.copy()
    for cells, body_col, markers, body in templates:
        # lift the template off its spot
        ops.append(bgc); sels.append(sel_of(cells))
        for p in cells:
            g[p] = bgc
        (c0, (r0, k0)) = markers[0]
        placements, seen_pl = [], set()
        for T in _DIHEDRAL:
            tm = [(col, T(p[0] - r0, p[1] - k0)) for col, p in markers]
            tb = [T(p[0] - r0, p[1] - k0) for p in body]
            for ar in range(h):
                for ac in range(w):
                    ok = True
                    for col, (dr, dc) in tm:
                        rr, cc = ar + dr, ac + dc
                        if not (0 <= rr < h and 0 <= cc < w) or bare[rr, cc] != col:
                            ok = False; break
                    if not ok:
                        continue
                    pc = frozenset((ar + dr, ac + dc) for dr, dc in tb
                                   if 0 <= ar + dr < h and 0 <= ac + dc < w)
                    if pc and pc not in seen_pl:
                        seen_pl.add(pc); placements.append(sorted(pc))
        # lay the transformed template body down around each matching triple
        for pc in placements:
            todo = [p for p in pc if g[p] != body_col]
            if not todo:
                continue
            ops.append(int(body_col)); sels.append(sel_of(todo))
            for p in todo:
                g[p] = body_col

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
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
                                f"for task 0e206a2e"
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
                    f"Failed to build a complete episode for task 0e206a2e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0e206a2e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
