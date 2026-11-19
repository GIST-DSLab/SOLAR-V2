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
    cols = list(range(10))
    bgc, acol, bcol, ccol, Dcol = random.sample(cols, 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 5)))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1):
            shuffle(shp)
            a, b, c = shp[:3]
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading: each multicolour TEMPLATE object (4 colours: one body colour + 3
    single-cell markers) is picked up, flipped/rotated in place, and slid with
    unit Moves until its markers sit on a bare marker triple of the same
    colours elsewhere.  If a template fits several triples, after it leaves
    home the original is pasted back (CopyI of the template) and transplanted
    again.  Finally the template's home is left as background.
    Only I is inspected; O is never read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # ---- components (4-conn, non-bg) ----
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            st.append((na, nb))
                comps.append(sorted(cells))

    templates = []
    tcells_all = set()
    for cells in comps:
        colors = [int(I[r, c]) for r, c in cells]
        if len(set(colors)) != 4:
            continue
        body_col = int(Counter(colors).most_common(1)[0][0])
        if sum(1 for v in colors if v != body_col) != 3:
            continue
        templates.append((cells, body_col))
        tcells_all |= set(cells)

    NP = {26: np.fliplr, 27: np.flipud,
          24: lambda a: np.rot90(a, 1), 25: lambda a: np.rot90(a, 3)}
    SEQS = [[], [26], [27], [26, 27], [24], [25], [26, 24], [26, 25]]

    def rot_pos(xmin, xmax, ymin, ymax, k):
        # ARCLE gen_rotate position update for a freshly grabbed object
        hh = xmax - xmin + 1
        ww = ymax - ymin + 1
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        if hh % 2 == ww % 2:
            return int(np.floor(cx - cy + ymin)), int(np.floor(cy - cx + xmin))
        sig = (k + 2) % 4 - 2
        mod = 1 - (k % 2)
        mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
        my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
        return int(np.floor(mx)), int(np.floor(my))

    # ---- tiny mirror of the visible grid, to keep every step meaningful ----
    g = I.copy()
    ops, sels = [], []
    hist = {g.tobytes(): -1}
    last_clip = [-10 ** 9]

    def visible(bg, obj, pos):
        v = bg.copy()
        for (i, j), val in obj.items():
            rr, cc = pos[0] + i, pos[1] + j
            if 0 <= rr < h and 0 <= cc < w and val != 0:
                v[rr, cc] = val
        return v

    def push(op, sel, newg):
        nonlocal g
        ops.append(op); sels.append(sel)
        g = newg
        if op in (28, 29, 30):
            last_clip[0] = len(ops) - 1
        hist[g.tobytes()] = len(ops) - 1

    def ok_state(state, pending):
        b = state.tobytes()
        if b in pending:
            return False
        prev = hist.get(b)
        return prev is None or prev < last_clip[0]

    used_triples = set()
    placed = {}

    for cells, body_col in templates:
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells); c1 = max(c for _, c in cells)
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        patch = np.full((th, tw), -1, dtype=int)   # -1 = not part of object
        for r, c in cells:
            patch[r - r0, c - c0] = I[r, c]
        home = sorted(cells)

        placements = []
        for seq in SEQS:
            p = patch
            for op in seq:
                p = NP[op](p)
            ph, pw = p.shape
            marks = [(i, j, int(p[i, j])) for i in range(ph) for j in range(pw)
                     if p[i, j] >= 0 and p[i, j] != body_col]
            for i0 in range(-ph + 1, h):
                for j0 in range(-pw + 1, w):
                    pos = []
                    for i, j, v in marks:
                        rr, cc = i0 + i, j0 + j
                        if not (0 <= rr < h and 0 <= cc < w) or I[rr, cc] != v \
                                or (rr, cc) in tcells_all:
                            break
                        pos.append((rr, cc))
                    else:
                        key = frozenset(pos)
                        if key not in used_triples:
                            used_triples.add(key)
                            placements.append((seq, p, i0, j0))

        zeroed = set()
        if placements and body_col == 0:
            # a colour-0 body is transparent to Move/Paste (ARCLE never writes 0),
            # so a slid copy would leave no trace: draw each body around its triple
            for seq, p, ti, tj in placements:
                body = [(ti + i, tj + j) for i in range(p.shape[0]) for j in range(p.shape[1])
                        if p[i, j] == 0 and 0 <= ti + i < h and 0 <= tj + j < w]
                todo = [q for q in body if g[q] != 0]
                if todo:
                    ng = g.copy()
                    for q in todo:
                        ng[q] = 0
                    push(0, sel_of(todo), ng)
            zeroed = set(home)
        elif placements:
            copied = False
            for k, (seq, p, ti, tj) in enumerate(placements):
                if k > 0:
                    # put the template back at home for another transplant
                    if not copied:
                        push(28, sel_of(home), g)
                        copied = True
                    ng = g.copy()
                    for r, c in home:
                        if I[r, c] != 0:
                            ng[r, c] = I[r, c]
                    push(30, [r0, c0, 0, 0], ng)
                # orient in place: every transform op grabs the object's cells
                pos = (r0, c0)
                cur = patch.copy()
                bg = None
                obj = None
                regrab = set()
                for op in seq:
                    sel_cells = [(pos[0] + i, pos[1] + j) for i in range(cur.shape[0])
                                 for j in range(cur.shape[1]) if cur[i, j] >= 0]
                    if bg is not None:
                        # re-grab: a 0-coloured part of the object is invisible
                        # on the grid, so restore it before picking it up again
                        z = [(pos[0] + i, pos[1] + j) for i in range(cur.shape[0])
                             for j in range(cur.shape[1])
                             if cur[i, j] == 0 and 0 <= pos[0] + i < h and 0 <= pos[1] + j < w
                             and g[pos[0] + i, pos[1] + j] != 0]
                        if z:
                            ng = g.copy()
                            for q in z:
                                ng[q] = 0
                            push(0, sel_of(z), ng)
                    if bg is not None:
                        regrab |= set(sel_cells)
                    bg = g.copy()
                    for r, c in sel_cells:
                        bg[r, c] = 0
                    zeroed |= set(sel_cells)
                    if op in (24, 25):
                        pos = rot_pos(pos[0], pos[0] + cur.shape[0] - 1,
                                      pos[1], pos[1] + cur.shape[1] - 1, 1 if op == 24 else 3)
                    cur = NP[op](cur)
                    obj = {(i, j): int(cur[i, j]) for i in range(cur.shape[0])
                           for j in range(cur.shape[1]) if cur[i, j] >= 0}
                    push(op, sel_of(sel_cells), visible(bg, obj, pos))
                if obj is None:
                    obj = {(i, j): int(cur[i, j]) for i in range(cur.shape[0])
                           for j in range(cur.shape[1]) if cur[i, j] >= 0}
                    bg = g.copy()
                    for r, c in home:
                        bg[r, c] = 0
                    zeroed |= set(home)
                    first_sel = sel_of(home)
                else:
                    first_sel = None
                # slide to the triple; choose a route that never revisits a grid
                dr, dc = ti - pos[0], tj - pos[1]
                vs = [21 if dr > 0 else 20] * abs(dr)
                hs = [22 if dc > 0 else 23] * abs(dc)
                inter = []
                a, b = list(vs), list(hs)
                while a or b:
                    if a: inter.append(a.pop())
                    if b: inter.append(b.pop())
                routes = [vs + hs, hs + vs, inter]
                D = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
                chosen = None
                for route in routes:
                    pp = pos
                    pend = set()
                    states = []
                    good = True
                    for op in route:
                        pp = (pp[0] + D[op][0], pp[1] + D[op][1])
                        v = visible(bg, obj, pp)
                        if not ok_state(v, pend):
                            good = False
                            break
                        pend.add(v.tobytes())
                        states.append(v)
                    if good:
                        chosen = (route, states)
                        break
                if chosen is None:
                    chosen = (routes[0], [])
                    pp = pos
                    for op in routes[0]:
                        pp = (pp[0] + D[op][0], pp[1] + D[op][1])
                        chosen[1].append(visible(bg, obj, pp))
                for op, v in zip(*chosen):
                    if first_sel is not None:
                        push(op, first_sel, v)
                        first_sel = None
                    else:
                        push(op, sel_of([]), v)
                dest = {(ti + i, tj + j): val for (i, j), val in obj.items()
                        if 0 <= ti + i < h and 0 <= tj + j < w}
                # a re-grab inside the home box may have picked up part of an
                # already transplanted copy; give those cells back their colour
                lost = {}
                for q in sorted(regrab):
                    if q in placed and q not in dest and g[q] != placed[q]:
                        lost.setdefault(placed[q], []).append(q)
                for colr, qs in sorted(lost.items()):
                    ng = g.copy()
                    for q in qs:
                        ng[q] = colr
                    push(colr, sel_of(qs), ng)
                placed.update(dest)
        else:
            zeroed = set(home)
        # the template's home is now empty: lay the background there
        hole = sorted(q for q in zeroed if q not in placed and g[q] != bgc)
        if hole:
            ng = g.copy()
            for q in hole:
                ng[q] = bgc
            push(bgc, sel_of(hole), ng)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
