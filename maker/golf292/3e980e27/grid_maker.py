"""
ARC Task: 3e980e27 (RE-ARC) — LLM-generated grid_maker
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
    # bgc / rcol / gcol are the three randomly sampled colors of the generator
    # (2 and 3 are hardcoded anchor colors -> not sampled).
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, rcol, gcol = random.sample(cols, 3)

    # discrete structural variant: the generator may wipe every object carrying
    # color 2 or color 3 (delc), or keep both families (delc == 0)
    variants = [{"delc": 0}, {"delc": 2}, {"delc": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "rcol": rcol, "gcol": gcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rcol=None, gcol=None,
             delc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    if bgc is None or rcol is None or gcol is None:
        bgc, rcol, gcol = random.sample(cols, 3)
    if delc is None:
        delc = random.choice((2, 3)) if random.randint(1, 100) < 30 else 0

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def nbrs(cell):
        r, c = cell
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)]

    def dims(o):
        return max(r for r, _ in o) + 1, max(c for _, c in o) + 1

    H = max(11, min(30, int(max_h)))
    W = max(11, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (11, H))
    w = unifint(diff_lb, diff_ub, (11, W))

    # ---- build the two template objects (one anchored on 2, one on 3) ----
    objs = []
    for fixc, remc in ((2, rcol), (3, gcol)):
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        bset = set(bounds)
        obj = {random.choice(bounds)}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for _ in range(ncells - 1):
            cands = set()
            for cell in obj:
                for nb in nbrs(cell):
                    if nb in bset and nb not in obj:
                        cands.add(nb)
            if not cands:
                break
            obj.add(random.choice(sorted(cands)))
        minr = min(r for r, _ in obj)
        minc = min(c for _, c in obj)
        obj = {(r - minr, c - minc) for r, c in obj}
        fixp = random.choice(sorted(obj))
        cellmap = {fixp: fixc}
        for cell in obj:
            if cell != fixp:
                cellmap[cell] = remc
        objs.append(cellmap)

    robj, gobj = objs
    order = [0, 1]
    random.shuffle(order)
    obj1, obj2 = objs[order[0]], objs[order[1]]
    h1, w1 = dims(obj1)
    h2, w2 = dims(obj2)

    loci1 = random.randint(0, h - h1 - h2 - 1)
    locj1 = random.randint(0, w - w1)
    loci2 = random.randint(loci1 + h1 + 1, h - h2)
    locj2 = random.randint(0, w - w2)

    gi = [[bgc] * w for _ in range(h)]
    placed = set()
    for o, (dr, dc) in ((obj1, (loci1, locj1)), (obj2, (loci2, locj2))):
        for (r, c), v in o.items():
            gi[r + dr][c + dc] = v
            placed.add((r + dr, c + dc))

    excl = set(placed)
    for cell in placed:
        excl.update(nbrs(cell))
    inds = {(r, c) for r in range(h) for c in range(w) if (r, c) not in excl}

    go = [row[:] for row in gi]

    # scattered stamps: red is mirrored left<->right, green as-is
    wr = dims(robj)[1]
    robj_m = {(r, wr - 1 - c): v for (r, c), v in robj.items()}
    stamps = [robj_m, gobj]

    denom = max(1, int(1.5 * (len(robj) + len(gobj))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // denom)))
    succ = 0
    tr = 0
    maxtr = 5 * noccs

    def try_place(o, loc):
        oh, ow = dims(o)
        if loc[0] > h - oh or loc[1] > w - ow:
            return False
        plcd = {(r + loc[0], c + loc[1]): v for (r, c), v in o.items()}
        pi = set(plcd)
        if not pi <= inds:
            return False
        newexcl = set(pi)
        for cell in pi:
            newexcl.update(nbrs(cell))
        inds.difference_update(newexcl)
        for (r, c), v in plcd.items():
            if v in (2, 3):
                gi[r][c] = v
            go[r][c] = v
        return True

    while tr < maxtr and succ < noccs:
        tr += 1
        o = random.choice(stamps)
        oh, ow = dims(o)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        if try_place(o, random.choice(sorted(cands))):
            succ += 1

    if succ == 0:
        for o in stamps:
            for loc in sorted(inds):
                if try_place(o, loc):
                    succ += 1
                    break
            if succ:
                break

    # optional wipe of every object carrying color delc
    def wipe(g, col):
        res = [row[:] for row in g]
        seen = set()
        for r in range(h):
            for c in range(w):
                if g[r][c] != bgc and (r, c) not in seen:
                    seen.add((r, c))
                    comp = [(r, c)]
                    stack = [(r, c)]
                    while stack:
                        cur = stack.pop()
                        for nb in nbrs(cur):
                            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in seen \
                                    and g[nb[0]][nb[1]] != bgc:
                                seen.add(nb)
                                comp.append(nb)
                                stack.append(nb)
                    if any(g[a][b] == col for a, b in comp):
                        for a, b in comp:
                            res[a][b] = bgc
        return res

    if delc in (2, 3):
        gi2 = wipe(gi, delc)
        go2 = wipe(go, delc)
        if gi2 != go2:
            gi, go = gi2, go2

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O, examples=None):
    """Rule: two multi-coloured template shapes sit in the grid, one keyed by a 2 cell,
    one keyed by a 3 cell.  Every other lone 2 / 3 cell is a marker: stamp the matching
    template on it, anchored at its key cell; the 2-template is stamped left-right mirrored.
    Route: copy the 3-template once and paste it on every 3-marker; paste the 2-template
    once, mirror that stamp in place, then copy the mirrored stamp and paste it on every
    other 2-marker.  O is never read."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the generator paints the canvas with bgc first, and the scattered
    # markers are single cells, so bgc is by far the majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 8-connected non-background components of I ------------------------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack, comp = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] \
                                and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(comp)

    templates = {}
    for comp in comps:
        vals = set(int(I[y, x]) for y, x in comp)
        if len(vals) < 2:
            continue
        for fc in (2, 3):
            if fc in vals:
                templates[fc] = comp
    tcells = set()
    for comp in templates.values():
        tcells |= set(comp)
    markers = {2: [], 3: []}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v in (2, 3) and (r, c) not in tcells:
                markers[v].append((r, c))

    # --- tiny ARCLE simulator for the ops used here (masks: set of cells) --------
    def sim(grid, ops, sels):
        g = grid.copy()
        clip = {}
        for op, s in zip(ops, sels):
            if op <= 9:
                for (y, x) in s:
                    g[y, x] = op
            elif op == 26:  # bbox flip, s = (r, c, h, w) full rectangle
                r, c, h, w = s
                g[r:r + h, c:c + w] = np.fliplr(g[r:r + h, c:c + w])
            elif op in (28, 29):
                src = I if op == 28 else g
                r0 = min(y for y, _ in s)
                c0 = min(x for _, x in s)
                clip = {(y - r0, x - c0): int(src[y, x]) for (y, x) in s if src[y, x] != 0}
            elif op == 30:
                r, c = s
                for (dy, dx), v in clip.items():
                    if 0 <= r + dy < hi and 0 <= c + dx < wi:
                        g[r + dy, c + dx] = v
        return g

    def emit(ops_l, sels_l):
        out = []
        for op, s in zip(ops_l, sels_l):
            if op == 26:
                r, c, h, w = s
                # whole landing rectangle: the stamp mirrored together with the bg around it
                out.append([r, c, h - 1, w - 1])
            elif op == 30:
                out.append([s[0], s[1], 0, 0])
            else:
                out.append(sel_of(s))
        return out

    ops, sels = [], []
    cur = I.copy()
    for fc in (3, 2):
        comp = templates.get(fc)
        if comp is None or not markers[fc]:
            continue
        mirror = (fc == 2)
        tr = min(y for y, _ in comp)
        tc = min(x for _, x in comp)
        th = max(y for y, _ in comp) - tr + 1
        tw = max(x for _, x in comp) - tc + 1
        ar, ac = [(y, x) for y, x in comp if int(I[y, x]) == fc][0]
        lr, lc = ar - tr, ac - tc
        body = [(y - ar, x - ac) for y, x in comp if int(I[y, x]) != fc]
        bodycol = int(I[body[0][0] + ar, body[0][1] + ac])

        def stamp_cells(m, with_key=False):
            mr, mc = m
            cells = [(mr + dy, mc - dx if mirror else mc + dx) for dy, dx in body]
            if with_key:
                cells.append((mr, mc))
            return cells

        def origin(m):
            mr, mc = m
            return (mr - lr, mc - (tw - 1 - lc) if mirror else mc - lc)

        # rule-derived expectation (from I and the template only) for self-checking
        target = cur.copy()
        for m in markers[fc]:
            for (y, x) in stamp_cells(m):
                target[y, x] = bodycol

        chosen = None
        if bodycol != 0:
            if not mirror:
                c_ops = [28] + [30] * len(markers[fc])
                c_sels = [list(comp)] + [origin(m) for m in markers[fc]]
                if np.array_equal(sim(cur, c_ops, c_sels), target):
                    chosen = (c_ops, c_sels)
            else:
                for first in markers[fc]:
                    rest = [m for m in markers[fc] if m != first]
                    r0, c0 = origin(first)
                    c_ops = [28, 30]
                    c_sels = [list(comp), (r0, c0)]
                    box = sim(cur, c_ops, c_sels)[r0:r0 + th, c0:c0 + tw]
                    if not np.array_equal(box, np.fliplr(box)):
                        # mirror the freshly pasted stamp in place (skip if symmetric)
                        c_ops.append(26)
                        c_sels.append((r0, c0, th, tw))
                    if rest:
                        c_ops.append(29)
                        c_sels.append(stamp_cells(first, with_key=True))
                        for m in rest:
                            c_ops.append(30)
                            c_sels.append(origin(m))
                    if np.array_equal(sim(cur, c_ops, c_sels), target):
                        chosen = (c_ops, c_sels)
                        break
        if chosen is None:
            # body colour 0 (Copy/Paste cannot carry it) or a landing box collides:
            # paint each stamp's shape, one whole stamp per op
            c_ops = [bodycol] * len(markers[fc])
            c_sels = [stamp_cells(m) for m in markers[fc]]
            chosen = (c_ops, c_sels)
        ops += chosen[0]
        sels += emit(*chosen)
        cur = sim(cur, *chosen)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
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
                                f"for task 3e980e27"
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
                    f"Failed to build a complete episode for task 3e980e27 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3e980e27-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
