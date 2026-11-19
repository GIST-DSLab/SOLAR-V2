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
    # 2 and 3 are hardcoded key colours; bgc / rcol / gcol are sampled by the generator
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, rcol, gcol = random.sample(cols, 3)
    return {"bgc": bgc, "rcol": rcol, "gcol": gcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rcol=None, gcol=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    if bgc is None or rcol is None or gcol is None:
        bgc, rcol, gcol = random.sample(cols, 3)

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
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)]

    def dims(o):
        return max(r for r, _ in o) + 1, max(c for _, c in o) + 1

    H = max(11, min(30, int(max_h)))
    W = max(11, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (11, H))
    w = unifint(diff_lb, diff_ub, (11, W))

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

    wr = dims(robj)[1]
    robj_m = {(r, wr - 1 - c): v for (r, c), v in robj.items()}
    stamps = [robj_m, gobj]

    denom = max(1, int(1.5 * (len(robj) + len(gobj))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // denom)))
    succ, tr, maxtr = 0, 0, 5 * noccs

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

    def wipe(g, col):
        res = [row[:] for row in g]
        seen = set()
        for r in range(h):
            for c in range(w):
                if g[r][c] != bgc and (r, c) not in seen:
                    seen.add((r, c))
                    comp, stack = [(r, c)], [(r, c)]
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

    if unifint(diff_lb, diff_ub, (1, 100)) < 30:
        delc = random.choice((2, 3))
        gi2 = wipe(gi, delc)
        go2 = wipe(go, delc)
        if gi2 != go2:
            gi, go = gi2, go2

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O, examples=None):
    """Rule (read per marker, as the golf solution does): two 2-colour templates sit in the
    grid, one keyed by a 2 cell, one by a 3 cell.  Every lone 2 / 3 cell is a marker that
    receives a copy of its template anchored on the key; the 2-template is used in its
    left-right MIRRORED form.  The mirrored template is built ONCE (stamp + flip on the
    first marker) and that finished stamp is then copied as a unit onto every other marker."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 8-connected non-background components of I ---------------------------------
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

    templates, markers = {}, {2: [], 3: []}
    for comp in comps:
        vals = set(int(I[y, x]) for y, x in comp)
        if len(comp) >= 2 and len(vals) >= 2:
            for fc in (2, 3):
                if fc in vals:
                    templates[fc] = comp
        elif len(comp) == 1 and int(I[comp[0]]) in (2, 3):
            markers[int(I[comp[0]])].append(comp[0])

    ops, sels = [], []
    cur = I.copy()

    for fc in (2, 3):
        comp = templates.get(fc)
        mks = sorted(markers[fc])
        if comp is None or not mks:
            continue
        mirror = (fc == 2)
        tr = min(y for y, x in comp)
        tc = min(x for y, x in comp)
        tw = max(x for y, x in comp) - tc + 1
        ar, ac = [(y, x) for y, x in comp if int(I[y, x]) == fc][0]
        rel = {(y - ar, x - ac): int(I[y, x]) for y, x in comp}      # key at (0,0)
        if mirror:
            rel_use = {(dy, -dx): v for (dy, dx), v in rel.items()}
        else:
            rel_use = dict(rel)
        body = [d for d in rel_use if d != (0, 0)]
        bodycol = rel_use[body[0]]

        def stamp(m, offs):
            return [(m[0] + dy, m[1] + dx) for dy, dx in offs]

        if bodycol == 0:
            # 0 is invisible to Copy/Paste: paint each stamp's body as one object
            for m in mks:
                ops.append(0); sels.append(sel_of(stamp(m, body)))
                for p in stamp(m, body):
                    cur[p] = 0
            continue

        if not mirror or rel_use == rel:
            # template used as-is: copy its exact cells from the input, paste on each marker
            ops.append(28); sels.append(sel_of(comp))
            for m in mks:
                ops.append(30); sels.append([m[0] - (ar - tr), m[1] - (ac - tc), 0, 0])
                for (dy, dx), v in rel.items():
                    cur[m[0] + dy, m[1] + dx] = v
            continue

        # ---- mirrored template: build it once on the first marker -------------------
        m0 = mks[0]
        unm = {(m0[0] + dy, m0[1] + dx + (2 * (ac - tc) - (tw - 1))): v
               for (dy, dx), v in rel.items()}   # unmirrored copy inside the mirrored stamp's box
        clean = all(0 <= y < hi and 0 <= x < wi and (cur[y, x] == bgc or (y, x) == m0)
                    for (y, x) in unm)
        if clean:
            ops.append(28); sels.append(sel_of(comp))                       # copy template cells
            box_r = m0[0] - (ar - tr)
            box_c = m0[1] - (tw - 1 - (ac - tc))
            ops.append(30); sels.append([box_r, box_c, 0, 0])               # stamp it
            for p, v in unm.items():
                cur[p] = v
            ops.append(26); sels.append(sel_of(unm.keys()))                 # mirror it in place
            for p in unm:
                cur[p] = 0
            for p, v in zip(stamp(m0, rel_use.keys()), rel_use.values()):
                cur[p] = v
            hole = sorted(set(unm) - set(stamp(m0, rel_use.keys())))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))                  # 0s the flip left behind
                for p in hole:
                    cur[p] = bgc
        else:
            ops.append(bodycol); sels.append(sel_of(stamp(m0, body)))       # draw first stamp
            for p in stamp(m0, body):
                cur[p] = bodycol
        if len(mks) > 1:
            # the finished mirrored stamp is now the unit: copy it, paste on the rest
            cells0 = stamp(m0, rel_use.keys())
            ops.append(29); sels.append(sel_of(cells0))
            r0 = min(y for y, x in cells0)
            c0 = min(x for y, x in cells0)
            for m in mks[1:]:
                ops.append(30); sels.append([m[0] + r0 - m0[0], m[1] + c0 - m0[1], 0, 0])
                for p, v in zip(stamp(m, rel_use.keys()), rel_use.values()):
                    cur[p] = v

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
