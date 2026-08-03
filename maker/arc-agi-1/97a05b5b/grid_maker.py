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
def sample_colors(num_examples=None) -> dict:
    # bgc/sqc are the two colours `sample(cols, 2)` draws in the generator, so both
    # must be pinned for the whole episode.  bgc is pinned to 0 specifically: that
    # keeps every square/marker cell non-zero, which is what makes CopyI/Paste
    # (which treat 0 as "nothing") able to carry a key across the grid intact.
    # The per-key marker colours stay free — the rule matches shapes, not colours.
    bgc = 0
    sqc = random.choice([c for c in range(1, 10)])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(15, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(15, max_w), max_w))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh // 2)
    ow = randint(2, sgw // 2)
    nobjs = unifint(diff_lb, diff_ub, (1, 8))
    objs = set()
    cands = asindices(canvas(-1, (oh, ow)))
    forbidden = set()
    tr = 0
    maxtr = 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {choice(totuple(cands))}
        ncells = randint(1, oh * ow - 1)
        for k in range(ncells - 1):
            obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
        obj |= choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
        objn = normalize(obj)
        if objn not in forbidden:
            objs.add(objn)
        for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                forbidden.add(augmf1(augmf2(objn)))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
    gi = fill(gi, sqc, bd)
    go = canvas(sqc, (sgh, sgw))
    goinds = asindices(go)
    giinds = asindices(gi) - shift(goinds, (loci, locj))
    giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f = compose(f1, f2)
        subgo = f(subgi)
        giobj = asobject(subgi)
        goobj = asobject(subgo)
        ohi, owi = shape(giobj)
        oho, owo = shape(goobj)
        gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
        if len(gocands) == 0:
            continue
        goloc = choice(totuple(gocands))
        goplcd = shift(goobj, goloc)
        goplcdi = toindices(goplcd)
        if goplcdi.issubset(goinds):
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
            if len(gicands) == 0:
                continue
            giloc = choice(totuple(gicands))
            giplcd = shift(giobj, giloc)
            giplcdi = toindices(giplcd)
            if giplcdi.issubset(giinds):
                succ += 1
                remcols = remove(col, remcols)
                objs = remove(obj, objs)
                goinds = goinds - goplcdi
                giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                gi = paint(gi, giplcd)
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # I holds one big square with notches punched out of it, plus a scatter of
    # small solid two-colour rectangles ("keys").  Each key's marker cells mark
    # where the square is still solid and its square-coloured cells mark the
    # notch, so every key fits exactly one notch under one of the 8 symmetries.
    # O = the square with each key dropped, turned, into its own notch.
    # So: copy each key out of I, paste it onto the notch, turn it there, crop.
    from collections import Counter, deque

    I = np.asarray(I, dtype=int); O = np.asarray(O, dtype=int)
    hi, wi = I.shape; ho, wo = O.shape

    def components(bg):
        seen = np.zeros((hi, wi), bool); out = []
        for r in range(hi):
            for c in range(wi):
                if seen[r, c] or I[r, c] == bg: continue
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    a, b = q.popleft(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] != bg:
                            seen[na, nb] = True; q.append((na, nb))
                out.append(cells)
        return out

    TRANSFORMS = [
        ("id",     lambda a: a,                         []),
        ("flipud", lambda a: np.flipud(a),              [27]),
        ("fliplr", lambda a: np.fliplr(a),              [26]),
        ("rot180", lambda a: np.rot90(a, 2),            [26, 27]),
        ("ccw",    lambda a: np.rot90(a, 1),            [24]),
        ("cw",     lambda a: np.rot90(a, 3),            [25]),
        ("dmir",   lambda a: np.flipud(np.rot90(a, 1)), [24, 27]),
        ("cmir",   lambda a: np.fliplr(np.rot90(a, 1)), [24, 26]),
    ]
    FLIPPY = ("id", "flipud", "fliplr", "rot180")

    def rect(r, c, h, w): return (r, c, r + h, c + w)
    def hits(a, b): return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    def cellsof(x): return {(r, c) for r in range(x[0], x[2]) for c in range(x[1], x[3])}

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Keys are the solid two-colour rectangles; everything else is square.
    keys, rest = [], []
    for cells in components(bgc):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        pal = {I[a, b] for a, b in cells}
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(pal) == 2:
            keys.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        else:
            rest.append((r0, r1, c0, c1, I[cells[0][0], cells[0][1]]))
    sqc = rest[0][4]
    r0 = min(x[0] for x in rest); r1 = max(x[1] for x in rest)
    c0 = min(x[2] for x in rest); c1 = max(x[3] for x in rest)

    def assign(cands, need):
        order = sorted(range(len(cands)), key=lambda i: len(cands[i]))
        chosen = [None] * len(cands)
        def bt(k, used, cov):
            if k == len(order): return cov.issuperset(need)
            i = order[k]
            for cand in cands[i]:
                rc = rect(cand[2], cand[3], cand[4], cand[5])
                if any(hits(rc, u) for u in used): continue
                chosen[i] = cand
                if bt(k + 1, used + [rc], cov | cellsof(rc)): return True
            return False
        return chosen if bt(0, [], set()) else None

    # Place the square (its shape is O's) and solve every key -> notch match at
    # once: the picks must tile disjointly and together account for every cell
    # where the square and O disagree.  Matching keys one at a time is ambiguous.
    picks = board = None
    for br0 in range(max(0, r1 - ho + 1), min(r0, hi - ho) + 1):
        for bc0 in range(max(0, c1 - wo + 1), min(c0, wi - wo) + 1):
            bd = I[br0:br0 + ho, bc0:bc0 + wo]
            if not np.isin(bd, [sqc, bgc]).all(): continue
            if ((bd == bgc) & (O != sqc)).any(): continue
            need = {(r, c) for r in range(ho) for c in range(wo) if bd[r, c] != O[r, c]}
            cands = []
            for (sr, sc, h, w) in keys:
                sub = I[sr:sr + h, sc:sc + w]
                pat = np.where(sub == sqc, bgc, sqc)   # the notch this key punched
                lst = []
                for name, fn, fops in TRANSFORMS:
                    pf, cf = fn(pat), fn(sub)
                    ph, pw = pf.shape
                    for r in range(ho - ph + 1):
                        for c in range(wo - pw + 1):
                            if np.array_equal(bd[r:r + ph, c:c + pw], pf) and \
                               np.array_equal(O[r:r + ph, c:c + pw], cf):
                                lst.append((name, fops, r, c, ph, pw))
                cands.append(lst)
            picks = assign(cands, need)
            if picks is not None:
                board = (br0, bc0); break
        if picks is not None: break
    br0, bc0 = board
    board_rect = rect(br0, bc0, ho, wo)
    dests = [rect(br0 + p[2], bc0 + p[3], p[4], p[5]) for p in picks]

    # ARCLE turns a hs*ws selection about its centre: the object comes back as
    # ws*hs anchored at (R+(hs-ws)//2, C+(ws-hs)//2).  So a turn needs elbow room
    # and must not reach into the square.
    def plan_inplace(src, k, keep):
        sr, sc, h, w = src
        own = rect(sr, sc, h, w)
        sizes = sorted((hs * ws, abs(hs - ws), hs, ws)
                       for hs in range(h, h + 10) for ws in range(w, w + 10))
        for _, __, hs, ws in sizes:
            for R in range(sr - (hs - h), sr + 1):
                for C in range(sc - (ws - w), sc + 1):
                    if R < 0 or C < 0 or R + hs > hi or C + ws > wi: continue
                    free = nz[R:R + hs, C:C + ws].copy()
                    free[sr - R:sr - R + h, sc - C:sc - C + w] = False
                    if free.any(): continue
                    R2, C2 = R + (hs - ws) // 2, C + (ws - hs) // 2
                    if R2 < 0 or C2 < 0 or R2 + ws > hi or C2 + hs > wi: continue
                    a0, b0 = sr - R, sc - C
                    rr, cc = (b0, hs - a0 - h) if k == 3 else (ws - b0 - w, a0)
                    old, new = rect(R, C, hs, ws), rect(R2 + rr, C2 + cc, w, h)
                    if any(hits(x, b) for x in (old, new) for b in keep if b != own): continue
                    return R, C, hs, ws, R2 + rr, C2 + cc, [old, new]
        return None

    def plan_scratch(src, protect):
        sr, sc, h, w = src
        for R in range(hi - h + 1):
            for C in range(wi - w + 1):
                R2, C2 = R + (h - w) // 2, C + (w - h) // 2
                if R2 < 0 or C2 < 0 or R2 + w > hi or C2 + h > wi: continue
                old, new = rect(R, C, h, w), rect(R2, C2, w, h)
                if any(hits(x, b) for x in (old, new) for b in protect): continue
                return R, C, R2, C2, [old, new]
        return None

    nz = (I != bgc)
    rot = [i for i, p in enumerate(picks) if p[0] not in FLIPPY]
    # Last resort for a key too big to be turned anywhere off the square: turn it
    # on its own notch, landing it from a pad offset by the turn's own shift.
    pads = {}
    for i in rot:
        sr, sc, h, w = keys[i]
        if plan_scratch(keys[i], [board_rect]) is None:
            pads[i] = rect(br0 + picks[i][2] - (h - w) // 2,
                           bc0 + picks[i][3] - (w - h) // 2, h, w)
    keep = [rect(*k) for k in keys] + [board_rect] + list(pads.values())
    plan = {}
    for i in rot:
        if i in pads: continue
        got = plan_inplace(keys[i], 3 if picks[i][1][0] == 25 else 1, keep)
        if got is not None:
            keep.extend(got[6]); plan[i] = got

    ops, sels = [], []
    order = sorted(range(len(keys)), key=lambda j: j not in pads)
    for pos, i in enumerate(order):
        sr, sc, h, w = keys[i]
        name, fops, dr, dc, ph, pw = picks[i]
        dr += br0; dc += bc0
        if name in FLIPPY:
            # Key keeps its footprint: copy it, drop it on the notch, mirror it
            # there.  The paste fills the whole box, so the mirror catches
            # nothing but the key itself.
            ops.append(28); sels.append([sr, sc, h - 1, w - 1])
            ops.append(30); sels.append([dr, dc, 0, 0])
            for o in fops:
                ops.append(o); sels.append([dr, dc, h - 1, w - 1])
            continue
        if i in plan:                       # turn it right where it lies
            R, C, hs, ws, nr, nc, _ = plan[i]
        else:
            if i in pads:
                R, C = pads[i][0], pads[i][1]
            else:                           # turn it on spare canvas
                later = [board_rect] + [plan[j][6][0] for j in rot if j in plan and j > i]
                R, C, _nr, _nc, _rc = plan_scratch(keys[i], later)
            hs, ws = h, w
            ops.append(28); sels.append([sr, sc, h - 1, w - 1])
            ops.append(30); sels.append([R, C, 0, 0])
            nr, nc = (dr, dc) if i in pads else (_nr, _nc)
        ops.append(fops[0]); sels.append([R, C, hs - 1, ws - 1])
        for o in fops[1:]:
            ops.append(o); sels.append([nr, nc, w - 1, h - 1])
        if (nr, nc) != (dr, dc):
            ops.append(29); sels.append([nr, nc, w - 1, h - 1])
            ops.append(30); sels.append([dr, dc, 0, 0])
        else:
            # The turn emptied the pad cells the key no longer covers; restore
            # only those the square still needs and nothing later rewrites.
            done = set()
            for j in order[pos + 1:]:
                done |= cellsof(dests[j])
                if j in pads: done |= cellsof(pads[j])
            gone = sorted(((cellsof(pads[i]) - cellsof(dests[i])) & cellsof(board_rect)) - done)
            for (r, c) in gone:
                ops.append(int(O[r - br0, c - bc0])); sels.append([r, c, 0, 0])
    ops.append(33); sels.append([br0, bc0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
