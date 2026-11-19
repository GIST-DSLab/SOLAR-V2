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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # the generator samples exactly two structural colours: the canvas background
    # and the colour of the big framed rectangle.  Per-object colours stay random
    # (the rule matches objects by shape, not by colour).
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int) -> dict:
    cols = interval(0, 10, 1)
    hlo = min(15, max_h)
    wlo = min(15, max_w)
    last = None
    for _attempt in range(40):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        sgh = randint(h // 3, h // 3 * 2)
        sgw = randint(w // 3, w // 3 * 2)
        if sgh < 4 or sgw < 4 or sgh > h or sgw > w:
            continue
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
                rem = totuple((cands - obj) & mapply(neighbors, obj))
                if len(rem) == 0:
                    break
                obj.add(choice(rem))
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
                    gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)),
                                               lambda cij: cij[0] == sqc))
                    go = paint(go, goplcd)
        last = {'input': gi, 'output': go}
        if succ > 0:
            return last
    return last


def derive_operations(I, O):
    """I holds one big rectangle of a 'frame' colour sqc with bgc-coloured holes
    punched into it, plus small solid two-colour rectangles lying outside it.
    Each small rectangle is a block of colour `col` whose sqc-cells form a shape;
    that shape, under one of the 8 dihedral transforms, IS one of the holes.
    Output = the framed rectangle with every block stamped, in the orientation
    the hole demands, onto the hole it belongs to.
    Everything below is measured from I alone; O is never inspected."""
    I = np.asarray(I, dtype=int)
    hI, wI = I.shape

    TFS = [
        lambda a: a.copy(),                  # identity
        lambda a: np.rot90(a, 2),            # rot180
        lambda a: np.rot90(a, 3),            # rot90  (CW)
        lambda a: np.rot90(a, 1),            # rot270 (CCW)
        lambda a: a[::-1, :].copy(),         # hmirror
        lambda a: a[:, ::-1].copy(),         # vmirror
        lambda a: np.rot90(a, 2).T.copy(),   # cmirror
        lambda a: a.T.copy(),                # dmirror
    ]

    def components(bgc):
        seen = np.zeros((hI, wI), dtype=bool)
        comps = []
        for r in range(hI):
            for c in range(wI):
                if I[r, c] != bgc and not seen[r, c]:
                    stack = [(r, c)]; seen[r, c] = True; cells = []
                    while stack:
                        y, x = stack.pop(); cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hI and 0 <= nx < wI and not seen[ny, nx] \
                                    and I[ny, nx] != bgc:
                                seen[ny, nx] = True; stack.append((ny, nx))
                    comps.append(cells)
        return comps

    def bbox(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def analyze(bgc):
        """Split I (given a candidate background) into solid 2-colour blocks and
        the frame; reject the candidate if that reading is not self-consistent."""
        comps = components(bgc)
        if not comps:
            return None
        patches, others = [], []
        for cells in comps:
            r0, c0, bh, bw = bbox(cells)
            cs = {int(I[r, c]) for r, c in cells}
            if len(cells) == bh * bw and len(cs) == 2:
                patches.append((r0, c0, bh, bw))
            else:
                others.append(cells)
        if not others:
            return None
        ocols = set()
        for cells in others:
            for r, c in cells:
                ocols.add(int(I[r, c]))
        if len(ocols) != 1:            # the frame must be one single colour
            return None
        sqc = ocols.pop()
        allo = [p for cells in others for p in cells]
        r0, c0, ho, wo = bbox(allo)
        sub = I[r0:r0 + ho, c0:c0 + wo]
        if not np.all((sub == sqc) | (sub == bgc)):
            return None
        for (pr, pc, ph, pw) in patches:
            if not np.any(I[pr:pr + ph, pc:pc + pw] == sqc):
                return None            # every block carries frame-coloured cells
        return patches, sqc, (r0, c0, ho, wo)

    best = None
    for cand in sorted(set(I.flatten().tolist())):
        res = analyze(cand)
        if res is None:
            continue
        patches, sqc, (r0, c0, ho, wo) = res
        score = (len(patches), -(ho * wo))
        if best is None or score > best[0]:
            best = (score, cand, res)
    if best is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        sqc = bgc
        patches = []
        r0, c0, ho, wo = 0, 0, hI, wI
    else:
        _, bgc, (patches, sqc, (r0, c0, ho, wo)) = best

    crop = I[r0:r0 + ho, c0:c0 + wo]
    all_holes = frozenset((int(r), int(c)) for r, c in zip(*np.where(crop == bgc)))

    # every orientation/position at which a block's hole-print fits the frame
    pgs, cols, cands = [], [], []
    for (pr, pc, ph, pw) in patches:
        pg = I[pr:pr + ph, pc:pc + pw]
        rest = {int(v) for v in np.unique(pg)} - {sqc}
        if len(rest) != 1:
            continue
        col = rest.pop()
        hgrid = np.where(pg == sqc, bgc, sqc)   # how this block prints as a hole
        seen, opts = set(), []
        for ti, T in enumerate(TFS):
            tg = T(hgrid); th, tw = tg.shape
            key0 = tg.tobytes() + bytes([th, tw])
            for r in range(ho - th + 1):
                for c in range(wo - tw + 1):
                    if (key0, r, c) in seen:
                        continue
                    if np.array_equal(crop[r:r + th, c:c + tw], tg):
                        seen.add((key0, r, c))
                        box = frozenset((r + i, c + j)
                                        for i in range(th) for j in range(tw))
                        holes = frozenset((r + i, c + j)
                                          for i in range(th) for j in range(tw)
                                          if tg[i, j] == bgc)
                        opts.append((ti, r, c, box, holes))
        pgs.append(pg); cols.append(col); cands.append(opts)

    # the blocks must account for every hole exactly once and never overlap
    order = sorted(range(len(cands)), key=lambda i: len(cands[i]))
    sol = {}
    budget = [200000]

    def rec(k, used, covered):
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        if k == len(order):
            return covered == all_holes
        i = order[k]
        for (ti, r, c, box, holes) in cands[i]:
            if box & used:
                continue
            sol[i] = (ti, r, c)
            if rec(k + 1, used | box, covered | holes):
                return True
            del sol[i]
        return False

    placements = []
    if rec(0, frozenset(), frozenset()):
        for i, (ti, r, c) in sol.items():
            placements.append((r, c, TFS[ti](pgs[i]), cols[i]))
    else:                                   # fallback: greedy non-overlapping fit
        used = set()
        for i in range(len(cands)):
            for (ti, r, c, box, holes) in cands[i]:
                if box & used:
                    continue
                used |= set(box)
                placements.append((r, c, TFS[ti](pgs[i]), cols[i]))
                break
    placements.sort(key=lambda p: (p[0], p[1]))

    ops, sels = [], []
    # 1. keep only the framed rectangle (selection IS exactly that full rectangle)
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
    # 2. lay the frame colour down as the base, covering the holes
    if bool(np.any(crop != sqc)):
        ops.append(int(sqc)); sels.append([0, 0, ho - 1, wo - 1])
    # 3. stamp each block, oriented as its hole demanded, over that hole
    for (r, c, tp, col) in placements:
        cells = [(r + i, c + j)
                 for i in range(tp.shape[0]) for j in range(tp.shape[1])
                 if int(tp[i, j]) == col]
        if cells:
            ops.append(int(col)); sels.append(sel_of(cells))
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
