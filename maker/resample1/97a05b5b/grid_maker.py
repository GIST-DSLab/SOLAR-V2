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
from dsl import *
from random import choice, randint, sample, shuffle, uniform
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + round((b - a) * diff_lb)
    hi = a + round((b - a) * diff_ub)
    return max(a, min(b, random.randint(min(lo, hi), max(lo, hi))))


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(range(10), 2)
    res = None
    for _ in range(50):
        res = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, sqc)
        if _well_posed(res):
            return res
    return res


def _well_posed(res):
    """Holes must admit exactly one way of being filled by the pieces, and
    that filling must be the generator's output."""
    try:
        I = np.array(res["input"], dtype=int)
        an = _analyse(I)
        sols = _covers(an, limit=2)
        if len(sols) != 1:
            return False
        reg = an["region"].copy()
        for B, _, (r, c), _ in sols[0].values():
            reg[r:r + B.shape[0], c:c + B.shape[1]] = B
        return np.array_equal(reg, np.array(res["output"], dtype=int))
    except Exception:
        return False


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, sqc):
    cols = interval(0, 10, 1)
    h = _unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = _unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh // 2)
    ow = randint(2, sgw // 2)
    nobjs = _unifint(diff_lb, diff_ub, (1, 8))
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
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


# ---------------------------------------------------------------- ARCLE mirror
class _Sim:
    """Minimal mirror of ARCLE's object-mode ops, used to plan positions."""

    def __init__(self, grid):
        self.g = np.array(grid, dtype=int)
        self.active = False
        self.obj = None
        self.osel = None
        self.bg = None
        self.pos = (0, 0)
        self.par = 0

    def _grab(self, cells):
        if cells:
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            x0, x1, y0, y1 = min(rs), max(rs), min(cs), max(cs)
            h, w = x1 - x0 + 1, y1 - y0 + 1
            self.obj = np.zeros((h, w), dtype=int)
            self.osel = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                self.obj[r - x0, c - y0] = self.g[r, c]
                self.osel[r - x0, c - y0] = True
            self.bg = self.g.copy()
            for r, c in cells:
                self.bg[r, c] = 0
            self.pos = (x0, y0)
            self.active = True
            self.par = 0
            return x0, x1, y0, y1
        x, y = self.pos
        h, w = self.obj.shape
        return x, x + h - 1, y, y + w - 1

    def _patch(self):
        self.g = self.bg.copy()
        gh, gw = self.g.shape
        x, y = self.pos
        h, w = self.obj.shape
        for i in range(h):
            for j in range(w):
                r, c = x + i, y + j
                if 0 <= r < gh and 0 <= c < gw and self.obj[i, j] > 0:
                    self.g[r, c] = self.obj[i, j]

    def move(self, cells, d):
        self._grab(cells)
        dx, dy = [(-1, 0), (1, 0), (0, 1), (0, -1)][d]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        self._patch()

    def flip(self, cells, axis):
        self._grab(cells)
        f = np.fliplr if axis == "H" else np.flipud
        self.obj = f(self.obj).copy()
        self.osel = f(self.osel).copy()
        self._patch()

    def rotate(self, cells, k):
        xmin, xmax, ymin, ymax = self._grab(cells)
        h, w = self.obj.shape
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        x, y = self.pos
        if h % 2 == w % 2:
            self.pos = (int(np.floor(cx - cy + y)), int(np.floor(cy - cx + x)))
        else:
            self.par = (self.par + k) % 2
            sig = (k + 2) % 4 - 2
            mod = 1 - self.par
            mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
            my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
            self.pos = (int(np.floor(mx)), int(np.floor(my)))
        self.obj = np.rot90(self.obj, k=k).copy()
        self.osel = np.rot90(self.osel, k=k).copy()
        self._patch()

    def op(self, op, cells):
        if op in (20, 21, 22, 23):
            self.move(cells, op - 20)
        elif op == 26:
            self.flip(cells, "H")
        elif op == 27:
            self.flip(cells, "V")
        elif op == 24:
            self.rotate(cells, 1)
        elif op == 25:
            self.rotate(cells, 3)
        elif op <= 9:
            for r, c in cells:
                self.g[r, c] = op
            self.active = False


# piece transforms, in order of preference, with the ARCLE ops realising them
_TRANSFORMS = [
    (lambda a: a, []),
    (np.fliplr, [26]),
    (np.flipud, [27]),
    (lambda a: np.flipud(np.fliplr(a)), [26, 27]),
    (lambda a: np.rot90(a, 3), [25]),
    (lambda a: np.rot90(a, 1), [24]),
    (lambda a: a.T, [25, 26]),
    (lambda a: np.flipud(np.rot90(a, 3)), [25, 27]),
]


def _components(I, bgc):
    hi, wi = I.shape
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] != bgc:
                        seen[na, nb] = True
                        stack.append((na, nb))
            comps.append(cells)
    return comps


def _analyse(I):
    """Find background, the holed region and the two-colour pieces, and list
    every placement (transformed piece + location) that fits a hole."""
    I = np.asarray(I, dtype=int)
    # background: the colour the canvas is painted with (majority colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pieces, region_cells = [], []
    for cells in _components(I, bgc):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        ph, pw = max(rs) - r0 + 1, max(cs) - c0 + 1
        ncol = len({int(I[r, c]) for r, c in cells})
        if len(cells) == ph * pw and ncol == 2:
            pieces.append((r0, c0, I[r0:r0 + ph, c0:c0 + pw].copy()))
        else:
            region_cells += cells
    rr = [r for r, _ in region_cells]
    cc = [c for _, c in region_cells]
    R0, C0 = min(rr), min(cc)
    RH, RW = max(rr) - R0 + 1, max(cc) - C0 + 1
    sqc = Counter(int(I[r, c]) for r, c in region_cells).most_common(1)[0][0]
    region = I[R0:R0 + RH, C0:C0 + RW]
    holes = frozenset((r, c) for r in range(RH) for c in range(RW) if region[r, c] == bgc)
    cand_lists = []
    for (pr, pc, A) in pieces:
        cl, seen = [], set()
        for f, tops in _TRANSFORMS:
            B = f(A)
            pat = np.where(B == sqc, bgc, sqc)
            ph, pw = pat.shape
            for r in range(RH - ph + 1):
                for c in range(RW - pw + 1):
                    if np.array_equal(region[r:r + ph, c:c + pw], pat):
                        key = (B.tobytes(), B.shape, r, c)
                        if key in seen:
                            continue
                        seen.add(key)
                        cov = frozenset((r + i, c + j) for i in range(ph) for j in range(pw)
                                        if B[i, j] == sqc)
                        cl.append((B, tops, (r, c), cov))
        cand_lists.append(cl)
    return dict(bgc=bgc, sqc=sqc, pieces=pieces, R0=R0, C0=C0, RH=RH, RW=RW,
                region=region, holes=holes, cands=cand_lists)


def _covers(an, limit=2):
    """Exact covers of the holes by the pieces (each hole filled exactly once)."""
    cands, holes = an["cands"], an["holes"]
    order = sorted(range(len(cands)), key=lambda k: len(cands[k]))
    out, assign = [], {}

    def solve(idx, used):
        if len(out) >= limit:
            return
        if idx == len(order):
            if used == holes:
                out.append(dict(assign))
            return
        k = order[idx]
        for cand in cands[k]:
            if cand[3] & used:
                continue
            assign[k] = cand
            solve(idx + 1, used | cand[3])
        assign.pop(k, None)

    solve(0, frozenset())
    return out


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    an = _analyse(I)
    pieces = an["pieces"]
    R0, C0, RH, RW = an["R0"], an["C0"], an["RH"], an["RW"]
    sols = _covers(an, limit=1)
    if sols:
        assign = sols[0]
    else:
        assign = {k: cl[0] for k, cl in enumerate(an["cands"]) if cl}

    sim = _Sim(I)
    ops, sels = [], []

    for pi, (pr, pc, A) in enumerate(pieces):
        if pi not in assign:
            continue
        B, tops, loc, _ = assign[pi]
        dr0, dc0 = R0 + loc[0], C0 + loc[1]
        src = [(pr + i, pc + j) for i in range(A.shape[0]) for j in range(A.shape[1])]

        # transform in place first (grab once), then slide with empty selections
        x, y = (pr, pc)
        if tops:
            s0 = _Sim(sim.g)
            for k, op in enumerate(tops):
                s0.op(op, src if k == 0 else [])
            x, y = s0.pos
        dv, dh = dr0 - x, dc0 - y
        vs = [21 if dv > 0 else 20] * abs(dv)
        hs = [22 if dh > 0 else 23] * abs(dh)
        inter = []
        vv, hh = list(vs), list(hs)
        while vv or hh:
            if vv:
                inter.append(vv.pop())
            if hh:
                inter.append(hh.pop())
        best = None
        for mseq in (vs + hs, hs + vs, inter):
            allops = list(tops) + mseq
            t = _Sim(sim.g)
            states = {t.g.tobytes()}
            ok = True
            for k, op in enumerate(allops):
                t.op(op, src if k == 0 else [])
                key = t.g.tobytes()
                if key in states:          # no visible change / revisited state
                    ok = False
                    break
                states.add(key)
            if ok:
                best = (allops, t)
                break
        if best is not None:
            allops, sim = best
            for k, op in enumerate(allops):
                ops.append(op)
                sels.append(sel_of(src if k == 0 else []))
            # ARCLE never draws colour-0 object cells: paint them explicitly
            zero = [(dr0 + i, dc0 + j) for i in range(B.shape[0]) for j in range(B.shape[1])
                    if B[i, j] == 0 and sim.g[dr0 + i, dc0 + j] != 0]
            if zero:
                ops.append(0); sels.append(sel_of(zero))
                sim.op(0, zero)
        else:
            # the slide would be invisible in ARCLE (colour-0 piece gliding over
            # the region): paint the piece into its hole directly, colour by colour
            for colr in sorted(set(B.flatten().tolist())):
                cells = [(dr0 + i, dc0 + j) for i in range(B.shape[0]) for j in range(B.shape[1])
                         if B[i, j] == colr and sim.g[dr0 + i, dc0 + j] != colr]
                if cells:
                    ops.append(int(colr)); sels.append(sel_of(cells))
                    sim.op(int(colr), cells)
        sim.active = False

    # crop to the square region (its exact full rectangle)
    ops.append(33); sels.append([R0, C0, RH - 1, RW - 1])
    ops.append(34); sels.append([0, 0, RH - 1, RW - 1])
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
