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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc = canvas colour, sqc = colour of the big rectangle (and of every key's
    # 'shape' cells).  Each key's own second colour is drawn per key inside generate().
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    hlo, wlo = min(15, max_h), min(15, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max(hlo, max_h)))
    w = unifint(diff_lb, diff_ub, (wlo, max(wlo, max_w)))
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
    if succ == 0:
        raise ValueError("no key placed")
    # keep only instances whose key->bite matching is decidable from the input
    info = _analyze(np.array(gi, dtype=int), bgc)
    if info is None or len(info[6]) != 1 or not np.array_equal(info[6][0], np.array(go)):
        raise ValueError("ambiguous key placement")
    return {'input': gi, 'output': go}


# ── trajectory ────────────────────────────────────────────────────────────────
# Reading: the big rectangle carries "bites" (background-coloured holes).  Every
# small two-colour key tile lying outside is the plug for exactly one bite, up to
# a reflection / rotation.  So each key is picked up as ONE object, turned the
# right way (Flip / Rotate on the grabbed object), slid into its bite with unit
# Moves, and finally the canvas is cropped to the rectangle.

# dihedral group as ARCLE op chains applied to a grabbed object
_DIHEDRAL = [
    ([], lambda A: A),
    ([26], lambda A: np.fliplr(A)),
    ([27], lambda A: np.flipud(A)),
    ([24], lambda A: np.rot90(A, 1)),
    ([25], lambda A: np.rot90(A, 3)),
    ([26, 27], lambda A: np.rot90(A, 2)),
    ([26, 24], lambda A: np.rot90(np.fliplr(A), 1)),
    ([26, 25], lambda A: np.rot90(np.fliplr(A), 3)),
]


class _ObjSim:
    """Minimal replica of ARCLE's object-selection mode (Move / Flip / Rotate)."""

    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)
        self.active = False

    def grab(self, cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        x0, x1, y0, y1 = min(rs), max(rs), min(cs), max(cs)
        h, w = x1 - x0 + 1, y1 - y0 + 1
        self.obj = np.zeros((h, w), dtype=int)
        for r, c in cells:
            self.obj[r - x0, c - y0] = self.grid[r, c]
        self.bg = self.grid.copy()
        for r, c in cells:
            self.bg[r, c] = 0
        self.pos = (x0, y0)
        self.parity = 0
        self.active = True

    def _patch(self):
        g = self.bg.copy()
        gh, gw = g.shape
        x, y = self.pos
        h, w = self.obj.shape
        for a in range(h):
            for b in range(w):
                r, c = x + a, y + b
                if 0 <= r < gh and 0 <= c < gw and self.obj[a, b] > 0:
                    g[r, c] = self.obj[a, b]
        self.grid = g

    def apply(self, op):
        h, w = self.obj.shape
        x, y = self.pos
        if op in (20, 21, 22, 23):
            dx, dy = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
            self.pos = (x + dx, y + dy)
        elif op == 26:
            self.obj = np.fliplr(self.obj)
        elif op == 27:
            self.obj = np.flipud(self.obj)
        elif op in (24, 25):
            k = 1 if op == 24 else 3
            xmin, xmax, ymin, ymax = x, x + h - 1, y, y + w - 1
            cx, cy = (xmax + xmin) * 0.5, (ymax + ymin) * 0.5
            if h % 2 == w % 2:
                self.pos = (int(np.floor(cx - cy + y)), int(np.floor(cy - cx + x)))
            else:
                self.parity = (self.parity + k) % 2
                sig = (k + 2) % 4 - 2
                mod = 1 - self.parity
                mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
                my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
                self.pos = (int(np.floor(mx)), int(np.floor(my)))
            self.obj = np.rot90(self.obj, k)
        self._patch()


def _components(I, bg):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    out = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bg:
                continue
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < hi and 0 <= ny < wi and not seen[nx, ny] and I[nx, ny] != bg:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            out.append(cells)
    return out


def _analyze(I, bg, limit=2):
    """Split I into the bitten rectangle and the key tiles, and match every key to
    one bite (orientation + slot) using I alone.  Returns up to `limit` distinct
    complete matchings (their resulting rectangles in the last field)."""
    tiles, rest = [], []
    for cells in _components(I, bg):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        pal = {int(I[p]) for p in cells}
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(pal) == 2:
            tiles.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        else:
            rest.append((r0, c0, r1, c1, pal))
    if not rest:
        return None
    pals = set().union(*[p for *_, p in rest])
    if len(pals) != 1:
        return None
    sqc = pals.pop()
    R0 = min(x[0] for x in rest); C0 = min(x[1] for x in rest)
    R1 = max(x[2] for x in rest); C1 = max(x[3] for x in rest)
    sub = I[R0:R1 + 1, C0:C1 + 1]
    rh, rw = sub.shape
    holes = frozenset((a, b) for a in range(rh) for b in range(rw) if sub[a, b] == bg)
    if not all(sqc in set(I[r:r + th, c:c + tw].flatten().tolist()) for r, c, th, tw in tiles):
        return None

    cand = []
    for (r0, c0, th, tw) in tiles:
        G = I[r0:r0 + th, c0:c0 + tw]
        lst, seenT = [], []
        for fops, fn in _DIHEDRAL:
            T = np.array(fn(G))
            if any(np.array_equal(T, S) for S in seenT):
                continue          # same picture already reachable with fewer ops
            seenT.append(T)
            ph, pw = T.shape
            if ph > rh or pw > rw:
                continue
            bitten = np.where(T == sqc, bg, sqc)
            for rr in range(rh - ph + 1):
                for cc in range(rw - pw + 1):
                    if np.array_equal(sub[rr:rr + ph, cc:cc + pw], bitten):
                        hc = frozenset((rr + a, cc + b) for a in range(ph) for b in range(pw)
                                       if T[a, b] == sqc)
                        lst.append((fops, T, rr, cc, hc))
        cand.append(lst)

    order = sorted(range(len(tiles)), key=lambda i: len(cand[i]))
    pick = [None] * len(tiles)
    sols, outs = [], []

    def bt(k, used):
        if len(sols) >= limit:
            return
        if k == len(order):
            if used != holes:
                return
            out = np.full((rh, rw), sqc, dtype=int)
            for i in range(len(tiles)):
                _, T, rr, cc, _ = pick[i]
                out[rr:rr + T.shape[0], cc:cc + T.shape[1]] = T
            if not any(np.array_equal(out, o) for o in outs):
                outs.append(out)
                sols.append(list(pick))
            return
        i = order[k]
        for cnd in cand[i]:
            if cnd[4] & used or not cnd[4] <= holes:
                continue
            pick[i] = cnd
            bt(k + 1, used | cnd[4])
        pick[i] = None

    bt(0, frozenset())
    if not sols:
        return None
    return R0, C0, rh, rw, sqc, [(tiles[i], sols[0][i]) for i in range(len(tiles))], outs


def _route(grid, cells, fops, dest, history):
    """BFS over (position, turned?) for one grabbed key: unit Moves plus the turn
    (fops, done once wherever it is visible) so that EVERY op changes the grid and
    the turned key ends with its top-left at dest.  Returns (ops, grid after)."""
    from collections import deque
    gh, gw = grid.shape
    x0 = min(r for r, _ in cells)
    y0 = min(c for _, c in cells)
    base = _ObjSim(grid)
    base.grab(cells)
    G0 = base.obj.copy()
    bgsnap = base.bg

    def render(obj, pos):
        s = _ObjSim.__new__(_ObjSim)
        s.bg, s.obj, s.pos = bgsnap, obj, pos
        s._patch()
        return s.grid

    def turn_at(pos):
        s = _ObjSim.__new__(_ObjSim)
        s.bg, s.obj, s.pos, s.parity = bgsnap, G0.copy(), pos, 0
        s._patch()
        prev, ok, mids = s.grid.copy(), True, []
        for op in fops:
            s.apply(op)
            if np.array_equal(prev, s.grid):
                ok = False
            prev = s.grid.copy()
            mids.append(prev.tobytes())
        return ok, s.obj, s.pos, mids

    def inside(obj, p):
        return 0 <= p[0] <= gh - obj.shape[0] and 0 <= p[1] <= gw - obj.shape[1]

    turned_obj = G0 if not fops else turn_at((x0, y0))[1]
    claimed = set(history)   # grids already shown: never show one again
    start = ((x0, y0), 0 if fops else 1)
    claimed.add(render(G0, (x0, y0)).tobytes())
    prev = {start: None}
    dq = deque([start])
    goal = None
    while dq:
        st = dq.popleft()
        pos, ph = st
        if ph == 1 and pos == tuple(dest):
            goal = st
            break
        obj = G0 if ph == 0 else turned_obj
        cur = render(obj, pos)
        nxt = []
        for op, (dx, dy) in ((20, (-1, 0)), (21, (1, 0)), (22, (0, 1)), (23, (0, -1))):
            np_ = (pos[0] + dx, pos[1] + dy)
            if inside(obj, np_):
                nxt.append(([op], (np_, ph), [render(obj, np_).tobytes()]))
        if ph == 0:
            ok, tobj, tpos, mids = turn_at(pos)
            if ok and inside(tobj, tpos):
                nxt.append((list(fops), (tuple(tpos), 1), mids))
        for o, ns, hs in nxt:
            if ns in prev or any(hh in claimed for hh in hs):
                continue
            claimed.update(hs)
            prev[ns] = (st, o)
            dq.append(ns)
    if goal is None:
        return None
    seq, st = [], goal
    while prev[st] is not None:
        st, o = prev[st]
        seq = o + seq
    s = _ObjSim(grid)
    s.grab(cells)
    for op in seq:
        s.apply(op)
        history.add(s.grid.tobytes())
    return seq, s.grid


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour; take the most frequent colour that yields
    # a complete key<->bite matching (the demonstrations share this layout).
    freq = [c for c, _ in Counter(I.flatten().tolist()).most_common()]
    info = None
    for bg in freq:
        info = _analyze(I, int(bg))
        if info is not None:
            break

    ops, sels = [], []
    if info is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    R0, C0, rh, rw, sqc, matches, _ = info
    grid = I.copy()
    history = {grid.tobytes()}
    # one key at a time, in reading order of where the keys lie in the input
    for (r0, c0, th, tw), (fops, T, rr, cc, _) in sorted(matches, key=lambda m: (m[0][0], m[0][1])):
        cells = [(r, c) for r in range(r0, r0 + th) for c in range(c0, c0 + tw)]
        dest = (R0 + rr, C0 + cc)
        ph, pw = T.shape
        res = _route(grid, cells, fops, dest, history)
        if res is None:
            # A key coloured 0 is transparent to ARCLE's object ops, so over the
            # rectangle it cannot travel visibly.  Plug that one bite in place:
            # fill the bite with the rectangle colour, then draw the key's own
            # colour on the key's (turned) non-shape cells in the slot.
            bite = [(dest[0] + a, dest[1] + b) for a in range(ph) for b in range(pw) if T[a, b] == sqc]
            body = [(dest[0] + a, dest[1] + b) for a in range(ph) for b in range(pw) if T[a, b] != sqc]
            keyc = [v for v in set(T.flatten().tolist()) if v != sqc][0]
            for colr, part in ((sqc, bite), (keyc, body)):
                if part:
                    ops.append(int(colr)); sels.append(sel_of(part))
                    for r, c in part:
                        grid[r, c] = colr
                    history.add(grid.tobytes())
            continue
        seq, grid = res
        for k, op in enumerate(seq):
            ops.append(op)
            sels.append(sel_of(cells) if k == 0 else sel_of([]))   # grab once, then continue
        # ARCLE only pastes nonzero object cells: a key colour that is 0 is laid
        # explicitly on the key's own (turned) 0-cells now sitting in the slot.
        zeros = [(dest[0] + a, dest[1] + b) for a in range(ph) for b in range(pw) if T[a, b] == 0]
        if zeros:
            ops.append(0); sels.append(sel_of(zeros))
            for r, c in zeros:
                grid[r, c] = 0
            history.add(grid.tobytes())

    # keep only the rectangle — the selection is exactly that full rectangle
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])
    ops.append(34); sels.append([0, 0, rh - 1, rw - 1])
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
                            ops, sels = derive_operations(I, O, list(zip(ex_in, ex_out)))
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
