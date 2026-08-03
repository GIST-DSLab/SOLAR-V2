"""
ARC Task: 6a1e5592 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

from maker.sel_helpers import sel_of

# the bar sits on one of the four sides (the generator's final rot90/180/270)
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]      # 1 is the answer colour
    barc, bgc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    max_h = max(5, min(30, int(max_h)))
    max_w = max(5, min(30, int(max_w)))
    hlim, wlim = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    if hlim < 9 or wlim < 5:
        rot = 0
        hlim, wlim = max_h, max_w
    hlim = max(9, hlim)
    wlim = max(5, wlim)

    rotf = (identity, rot90, rot180, rot270)[rot % 4]
    best = None
    for _attempt in range(400):
        h = unifint(diff_lb, diff_ub, (9, hlim))
        w = unifint(diff_lb, diff_ub, (5, wlim))
        barh = randint(3, h // 3)
        maxobjh = h - barh - 1
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
        c1 = canvas(barc, (barh, w))
        c2 = canvas(bgc, (h - barh, w))
        gi = vconcat(c1, c2)
        go = tuple(e for e in gi)
        tr = 0
        succ = 0
        maxtr = 10 * nobjs
        placopts = interval(1, w - 1, 1)
        iinds = ofcolor(gi, bgc)
        oinds = asindices(go)
        barinds = ofcolor(gi, barc)
        forbmarkers = set()
        while tr < maxtr and succ < nobjs:
            tr += 1
            oh = randint(1, maxobjh)
            ow = randint(1, min(4, w // 2))
            bounds = asindices(canvas(-1, (oh, ow)))
            ncells = randint(1, oh * ow)
            sp = choice(totuple(connect((0, 0), (0, ow - 1))))
            obj = {sp}
            for k in range(ncells - 1):
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            markerh = randint(1, min(oh, barh - 1))
            markpart = sfilter(obj, lambda ij: ij[0] < markerh)
            markpartn = normalize(markpart)
            isinvalid = False
            for k in range(1, markerh + 1):
                if normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers:
                    isinvalid = True
            if isinvalid:
                continue
            placoptcands = sfilter(
                placopts, lambda jj: set(interval(jj, jj + ow + 1, 1)).issubset(set(placopts)))
            if len(placoptcands) == 0:
                continue
            jloc = choice(placoptcands)
            iloc = barh - markerh
            oplcd = shift(obj, (iloc, jloc))
            if not oplcd.issubset(oinds):
                continue
            icands = sfilter(iinds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(icands) == 0:
                continue
            loc = choice(totuple(icands))
            iplcd = shift(obj, loc)
            if not iplcd.issubset(iinds):
                continue
            for k in range(1, markerh + 1):
                forbmarkers.add(normalize(sfilter(markpartn, lambda ij: ij[0] < k)))
            succ += 1
            iinds = (iinds - iplcd) - mapply(neighbors, iplcd)
            oinds = (oinds - oplcd)
            gi = fill(gi, objc, iplcd)
            gi = fill(gi, bgc, oplcd & barinds)
            go = fill(go, 1, oplcd)
            jm = apply(last, ofcolor(go, 1))
            placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
            if len(placopts) == 0:
                break
        if succ >= 1:
            best = (rotf(gi), rotf(go))
            # keep only instances whose marker->object reading is unambiguous and
            # whose slides stay visible the whole way
            if _trajectory_is_clean(np.array(best[0]), np.array(best[1])):
                break
    if best is None:                       # no object ever fitted: empty scene
        best = (rotf(gi), rotf(go))
    return {'input': best[0], 'output': best[1]}


# ── derive_operations ────────────────────────────────────────────────────────

_MOVE_OP = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}


def _canon_vec(dr, dc, ru, cu):
    """Canonical-frame translation -> original-frame (row, col) translation."""
    return (ru[0] * dr + cu[0] * dc, ru[1] * dr + cu[1] * dc)


def _canonicalize(I):
    """Find the rotation that puts the solid bar band on top; return
    (canonical grid, index map, band thickness, barc, bgc) or None."""
    h, w = I.shape
    idx = np.arange(h * w).reshape(h, w)
    for k in range(4):
        G = np.rot90(I, k)
        Gi = np.rot90(idx, k)
        gh, gw = G.shape
        if gh < 9 or gw < 5:
            continue
        X = int(G[0, 0])
        if not np.all(G[0] == X):
            continue
        rws = np.where((G == X).any(axis=1))[0]
        band = int(rws.max()) + 1
        if band < 3 or band > gh // 3:
            continue
        bandcols = set(int(v) for v in np.unique(G[:band]))
        if len(bandcols) != 2:                 # bar colour + the carved notches
            continue
        bgc = [c for c in bandcols if c != X][0]
        rest = set(int(v) for v in np.unique(G[band:]))
        if X in rest or bgc not in rest or len(rest) > 2:
            continue
        return G, Gi, band, X, bgc
    return None


def _components(cells):
    cells = set(cells)
    out = []
    while cells:
        seed = next(iter(cells))
        comp = set()
        dq = deque([seed])
        cells.discard(seed)
        while dq:
            r, c = dq.popleft()
            comp.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cells:
                    cells.discard((nr, nc))
                    dq.append((nr, nc))
        out.append(comp)
    out.sort(key=lambda s: min(s))
    return out


class _Sim:
    """Replay of the ARCLE ops used here (Color / Move) so the derivation can
    check that every op it emits actually changes the visible grid."""

    def __init__(self, grid):
        self.g = np.asarray(grid, dtype=int).copy()
        self.obj = None
        self.bg = None
        self.active = False

    def clone(self):
        s = _Sim(self.g)
        s.obj = None if self.obj is None else list(self.obj)
        s.bg = None if self.bg is None else self.bg.copy()
        s.active = self.active
        return s

    def color(self, n, cells):
        before = self.g.copy()
        for (r, c) in cells:
            self.g[r, c] = n
        self.active = False
        return not np.array_equal(before, self.g)

    def move(self, op, cells):
        before = self.g.copy()
        H, W = self.g.shape
        if cells:                       # non-empty selection grabs a new object
            self.obj = [(r, c, int(self.g[r, c])) for (r, c) in cells if self.g[r, c] != 0]
            self.bg = self.g.copy()
            for (r, c) in cells:
                self.bg[r, c] = 0
            self.active = True
        if not self.active:
            return False
        dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
        self.obj = [(r + dr, c + dc, v) for (r, c, v) in self.obj]
        self.g = self.bg.copy()
        for (r, c, v) in self.obj:
            if 0 <= r < H and 0 <= c < W:
                self.g[r, c] = v
        return not np.array_equal(before, self.g)


def _trajectory_is_clean(I, O):
    """Replay derive_operations on a candidate instance: every op must change the
    visible grid, no state may repeat, and the last state must be O. Instances
    that fail (e.g. two objects whose markers would each fit the other's notch,
    so the episode itself is ambiguous) are re-rolled by generate()."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    try:
        ops, sels = derive_operations(I, O)
    except Exception:
        return False
    sim = _Sim(I)
    seen = {sim.g.tobytes()}
    for op, sel in zip(ops, sels):
        if op == 34:
            break
        if not isinstance(sel, dict):
            return False
        cells = [tuple(c) for c in sel["cells"]]
        changed = sim.move(op, cells) if 20 <= op <= 23 else sim.color(op, cells)
        if not changed:
            return False
        key = sim.g.tobytes()
        if key in seen:
            return False
        seen.add(key)
    return sim.g.shape == O.shape and np.array_equal(sim.g, O)


def derive_operations(I, O):
    """Each object in the field belongs to the notch whose shape is the object's
    own top rows; the object slides up into that notch wearing colour 1."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    submit = [0, 0, ho - 1, wo - 1]          # whole grid: bbox == intended cells

    info = _canonicalize(I)
    if info is None:                         # no bar/notches: nothing happens
        return [34], [submit]
    G, Gi, band, barc, bgc = info
    gh, gw = G.shape
    w = I.shape[1]

    def orig(rc):                            # canonical cell -> original cell
        r, c = rc
        v = int(Gi[r, c])
        return (v // w, v % w)

    p00 = orig((0, 0))
    ru = (orig((1, 0))[0] - p00[0], orig((1, 0))[1] - p00[1])
    cu = (orig((0, 1))[0] - p00[0], orig((0, 1))[1] - p00[1])

    palette = set(int(v) for v in np.unique(G[band:]))
    objcs = [c for c in palette if c != bgc]
    if not objcs:
        return [34], [submit]
    objc = objcs[0]

    notch = {(int(r), int(c)) for r, c in zip(*np.where(G[:band] == bgc))}
    objcells = {(int(r), int(c)) for r, c in zip(*np.where(G == objc))}

    # every object's top k rows must land exactly on notch cells, and together the
    # objects have to account for every notch cell — that exact cover is what
    # pins each object to its own notch
    comps = _components(objcells)
    cand_lists = []
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, c0 = min(rs), min(cs)
        oh = max(rs) - r0 + 1
        ow = max(cs) - c0 + 1
        norm = {(r - r0, c - c0) for r, c in comp}
        cands = []
        for k in range(1, min(oh, band - 1) + 1):
            pref = {(r, c) for (r, c) in norm if r < k}
            if not pref or band - k + oh > gh:
                continue
            for jloc in range(1, gw - 1 - ow):   # cols jloc..jloc+ow stay interior
                mark = {(r + band - k, c + jloc) for (r, c) in pref}
                if not mark <= notch:
                    continue
                dest = {(r + band - k, c + jloc) for (r, c) in norm}
                cands.append((mark, dest, (band - k) - r0, jloc - c0))
        cands.sort(key=lambda t: (not (1 <= min(c for _, c in t[1])
                                       and max(c for _, c in t[1]) <= gw - 3),
                                  -len(t[0])))
        cand_lists.append(cands)

    order_idx = sorted(range(len(comps)), key=lambda i: len(cand_lists[i]))
    chosen = {}

    def _cover(pos, covered, taken):
        if pos == len(order_idx):
            return covered == notch
        i = order_idx[pos]
        for mark, dest, dr, dc in cand_lists[i]:
            if mark & covered or dest & taken:
                continue
            chosen[i] = (dest, dr, dc)
            if _cover(pos + 1, covered | mark, taken | dest):
                return True
            del chosen[i]
        return False

    _cover(0, set(), set())
    plans = [(comps[i], chosen[i][0], chosen[i][1], chosen[i][2])
             for i in range(len(comps)) if i in chosen]
    order = list(range(len(plans)))          # raster order of the field objects

    def _paths(bg, cells, tr, tc, seen_before):
        """Pick the slide from the object's place to (tr, tc): a straight one if
        every step of it redraws the grid, otherwise the shortest detour that
        does — an object gliding over ground of its own colour would be an
        invisible, meaningless step."""
        H, W = I.shape

        def comp(off):
            g = bg.copy()
            for (r, c) in cells:
                rr, cc = r + off[0], c + off[1]
                if 0 <= rr < H and 0 <= cc < W:
                    g[rr, cc] = 1
            return g

        def ok(seq):
            off, seen = (0, 0), set(seen_before)
            for (sr, sc) in seq:
                off = (off[0] + sr, off[1] + sc)
                key = comp(off).tobytes()
                if key in seen:
                    return False
                seen.add(key)
            return True

        vs = [((-1 if tr < 0 else 1), 0)] * abs(tr)
        hs = [(0, (-1 if tc < 0 else 1))] * abs(tc)
        for seq in (vs + hs, hs + vs):
            if ok(seq):
                return seq
        rlo, rhi = min(0, tr) - 6, max(0, tr) + 6
        clo, chi = min(0, tc) - 6, max(0, tc) + 6
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        prev = {(0, 0): None}
        seen = set(seen_before) | {comp((0, 0)).tobytes()}
        dq = deque([(0, 0)])
        while dq:
            off = dq.popleft()
            if off == (tr, tc):
                break
            for (sr, sc) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nxt = (off[0] + sr, off[1] + sc)
                if nxt in prev or not (rlo <= nxt[0] <= rhi and clo <= nxt[1] <= chi):
                    continue
                if min(rs) + nxt[0] < 0 or max(rs) + nxt[0] >= H:
                    continue
                if min(cs) + nxt[1] < 0 or max(cs) + nxt[1] >= W:
                    continue
                key = comp(nxt).tobytes()
                if key in seen:              # invisible step, or a grid seen before
                    continue
                seen.add(key)
                prev[nxt] = (off, (sr, sc))
                dq.append(nxt)
        if (tr, tc) in prev:
            seq, node = [], (tr, tc)
            while prev[node] is not None:
                node, step = prev[node]
                seq.append(step)
            seq.reverse()
            return seq
        return vs + hs

    sim = _Sim(I)
    ops, cellsels, states = [], [], [sim.g.copy()]
    placed = set()

    def do(op, cells, mv=False):
        (sim.move if mv else sim.color)(op, cells)
        ops.append(op)
        cellsels.append(cells)
        states.append(sim.g.copy())

    for pos, i in enumerate(order):
        src, dest, dr, dc = plans[i]
        later = set()
        for j in order[pos + 1:]:
            later |= plans[j][1]
        # the object takes the answer colour where it stands, then slides up into
        # the notch that spells out its own top rows (nothing to recolour if a
        # body placed earlier already covers it completely)
        src_o = [orig(x) for x in sorted(src)]
        if any(sim.g[r, c] != 1 for (r, c) in src_o):
            do(1, src_o)
        bg = sim.g.copy()
        for (r, c) in src_o:
            bg[r, c] = 0
        seq = _paths(bg, src_o, *_canon_vec(dr, dc, ru, cu),
                     {g.tobytes() for g in states})
        cur = set(src_o)
        first = True
        for (sr, sc) in seq:
            # first step grabs the object; the rest carry an empty selection so
            # ARCLE keeps it grabbed and restores everything it glides over
            do(_MOVE_OP[(sr, sc)], sorted(cur) if first else [], mv=True)
            first = False
            cur = {(r + sr, c + sc) for (r, c) in cur}
        hole = set(src_o) - cur
        later_o = set(orig(x) for x in later)
        # The grab left the vacated footprint at 0. Where this object had been
        # standing on a body already placed, that body shows through again;
        # everything else goes back to the background.
        overlap = sorted(hole & placed)
        if overlap:
            do(1, overlap)
        rest = sorted(hole - placed)
        # skipped only when that whole footprint is where a still-to-be-placed
        # object lands, so there is no background left to restore there
        if bgc != 0 and rest and not set(rest) <= later_o:
            do(bgc, rest)
        placed |= cur

    ops.append(34)
    cellsels.append(None)
    sels = [sel_of(cs) if cs is not None else submit for cs in cellsels[:-1]] + [submit]
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
                        f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
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
                                f"for task 6a1e5592"
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
                    f"Failed to build a complete episode for task 6a1e5592 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6a1e5592-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
