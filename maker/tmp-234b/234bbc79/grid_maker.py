"""
ARC Task: 234bbc79 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for (r, c) in cells]}


# ----------------------------------------------------------------------------- helpers
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
# The snake was cut into segments at its rightward steps and the segments were
# scattered rightwards.  Every segment carries marker cells in one shared colour
# (its head, its tail, or both); re-assembly slides each segment so that its head
# marker sits one column right of the previous segment's tail marker, the markers
# become ordinary body cells, and the canvas is cropped to the rebuilt snake.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lb = int(a + (b - a) * diff_lb)
    ub = int(a + (b - a) * diff_ub)
    lb = max(a, min(b, lb))
    ub = max(a, min(b, ub))
    if ub < lb:
        lb, ub = ub, lb
    return random.randint(lb, ub)


def _components(I, bgc):
    """4-connected components of the non-background cells."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((int(y), int(x)))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(sorted(cells))
    return comps


def _plan(I):
    """Read the puzzle off I alone: the pieces, the marker colour, the assembly.

    The marker colour is the one that sits on a single cell of the leftmost
    piece, on a single cell of the rightmost piece and on two cells of every
    piece between them -- a body colour cannot do that, because the leftmost
    piece keeps its own colour on both of its ends but one.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = Counter(I.reshape(-1).tolist())
    ranked = cnt.most_common()
    if not ranked:
        return None
    bgc = int(ranked[0][0])
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None
    comps = _components(I, bgc)
    n = len(comps)
    if n < 2:
        return None
    comps.sort(key=lambda cs: (min(c for (_, c) in cs), min(r for (r, _) in cs)))

    cands = []
    for col in sorted(set(int(v) for v in cnt) - {bgc}, key=lambda c: (cnt[c], c)):
        ok = True
        for i, cs in enumerate(comps):
            k = sum(1 for p in cs if int(I[p]) == col)
            if k != (1 if i in (0, n - 1) else 2) or len(set(int(I[p]) for p in cs)) != 2:
                ok = False
                break
        if ok:
            cands.append(col)

    for dotc in cands:
        pieces = []
        good = True
        for i, cs in enumerate(comps):
            dots = sorted(p for p in cs if int(I[p]) == dotc)
            bodies = set(int(I[p]) for p in cs if int(I[p]) != dotc)
            cmin = min(c for (_, c) in cs)
            cmax = max(c for (_, c) in cs)
            if len(bodies) != 1 or cmax == cmin:
                good = False
                break
            head = tail = None
            if i > 0:
                at = [p for p in dots if p[1] == cmin]
                if len(at) != 1:
                    good = False
                    break
                head = at[0]
            if i < n - 1:
                at = [p for p in dots if p[1] == cmax]
                if len(at) != 1:
                    good = False
                    break
                tail = at[0]
            pieces.append({"cells": cs, "dots": dots, "head": head, "tail": tail,
                           "body": int(bodies.pop())})
        if not good:
            continue

        offs = [(0, 0)]
        acc = pieces[0]["tail"]
        for i in range(1, n):
            hd = pieces[i]["head"]
            dr, dc = acc[0] - hd[0], acc[1] - hd[1] + 1
            offs.append((int(dr), int(dc)))
            tl = pieces[i]["tail"]
            if tl is not None:
                acc = (tl[0] + dr, tl[1] + dc)

        final = {}
        for i, p in enumerate(pieces):
            dr, dc = offs[i]
            for (r, c) in p["cells"]:
                q = (r + dr, c + dc)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or q in final:
                    good = False
                    break
                final[q] = p["body"]
            if not good:
                break
        if not good:
            continue
        if min(c for (_, c) in final) != 0:
            continue
        W = max(c for (_, c) in final) + 1

        # can the pieces be dealt with in any order?  Only if no piece's landing
        # place is somewhere another piece is still sitting.
        orig = [set(p["cells"]) for p in pieces]
        dest = [set((r + offs[i][0], c + offs[i][1]) for (r, c) in p["cells"])
                for i, p in enumerate(pieces)]
        free = all(not (dest[i] & orig[j]) for i in range(n) for j in range(n) if i != j)

        out = np.full((h, W), bgc, dtype=int)
        for (r, c), v in final.items():
            if c < W:
                out[r, c] = v
        return {"bgc": bgc, "dotc": int(dotc), "W": int(W), "pieces": pieces,
                "offs": offs, "final": final, "free": bool(free), "out": out}
    return None


def _build(I, plan, flags):
    """One route through the plan.  `grid` mirrors what ARCLE holds, so that no
    operation is written whose cells already carry the colour it would paint."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    grid = I.copy()
    ops, sels = [], []
    bgc, dotc, W = plan["bgc"], plan["dotc"], plan["W"]
    pieces, offs = plan["pieces"], plan["offs"]
    n = len(pieces)

    def color(col, cells):
        cells = sorted({(int(r), int(c)) for (r, c) in cells
                        if 0 <= r < hi and 0 <= c < wi and grid[r, c] != col})
        if not cells:
            return
        ops.append(int(col))
        sels.append(sel_of(cells))
        for (r, c) in cells:
            grid[r, c] = int(col)

    def slide(cells, dr, dc, vfirst):
        """Grab the piece once, then walk it; ARCLE keeps the cells it crosses."""
        vert = [((21 if dr > 0 else 20), (1 if dr > 0 else -1, 0))] * abs(dr)
        horz = [((22 if dc > 0 else 23), (0, 1 if dc > 0 else -1))] * abs(dc)
        seq = (vert + horz) if vfirst else (horz + vert)
        if not seq:
            return
        vals = {p: int(grid[p]) for p in cells}
        back = grid.copy()
        for p in cells:
            back[p] = 0
        sr = sc = 0
        for k, (op, (vr, vc)) in enumerate(seq):
            ops.append(op)
            sels.append(sel_of([(r + sr, c + sc) for (r, c) in cells]) if k == 0
                        else sel_of([]))
            sr += vr
            sc += vc
            grid[:, :] = back
            for (r, c), v in vals.items():
                if v > 0 and 0 <= r + sr < hi and 0 <= c + sc < wi:
                    grid[r + sr, c + sc] = v


    leftovers = []

    def sweep(gone):
        # whatever the piece no longer covers goes back to the background
        if bgc != 0 and gone:
            if flags["batch_holes"]:
                leftovers.extend(gone)          # swept up once, at the end
            else:
                color(bgc, gone)

    order = list(range(n - 1, -1, -1)) if flags["backward"] else list(range(n))
    for i in order:
        p = pieces[i]
        dr, dc = offs[i]
        body = p["body"]
        dst = [(r + dr, c + dc) for (r, c) in p["cells"]]
        gone = [q for q in p["cells"] if q not in set(dst) and q[1] < W]

        if body == 0:
            # ARCLE picks up no cell holding 0, so a piece whose own colour is 0
            # cannot be carried: it is laid down where it lands and wiped where
            # it was.  Its markers are 0 too once absorbed, so one Color0 does
            # the whole piece.
            sweep(gone)
            color(0, dst)
            continue

        # A marker only has to be read before it is absorbed, so either side of
        # the slide will do -- except when the marker colour is 0, which has to
        # leave the piece before ARCLE can carry it.
        absorb_first = True if dotc == 0 else flags["absorb_first"]
        if absorb_first:
            color(body, p["dots"])
        if i > 0 and (dr or dc):
            slide(p["cells"], dr, dc, flags["vfirst"])
            sweep(gone)
        if not absorb_first:
            color(body, [(r + dr, c + dc) for (r, c) in p["dots"]])

    if flags["batch_holes"] and leftovers:
        color(bgc, [q for q in leftovers if q not in plan["final"]])

    if W != wi:
        ops.append(33)
        sels.append([0, 0, hi - 1, W - 1])   # full rectangle: the answer's frame
    ops.append(34)
    sels.append([0, 0, hi - 1, W - 1])
    return ops, sels, grid[:, :W]


def _undone(ops, sels):
    """Do the moves sharing a selection add up to standing still?"""
    step = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
    groups = {}
    for op, sel in zip(ops, sels):
        if int(op) in step:
            key = tuple(sel) if isinstance(sel, (list, tuple)) else str(sel)
            a, b = groups.get(key, (0, 0))
            groups[key] = (a + step[int(op)][0], b + step[int(op)][1])
    return any(v == (0, 0) for v in groups.values())


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    # 0 is an ordinary colour of the picture here: it may be the marker or a
    # segment's own colour, and the route carries it either way.
    bgc = random.choice(list(range(10)))
    rest = [c for c in range(10) if c != bgc]
    dotc = random.choice(rest)
    remcols = [c for c in rest if c != dotc]
    k = random.randint(1, len(remcols))
    pool = random.sample(remcols, k)
    return {"bgc": bgc, "dotc": dotc, "pool": pool}


# ----------------------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, pool=None, **kwargs) -> dict:
    if bgc is None or dotc is None or pool is None:
        ck = sample_colors()
        bgc, dotc, pool = ck["bgc"], ck["dotc"], ck["pool"]
    pool = list(pool)

    hmax = max(5, min(30, int(max_h)))
    wlim = max(8, min(30, int(max_w)))
    wcap = min(20, wlim - 2)
    if wcap < 6:
        wcap = 6

    for _attempt in range(6000):
        h = _unifint(diff_lb, diff_ub, (5, hmax))
        w = _unifint(diff_lb, diff_ub, (6, wcap))

        # --- grow the snake (right / up / down, never revisiting) ---
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        occ = {(spi, 0)}
        while True:
            pi, pj = snek[-1]
            if pj == w - 1 and random.choice((True, False, False)):
                break
            options = []
            if pi < h - 1 and (pi + 1, pj) not in occ:
                options.append((pi + 1, pj))
            if pi > 0 and (pi - 1, pj) not in occ:
                options.append((pi - 1, pj))
            if pj < w - 1:
                options.append((pi, pj + 1))
            if not options:
                break
            loc = random.choice(options)
            snek.append(loc)
            occ.add(loc)

        # --- cut into segments at rightward steps ---
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            cw = (max(c for (_, c) in cobj) - min(c for (_, c) in cobj)) if cobj else 0
            if len(cobj) > 2 and cw > 0 and idx > 0 and snek[idx - 1] == (cel[0], cel[1] - 1):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if not objs:
            continue
        objs[-1] = objs[-1] + cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = _unifint(diff_lb, diff_ub, (2, nobjs))
        for _ in range(nobjs - ntokeep):
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        n = len(objs)
        if n < 2:
            continue

        ok = True
        for ob in objs:
            cs = [c for (_, c) in ob]
            if max(cs) - min(cs) < 1 or ob[0][1] != min(cs) or ob[-1][1] != max(cs):
                ok = False
                break
        if not ok:
            continue

        sw = max(c for (_, c) in snek) + 1
        minw = sw + (n - 1)
        wtop = min(30, int(max_w))
        if minw > wtop or sw > wtop or h > int(max_h):
            continue
        fullw = _unifint(diff_lb, diff_ub, (minw, wtop))
        extra = fullw - minw
        spacings = [1] * (n - 1)
        for _ in range(extra):
            i = random.randint(0, n - 1)
            if i < n - 1:
                spacings[i] += 1

        # --- colours per segment ---
        ncols = _unifint(diff_lb, diff_ub, (1, len(pool)))
        ccols = random.sample(pool, ncols)
        segcols = [random.choice(ccols) for _ in objs]

        # --- scatter the segments over the input canvas ---
        gi = [[bgc] * fullw for _ in range(h)]
        nextcol = 0
        placed_fail = False
        for i, ob in enumerate(objs):
            colr = segcols[i]
            cells = {}
            for j, p in enumerate(ob):
                if j == 0 and i > 0:
                    cells[p] = dotc
                elif j == len(ob) - 1 and i < n - 1:
                    cells[p] = dotc
                else:
                    cells[p] = colr
            rs = [r for (r, _) in cells]
            cs = [c for (_, c) in cells]
            minr, minc = min(rs), min(cs)
            hh, ww = max(rs) - minr + 1, max(cs) - minc + 1
            if i == 0:
                ulr, ulc = minr, 0
            else:
                if h - hh < 0:
                    placed_fail = True
                    break
                ulr = random.randint(0, h - hh)
                ulc = nextcol + spacings[i - 1]
            if ulc + ww > fullw:
                placed_fail = True
                break
            nextcol = ulc + ww
            for (pr, pc), v in cells.items():
                gi[ulr + pr - minr][ulc + pc - minc] = v
        if placed_fail:
            continue

        # --- the answer: the snake put back together, markers absorbed ---
        go = [[bgc] * sw for _ in range(h)]
        for i, ob in enumerate(objs):
            for (pr, pc) in ob:
                go[pr][pc] = segcols[i]

        gi_np = np.array(gi, dtype=int)
        go_np = np.array(go, dtype=int)
        if Counter(gi_np.reshape(-1).tolist()).most_common(1)[0][0] != bgc:
            continue
        pl = _plan(gi_np)
        if pl is None or pl["out"].shape != go_np.shape or not np.array_equal(pl["out"], go_np):
            continue

        return {
            "input": tuple(tuple(int(v) for v in row) for row in gi),
            "output": tuple(tuple(int(v) for v in row) for row in go),
        }

    raise RuntimeError("234bbc79: could not generate an instance")


# ----------------------------------------------------------------------------- trajectory
def derive_operations(I, O):
    """Re-assemble the snake.  Which piece to start from, which way to walk it,
    when to absorb its marker and when to sweep up behind it are all choices the
    rule leaves open, so they are taken from the puzzle itself -- the number of
    pieces, the row its first marker sits on, the two colours, the height of the
    grid -- and every instance answers them its own way.  O is never read."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    plan = _plan(I)
    if plan is None:
        return [34], [[0, 0, hi - 1, wi - 1]]

    n = len(plan["pieces"])
    marker = plan["pieces"][0]["tail"]           # the leftmost piece's marker
    chosen = {
        # an even number of pieces: start at the far end and work back
        "backward": plan["free"] and n % 2 == 0,
        # the marker's row says whether to drop the piece into its row first
        "vfirst": marker[0] % 2 == 0,
        # a marker whose colour sits below the background's is read, and
        # absorbed, before its piece sets off
        "absorb_first": plan["dotc"] < plan["bgc"],
        # on an even-height grid the holes are swept up once, at the end
        "batch_holes": plan["bgc"] != 0 and hi % 2 == 0,
    }
    plain = {"backward": False, "vfirst": True, "absorb_first": False,
             "batch_holes": False}

    for flags in (chosen, plain):
        ops, sels, got = _build(I, plan, flags)
        if got.shape == plan["out"].shape and np.array_equal(got, plan["out"]) \
                and not _undone(ops, sels):
            return ops, sels
    return [34], [[0, 0, hi - 1, wi - 1]]
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
                        f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
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
                                f"for task 234bbc79"
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
                    f"Failed to build a complete episode for task 234bbc79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"234bbc79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
