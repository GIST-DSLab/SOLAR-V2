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
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------------------
# Reading: the keys (small two-colour full rectangles) are lifted out of the
# scene, the big rectangle is cut out on its own, and then each key -- turned
# or mirrored as needed -- is stamped whole into the bite whose shape it fits.

def _apply_ops(a, fops):
    for op in fops:
        if op == 26:
            a = np.fliplr(a)
        elif op == 27:
            a = np.flipud(a)
        elif op == 25:
            a = np.rot90(a, 3)
        elif op == 24:
            a = np.rot90(a, 1)
    return a


_FLAT = [[], [26], [27], [26, 27]]
_TURN = [[25], [24], [26, 25], [27, 25]]


def _components(I, bg):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bg:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _analyze(I, bg):
    keys, rest = [], []
    for cells in _components(I, bg):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        colors = {int(I[r, c]) for r, c in cells}
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(colors) == 2:
            keys.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        else:
            rest.extend(cells)
    if not rest:
        return None
    sqcs = {int(I[r, c]) for r, c in rest}
    if len(sqcs) != 1:
        return None
    sqc = sqcs.pop()
    rs = [r for r, _ in rest]; cs = [c for _, c in rest]
    R0, C0, R1, C1 = min(rs), min(cs), max(rs), max(cs)
    # a bite may swallow a whole edge line of the rectangle; then the visible
    # rectangle is too small and the keys cannot all fit -- widen it by the
    # fewest background lines that lets every key find its bite.
    hI, wI = I.shape
    for e in range(0, 7):
        for t in range(e + 1):
            for bo in range(e - t + 1):
                for l in range(e - t - bo + 1):
                    rt = e - t - bo - l
                    r0, r1, c0, c1 = R0 - t, R1 + bo, C0 - l, C1 + rt
                    if r0 < 0 or c0 < 0 or r1 >= hI or c1 >= wI:
                        continue
                    if e and not _bands_bg(I, bg, R0, C0, R1, C1, r0, c0, r1, c1):
                        continue
                    res = _fit(I, bg, sqc, keys, r0, c0, r1, c1)
                    if res is not None:
                        return res
    return None


def _bands_bg(I, bg, R0, C0, R1, C1, r0, c0, r1, c1):
    sub = I[r0:r1 + 1, c0:c1 + 1].copy()
    sub[R0 - r0:R1 - r0 + 1, C0 - c0:C1 - c0 + 1] = bg
    return bool(np.all(sub == bg))


def _fit(I, bg, sqc, keys, R0, C0, R1, C1):
    rect = I[R0:R1 + 1, C0:C1 + 1]
    rh, rw = rect.shape
    holes = {(r, c) for r in range(rh) for c in range(rw) if rect[r, c] == bg}

    cand = []
    for (r0, c0, kh, kw) in keys:
        K = I[r0:r0 + kh, c0:c0 + kw]
        if sqc not in K:
            return None
        col = [v for v in set(K.flatten().tolist()) if v != sqc][0]
        opts = []
        for fops in _FLAT + _TURN:
            T = _apply_ops(K, fops)
            th, tw = T.shape
            for rr in range(rh - th + 1):
                for cc in range(rw - tw + 1):
                    win = rect[rr:rr + th, cc:cc + tw]
                    if np.all((T == sqc) == (win == bg)) and np.all(win[T != sqc] == sqc):
                        bite = frozenset((rr + a, cc + b) for a in range(th) for b in range(tw) if T[a, b] == sqc)
                        opts.append((fops, T, rr, cc, bite))
        cand.append(((r0, c0, kh, kw), col, opts))

    best = [None]

    def bt(k, used, chosen):
        if best[0] is not None:
            return
        if k == len(cand):
            if used == holes:
                best[0] = list(chosen)
            return
        key, col, opts = cand[k]
        for o in opts:
            if o[4] & used:
                continue
            chosen.append((key, col, o))
            bt(k + 1, used | o[4], chosen)
            chosen.pop()

    bt(0, frozenset(), [])
    if best[0] is None:
        return None
    return R0, C0, rh, rw, sqc, best[0]


def _landing(pr, pc, kh, kw, fops):
    """Top-left where ARCLE puts a kh x kw object grabbed at (pr, pc) after fops."""
    x, y, h, w = pr, pc, kh, kw
    parity = 0
    for op in fops:
        if op not in (24, 25):
            continue
        k = 1 if op == 24 else 3
        xmin, xmax, ymin, ymax = x, x + h - 1, y, y + w - 1
        cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
        if h % 2 == w % 2:
            x, y = int(np.floor(cx - cy + ymin)), int(np.floor(cy - cx + xmin))
        else:
            parity = (parity + k) % 2
            sig = (k + 2) % 4 - 2
            mod = 1 - parity
            mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
            my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
            x, y = int(np.floor(mx)), int(np.floor(my))
        h, w = w, h
    return x, y


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour the demonstrations share; take the most
    # frequent colour (in I and the example inputs) that gives a complete
    # key <-> bite matching.
    cnt = Counter(I.flatten().tolist())
    for ex in (examples or []):
        cnt.update(np.asarray(ex[0]).flatten().tolist())
    info = None
    for bg in [c for c, _ in cnt.most_common()]:
        info = _analyze(I, int(bg))
        if info is not None:
            break

    ops, sels = [], []
    if info is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    R0, C0, rh, rw, sqc, matches = info

    # 1. cut the big rectangle out on its own (selection = that full rectangle);
    #    the keys stay readable from the input for CopyI.
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])

    # 2. stamp each key, in reading order of where it lies in the input.
    #    `grid` tracks the visible canvas so a paste footprint can be restored.
    grid = np.zeros((rh, rw), dtype=int)
    rect = I[R0:R0 + rh, C0:C0 + rw]
    grid[rect != 0] = rect[rect != 0]
    for (r0, c0, kh, kw), col, (fops, T, rr, cc, bite) in sorted(matches, key=lambda m: (m[0][0], m[0][1])):
        th, tw = T.shape
        body = [(rr + a, cc + b) for a in range(th) for b in range(tw) if T[a, b] != sqc]
        if col == 0 or sqc == 0 or kh > rh or kw > rw:
            # a 0-coloured key is invisible to Copy/Paste: draw it instead --
            # fill the bite with the rectangle colour, then the key colour
            ops.append(int(sqc)); sels.append(sel_of(sorted(bite)))
            for r, c in bite:
                grid[r, c] = sqc
            if body:
                ops.append(int(col)); sels.append(sel_of(body))
                for r, c in body:
                    grid[r, c] = col
        else:
            # where to lay the copy down: on the bite itself when the key keeps
            # its proportions; for a quarter turn of an oblong key, the in-canvas
            # spot whose turned result lands closest to the bite.
            best = None
            for pr in range(rh - kh + 1):
                for pc in range(rw - kw + 1):
                    lr, lc = _landing(pr, pc, kh, kw, fops)
                    d = abs(lr - rr) + abs(lc - cc)
                    if best is None or d < best[0]:
                        best = (d, pr, pc, lr, lc)
            _, pr, pc, lr, lc = best
            before = grid.copy()
            ops.append(28); sels.append([r0, c0, kh - 1, kw - 1])   # CopyI: the key's full box
            ops.append(30); sels.append([pr, pc, 0, 0])             # Paste
            grid[pr:pr + kh, pc:pc + kw] = I[r0:r0 + kh, c0:c0 + kw]
            first = True
            for op in fops:                                         # turn / mirror the pasted key
                ops.append(op)
                sels.append([pr, pc, kh - 1, kw - 1] if first else sel_of([]))
                first = False
            mv = []
            mv += [21 if lr < rr else 20] * abs(rr - lr)
            mv += [22 if lc < cc else 23] * abs(cc - lc)
            for op in mv:                                           # slide it onto its bite
                ops.append(op)
                sels.append([pr, pc, kh - 1, kw - 1] if first else sel_of([]))
                first = False
            foot = {(pr + a, pc + b) for a in range(kh) for b in range(kw)}
            dest = {(rr + a, cc + b) for a in range(th) for b in range(tw)}
            for (r, c) in foot - dest:
                grid[r, c] = 0
            for (r, c) in dest:
                grid[r, c] = T[r - rr, c - cc]
            # the copy's first footprint reads 0 where the turned key no longer
            # covers it: give those cells back what they held before the paste
            left = foot - dest
            for colr in sorted({int(before[r, c]) for r, c in left}):
                if colr == 0:
                    continue
                cells = sorted((r, c) for r, c in left if before[r, c] == colr)
                ops.append(colr); sels.append(sel_of(cells))
                for r, c in cells:
                    grid[r, c] = colr

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
