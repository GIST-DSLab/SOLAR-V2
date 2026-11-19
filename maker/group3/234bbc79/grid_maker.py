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
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lb = max(a, min(b, int(a + (b - a) * diff_lb)))
    ub = max(a, min(b, int(a + (b - a) * diff_ub)))
    if ub < lb:
        lb, ub = ub, lb
    return random.randint(lb, ub)


def _components(I, bgc):
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
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _measure(I):
    """Read the snake pieces from I alone.

    Pieces are the components, ordered left to right.  The marker colour is
    the rarest colour present once or twice in every piece.  Piece k's left
    marker (head) must end one column right of piece k-1's right marker
    (tail), in the same row; every marker then takes its own piece's colour.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = Counter(I.reshape(-1).tolist())
    ranked = cnt.most_common()
    if len(ranked) < 2 or ranked[1][1] == ranked[0][1]:
        return None
    bgc = ranked[0][0]
    comps = _components(I, bgc)
    n = len(comps)
    if n < 2:
        return None
    comps.sort(key=lambda cs: min(c for (_, c) in cs))
    cands = [col for col in sorted(cnt) if col != bgc and
             all(sum(1 for p in cs if I[p] == col) in (1, 2) for cs in comps)]
    if not cands:
        return None
    dotc = min(cands, key=lambda col: (cnt[col], col))

    pieces = []
    for k, cs in enumerate(comps):
        cs = [(int(r), int(c)) for (r, c) in cs]
        dots = sorted([p for p in cs if I[p] == dotc], key=lambda p: (p[1], p[0]))
        need = (1 if k > 0 else 0) + (1 if k < n - 1 else 0)
        if len(dots) != need:
            return None
        if need == 2 and dots[0][1] == dots[1][1]:
            return None
        head = dots[0] if k > 0 else None
        tail = dots[-1] if k < n - 1 else None
        body = Counter(int(I[p]) for p in cs if I[p] != dotc)
        if len(body) != 1:
            return None
        pieces.append({"cells": cs, "head": head, "tail": tail,
                       "col": next(iter(body))})

    offs = [(0, 0)]
    for k in range(1, n):
        pt = pieces[k - 1]["tail"]
        po = offs[k - 1]
        hd = pieces[k]["head"]
        offs.append((pt[0] + po[0] - hd[0], pt[1] + po[1] + 1 - hd[1]))
    final = {}
    for pc, (dr, dc) in zip(pieces, offs):
        for (r, c) in pc["cells"]:
            q = (r + dr, c + dc)
            if q in final or not (0 <= q[0] < h) or q[1] < 0:
                return None
            final[q] = pc["col"]
    W = max(c for (_, c) in final) + 1
    out = np.full((h, W), bgc, dtype=int)
    for (r, c), v in final.items():
        out[r, c] = v
    for k in range(1, n):
        if offs[k][1] >= 0:      # every later piece must close a gap leftwards
            return None
    return {"bgc": int(bgc), "dotc": int(dotc), "pieces": pieces,
            "offs": offs, "W": int(W), "out": out}


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    rest = [c for c in range(1, 10) if c != bgc]   # foreground never 0
    dotc = random.choice(rest)
    remcols = [c for c in rest if c != dotc]
    k = random.randint(1, len(remcols))
    pool = random.sample(remcols, k)
    return {"bgc": bgc, "dotc": dotc, "pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, pool=None, **kwargs) -> dict:
    if bgc is None or dotc is None or pool is None:
        ck = sample_colors()
        bgc, dotc, pool = ck["bgc"], ck["dotc"], ck["pool"]
    pool = list(pool)
    hmax = max(5, min(30, int(max_h)))
    wlim = max(8, min(30, int(max_w)))
    wcap = max(6, min(20, wlim - 2))

    for _attempt in range(6000):
        h = _unifint(diff_lb, diff_ub, (5, hmax))
        w = _unifint(diff_lb, diff_ub, (6, wcap))
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
        if minw > wtop or h > int(max_h):
            continue
        fullw = _unifint(diff_lb, diff_ub, (minw, wtop))
        spacings = [1] * (n - 1)
        for _ in range(fullw - minw):
            i = random.randint(0, n - 1)
            if i < n - 1:
                spacings[i] += 1

        ncols = _unifint(diff_lb, diff_ub, (1, len(pool)))
        ccols = random.sample(pool, ncols)
        segcols = [random.choice(ccols) for _ in objs]

        gi = [[bgc] * fullw for _ in range(h)]
        nextcol = 0
        fail = False
        for i, ob in enumerate(objs):
            colr = segcols[i]
            cells = {}
            for j, p in enumerate(ob):
                if (j == 0 and i > 0) or (j == len(ob) - 1 and i < n - 1):
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
                ulr = random.randint(0, h - hh)
                ulc = nextcol + spacings[i - 1]
            if ulc + ww > fullw:
                fail = True
                break
            nextcol = ulc + ww
            for (pr, pc), v in cells.items():
                gi[ulr + pr - minr][ulc + pc - minc] = v
        if fail:
            continue

        go = [[bgc] * sw for _ in range(h)]
        for i, ob in enumerate(objs):
            for (pr, pc) in ob:
                go[pr][pc] = segcols[i]
        gi_np = np.array(gi, dtype=int)
        go_np = np.array(go, dtype=int)
        m = _measure(gi_np)
        if m is None or m["out"].shape != go_np.shape or not np.array_equal(m["out"], go_np):
            continue
        return {"input": tuple(tuple(int(v) for v in row) for row in gi),
                "output": tuple(tuple(int(v) for v in row) for row in go)}
    raise RuntimeError("234bbc79: could not generate an instance")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    m = _measure(I)                       # everything measured from I alone
    if m is None:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    bgc, pieces, offs, W = m["bgc"], m["pieces"], m["offs"], m["W"]
    n = len(pieces)
    g = I.copy()                          # mirror of the working grid

    def grab_and_slide(cells, step, count):
        """One grab, then empty-selection Moves; mirrors ARCLE on g."""
        op = {(0, -1): 23, (0, 1): 22, (-1, 0): 20, (1, 0): 21}[step]
        obj = {p: int(g[p]) for p in cells if g[p] != 0}
        snap = g.copy()
        for p in cells:
            snap[p] = 0
        for i in range(count):
            ops.append(op)
            sels.append(sel_of(cells) if i == 0 else sel_of([]))
        g[:] = snap
        for (r, c), v in obj.items():
            rr, cc = r + step[0] * count, c + step[1] * count
            if 0 <= rr < hi and 0 <= cc < wi:
                g[rr, cc] = v

    work = [pc["col"] for pc in pieces]

    def movable(k, dr, dc):
        # the cells ARCLE can actually carry: everything but literal-0 markers
        return [(r + dr, c + dc) for (r, c) in pieces[k]["cells"] if I[r, c] != 0]

    # Phase 1 — close the gaps: everything from piece k rightwards is one band
    # that slides left until piece k's head sits just right of piece k-1's tail.
    hc = [0] * n
    for k in range(1, n):
        target = pieces[k - 1]["tail"][1] + hc[k - 1] + 1
        gap = pieces[k]["head"][1] + hc[k] - target
        if gap <= 0:
            continue
        band = [p for j in range(k, n) for p in movable(j, 0, hc[j])]
        grab_and_slide(band, (0, -1), gap)
        for j in range(k, n):
            hc[j] -= gap

    # Phase 2 — line up rows: each piece slides vertically so its head is level
    # with the previous piece's tail.
    for k in range(1, n):
        dr = offs[k][0]
        if dr:
            cells = movable(k, 0, hc[k])
            grab_and_slide(cells, (1 if dr > 0 else -1, 0), abs(dr))

    # Phase 3 — the touching markers become body: one Color per piece colour.
    bycol = {}
    for k, pc in enumerate(pieces):
        dr, dc = offs[k]
        for key in ("head", "tail"):
            if pc[key] is not None:
                r, c = pc[key]
                bycol.setdefault(work[k], []).append((r + dr, c + dc))
    for col in sorted(bycol):
        cells = [p for p in bycol[col] if g[p] != col]
        if cells:
            ops.append(col)
            sels.append(sel_of(cells))
            for p in cells:
                g[p] = col

    # Phase 4 — drop the emptied right-hand columns (crop is transparent).
    if W != wi:
        ops.append(33)
        sels.append([0, 0, hi - 1, W - 1])
    g = g[:, :W]

    # Phase 5 — the slides left their vacated cells at 0: refill them with bgc.
    if bgc != 0:
        holes = [(r, c) for r in range(hi) for c in range(W) if g[r, c] == 0]
        if holes:
            ops.append(bgc)
            sels.append(sel_of(holes))
            for p in holes:
                g[p] = bgc
    ops.append(34)
    sels.append([0, 0, hi - 1, W - 1])
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
