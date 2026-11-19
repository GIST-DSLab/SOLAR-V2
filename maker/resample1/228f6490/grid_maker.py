"""
ARC Task: 228f6490 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, _tries=60) -> dict:
    def simulate(I, ops, sels):
        g = np.array(I, dtype=int); grab = None
        for op, s in zip(ops, sels):
            if op == 34:
                break
            cells = s["cells"] if isinstance(s, dict) else [
                (r, c) for r in range(s[0], s[0] + s[2] + 1) for c in range(s[1], s[1] + s[3] + 1)]
            if op < 10:
                for r, c in cells:
                    g[r, c] = op
            elif 20 <= op <= 23:
                d = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                if cells:
                    bg = g.copy()
                    objc = {(r, c): g[r, c] for r, c in cells if g[r, c] != 0}
                    for r, c in cells:
                        bg[r, c] = 0
                    grab = [bg, objc, (0, 0)]
                bg, objc, off = grab
                off = (off[0] + d[0], off[1] + d[1]); grab[2] = off
                g = bg.copy()
                for (r, c), v in objc.items():
                    x, y = r + off[0], c + off[1]
                    if 0 <= x < g.shape[0] and 0 <= y < g.shape[1]:
                        g[x, y] = v
        return g

    def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, sqc):
        def unifint(lb, ub, bounds):
            a, b = bounds
            if b < a:
                b = a
            d = random.uniform(lb, ub)
            return min(max(a, round(a + (b - a) * d)), b)

        def dnb(cells):
            out = set()
            for (i, j) in cells:
                out |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            return out

        def normalize(cells):
            mi = min(i for i, j in cells); mj = min(j for i, j in cells)
            return frozenset((i - mi, j - mj) for i, j in cells)

        def shift(cells, d):
            return frozenset((i + d[0], j + d[1]) for i, j in cells)

        def rect(hh, ww, off=(0, 0)):
            return frozenset((i + off[0], j + off[1]) for i in range(hh) for j in range(ww))

        cols = list(range(10))
        if bgc is None:
            bgc = random.choice(cols)
        if sqc is None:
            sqc = random.choice([c for c in cols if c != bgc])
        remcols = [c for c in cols if c != bgc and c != sqc]
        h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
        w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
        nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]

        def fill(g, c, cells):
            for i, j in cells:
                if 0 <= i < h and 0 <= j < w:
                    g[i][j] = c

        inds = set(rect(h, w))
        forbidden = []
        succ = 0; tr = 0; maxtr = 5 * nsq
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = random.randint(3, 6); ow = random.randint(3, 6)
            bd = rect(oh, ow)
            bounds = rect(oh - 2, ow - 2, (1, 1))
            obj = {random.choice(sorted(bounds))}
            ncells = random.randint(1, (oh - 2) * (ow - 2))
            for _ in range(ncells - 1):
                obj.add(random.choice(sorted((bounds - obj) & dnb(obj))))
            sqcands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not sqcands:
                continue
            loc = random.choice(sorted(sqcands))
            bdplcd = shift(bd, loc)
            if bdplcd.issubset(inds):
                tmpinds = inds - bdplcd
                inobjn = normalize(obj)
                ih = max(i for i, j in inobjn) + 1; iw = max(j for i, j in inobjn) + 1
                cands = [ij for ij in inds if ij[0] <= h - ih and ij[1] <= w - iw]
                if not cands:
                    continue
                loc2 = random.choice(sorted(cands))
                inobjplcd = shift(inobjn, loc2)
                bdnorm = bd - frozenset(obj)
                if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                    forbidden.append(bdnorm); forbidden.append(inobjn)
                    succ += 1
                    inds = (inds - (bdplcd | inobjplcd)) - dnb(inobjplcd)
                    col = random.choice(remcols)
                    oplcd = shift(obj, loc)
                    fill(gi, sqc, bdplcd - oplcd)
                    fill(go, sqc, bdplcd)
                    fill(go, col, oplcd)
                    fill(gi, col, inobjplcd)
        nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
        succ = 0; tr = 0; maxtr = 10 * nremobjs
        while tr < maxtr and succ < nremobjs:
            tr += 1
            oh = random.randint(1, 4); ow = random.randint(1, 4)
            bounds = rect(oh, ow)
            obj = {random.choice(sorted(bounds))}
            ncells = random.randint(1, oh * ow)
            for _ in range(ncells - 1):
                obj.add(random.choice(sorted((bounds - obj) & dnb(obj))))
            obj = normalize(obj)
            if obj in forbidden:
                continue
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            plcd = shift(obj, loc)
            if plcd.issubset(inds):
                succ += 1
                inds = (inds - plcd) - dnb(plcd)
                col = random.choice(remcols)
                fill(gi, col, plcd); fill(go, col, plcd)
        return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}

    for _ in range(_tries):
        ex = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, sqc)
        try:
            ops, sels = derive_operations(ex["input"], ex["output"])
            if (simulate(ex["input"], ops, sels) == np.array(ex["output"])).all():
                return ex
        except Exception:
            pass
    return ex


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def comps(diag):
        seen = np.zeros_like(I, dtype=bool)
        res = []
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diag:
            nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for r in range(hi):
            for c in range(wi):
                if seen[r, c]:
                    continue
                col = I[r, c]
                stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    a, b = stack.pop(); cells.append((a, b))
                    for da, db in nb:
                        x, y = a + da, b + db
                        if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True; stack.append((x, y))
                res.append((int(col), cells))
        return res

    def norm(cells):
        mi = min(r for r, c in cells); mj = min(c for r, c in cells)
        return frozenset((r - mi, c - mj) for r, c in cells)

    allc = comps(True)
    holes = []
    for col, cells in allc:
        if col != bgc:
            continue
        if any(r in (0, hi - 1) or c in (0, wi - 1) for r, c in cells):
            continue
        cs = set(cells)
        ring = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    x, y = r + dr, c + dc
                    if 0 <= x < hi and 0 <= y < wi and (x, y) not in cs:
                        ring.add((x, y))
        sc = Counter(int(I[x, y]) for x, y in ring).most_common(1)[0][0]
        holes.append((sc, cells))
    frame_col = None
    if examples:
        votes = Counter()
        for EI, EO in examples:
            EI = np.asarray(EI, dtype=int); EO = np.asarray(EO, dtype=int)
            if EI.shape != EO.shape:
                continue
            eb = Counter(EI.flatten().tolist()).most_common(1)[0][0]
            eh, ew = EI.shape
            for r, c in zip(*np.nonzero((EI == eb) & (EO != eb))):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    x, y = r + dr, c + dc
                    if 0 <= x < eh and 0 <= y < ew and EI[x, y] != eb and EO[x, y] == EI[x, y]:
                        votes[int(EI[x, y])] += 1
        if votes:
            frame_col = votes.most_common(1)[0][0]
    if holes and (frame_col is None or frame_col not in {sc for sc, _ in holes}):
        frame_col = Counter(sc for sc, _ in holes).most_common(1)[0][0]
    if holes:
        holes = [cells for sc, cells in holes if sc == frame_col]
    fg = [(col, cells) for col, cells in allc if col != bgc]
    fg4 = [(col, cells) for col, cells in comps(False) if col != bgc]

    ops, sels = [], []
    used = set()
    G = I.copy()  # working grid as it stands
    for hole in holes:
        hn = norm(hole)
        match = None
        for pool in (fg, fg4):
            for k, (col, cells) in enumerate(pool):
                if norm(cells) == hn and frozenset(cells) not in used:
                    match = (col, cells); break
            if match:
                break
        if match is None:
            continue
        col, src = match
        used.add(frozenset(src))
        dr = min(r for r, c in hole) - min(r for r, c in src)
        dc = min(c for r, c in hole) - min(c for r, c in src)
        if col == 0:
            # a colour-0 object cannot be grabbed by Move (0 is "empty"); paint it
            ops.append(bgc); sels.append(sel_of(src))
            ops.append(0); sels.append(sel_of(hole))
            for r, c in src:
                G[r, c] = bgc
            for r, c in hole:
                G[r, c] = 0
            continue
        snap = G.copy()
        for r, c in src:
            snap[r, c] = 0
        r0 = min(r for r, c in src); c0 = min(c for r, c in src)
        rel = [(r - r0, c - c0) for r, c in src]
        oh = max(r for r, c in rel) + 1; ow = max(c for r, c in rel) + 1

        def grid_at(p):
            g = snap.copy()
            for r, c in rel:
                g[r0 + p[0] + r, c0 + p[1] + c] = col
            return g

        # shortest slide in which every unit step visibly changes the grid and
        # no grid state repeats
        from collections import deque
        MOVES = ((20, -1, 0), (21, 1, 0), (22, 0, 1), (23, 0, -1))
        start = (0, 0); goal = (dr, dc)
        par = {start: None}
        seen = {G.tobytes()}
        dq = deque([start])
        while dq and goal not in par:
            p = dq.popleft()
            order = sorted(MOVES, key=lambda m: abs(goal[0] - p[0] - m[1]) + abs(goal[1] - p[1] - m[2]))
            for m in order:
                q = (p[0] + m[1], p[1] + m[2])
                if q in par:
                    continue
                if not (0 <= r0 + q[0] and r0 + q[0] + oh <= hi and 0 <= c0 + q[1] and c0 + q[1] + ow <= wi):
                    continue
                key = grid_at(q).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                par[q] = (p, m)
                dq.append(q)
        if goal not in par:
            # no visible slide exists: erase and repaint
            ops.append(bgc); sels.append(sel_of(src))
            ops.append(col); sels.append(sel_of(hole))
            for r, c in src:
                G[r, c] = bgc
            for r, c in hole:
                G[r, c] = col
            continue
        steps = []
        node = goal
        while par[node] is not None:
            node, m = par[node]
            steps.append(m)
        steps.reverse()
        cur = [(r + dr, c + dc) for r, c in src]
        G = grid_at(goal)
        if bgc != 0:
            for r, c in sorted(set(src) - set(cur)):
                G[r, c] = bgc
        cur = list(src)
        first = True
        for op, sr, sc_ in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc_) for r, c in cur]
        holecells = sorted(set(src) - set(cur))
        if bgc != 0 and holecells:
            ops.append(bgc); sels.append(sel_of(holecells))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 228f6490"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 228f6490"
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
                                f"for task 228f6490"
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
                    f"Failed to build a complete episode for task 228f6490 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"228f6490-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
