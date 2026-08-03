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
import numpy as np
import random
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1) episode-level colors
#    Task rule: each "box" (solid rectangle of sqc with a hole shaped like some
#    loose object) receives that loose object, which slides in from elsewhere.
#    Structure colors: bgc (canvas) and sqc (box color) -> fixed per episode.
#    Individual object colors carry no rule information (only shape matching
#    matters), so they stay random -- but 0 is excluded from object colors so
#    ARCLE's object-mode Move can actually grab them (0 == "nothing" there).
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


# ----------------------------------------------------------------------------
# 2) generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, **kwargs) -> dict:

    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def n4(cells):
        s = set()
        for (r, c) in cells:
            s |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return s

    def n8(cells):
        s = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        s.add((r + dr, c + dc))
        return s

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    hub = max(6, min(30, int(max_h)))
    wub = max(6, min(30, int(max_w)))
    hlb = min(10, hub)
    wlb = min(10, wub)
    remcols = [c for c in range(10) if c not in (bgc, sqc, 0)]

    gi = go = None
    for _attempt in range(100):
        h = unifint(diff_lb, diff_ub, (hlb, hub))
        w = unifint(diff_lb, diff_ub, (wlb, wub))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        inds = {(i, j) for i in range(h) for j in range(w)}
        forbidden = []

        nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = random.randint(3, min(6, h))
            ow = random.randint(3, min(6, w))
            bd = {(i, j) for i in range(oh) for j in range(ow)}
            bounds = {(i + 1, j + 1) for i in range(oh - 2) for j in range(ow - 2)}
            if not bounds:
                continue
            obj = {random.choice(sorted(bounds))}
            ncells = random.randint(1, len(bounds))
            for _k in range(ncells - 1):
                cands = sorted((bounds - obj) & n4(obj))
                if not cands:
                    break
                obj.add(random.choice(cands))

            sqcands = sorted(ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow)
            if not sqcands:
                continue
            loc = random.choice(sqcands)
            bdplcd = {(i + loc[0], j + loc[1]) for i, j in bd}
            if not bdplcd <= inds:
                continue
            # keep a 1-cell moat around boxes so nothing merges/encloses spuriously
            tmpinds = inds - bdplcd - n8(bdplcd)

            inobjn = norm(obj)
            ohh = max(r for r, _ in inobjn) + 1
            oww = max(c for _, c in inobjn) + 1
            cands2 = sorted(ij for ij in tmpinds if ij[0] <= h - ohh and ij[1] <= w - oww)
            if not cands2:
                continue
            loc2 = random.choice(cands2)
            inobjplcd = {(r + loc2[0], c + loc2[1]) for r, c in inobjn}
            bdnorm = frozenset(bd - obj)
            if inobjplcd <= tmpinds and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = inds - bdplcd - n8(bdplcd) - inobjplcd - n8(inobjplcd)
                col = random.choice(remcols)
                oplcd = {(r + loc[0], c + loc[1]) for r, c in obj}
                for (r, c) in (bdplcd - oplcd):
                    gi[r][c] = sqc
                for (r, c) in bdplcd:
                    go[r][c] = sqc
                for (r, c) in oplcd:
                    go[r][c] = col
                for (r, c) in inobjplcd:
                    gi[r][c] = col

        # distractor objects (never move)
        nrem = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
        succ2 = 0
        tr = 0
        maxtr2 = 10 * nrem
        while tr < maxtr2 and succ2 < nrem:
            tr += 1
            oh = random.randint(1, 4)
            ow = random.randint(1, 4)
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            obj = {random.choice(sorted(bounds))}
            ncells = random.randint(1, oh * ow)
            for _k in range(ncells - 1):
                cands = sorted((bounds - obj) & n4(obj))
                if not cands:
                    break
                obj.add(random.choice(cands))
            objn = norm(obj)
            if objn in forbidden:
                continue
            ohh = max(r for r, _ in objn) + 1
            oww = max(c for _, c in objn) + 1
            cands = sorted(ij for ij in inds if ij[0] <= h - ohh and ij[1] <= w - oww)
            if not cands:
                continue
            loc = random.choice(cands)
            plcd = {(r + loc[0], c + loc[1]) for r, c in objn}
            if plcd <= inds:
                succ2 += 1
                inds = inds - plcd - n8(plcd)
                col = random.choice(remcols)
                for (r, c) in plcd:
                    gi[r][c] = col
                    go[r][c] = col

        if succ >= 1:
            break

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


# ----------------------------------------------------------------------------
# 3) derive_operations
#    Each loose object TRANSLATES into the box hole whose shape it matches.
#    -> expressed as unit Move chains (grab once, then empty selections),
#       plus one bgc repair of the vacated footprint when bgc != 0.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- enclosed background regions (holes inside boxes) ----
    seen = np.zeros((h, w), dtype=bool)
    holes = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] != bgc or seen[r0, c0]:
                continue
            comp = []
            dq = deque([(r0, c0)])
            seen[r0, c0] = True
            touches = False
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == bgc:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            if touches:
                continue
            cs = set(comp)
            ring = set()
            for rr, cc in comp:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cs:
                            ring.add((nr, nc))
            ringcols = {int(I[x, y]) for x, y in ring}
            if len(ringcols) != 1:
                continue                              # not a clean box hole
            encl = ringcols.pop()
            fills = {int(O[x, y]) for x, y in comp}
            if len(fills) != 1 or fills == {bgc}:
                continue                              # hole is not filled -> not a target
            holes.append((sorted(comp), encl, fills.pop()))

    if holes:
        common_encl = Counter(e for _, e, _ in holes).most_common(1)[0][0]
        holes = [x for x in holes if x[1] == common_encl]

    # ---- objects that vanish (the pieces that slide into the holes) ----
    seen2 = np.zeros((h, w), dtype=bool)
    movers = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] == bgc or seen2[r0, c0]:
                continue
            col = int(I[r0, c0])
            comp = []
            dq = deque([(r0, c0)])
            seen2[r0, c0] = True
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen2[nr, nc] and I[nr, nc] == col:
                            seen2[nr, nc] = True
                            dq.append((nr, nc))
            if all(O[x, y] == bgc for x, y in comp):
                movers.append((sorted(comp), col))

    def normkey(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    # ---- pair hole <-> object by identical normalized shape (+ color) ----
    pairs = []
    used = set()
    for comp, _encl, fillcol in sorted(holes, key=lambda x: (x[0][0][0], x[0][0][1])):
        key = normkey(comp)
        pick = None
        for idx, (cells, col) in enumerate(movers):
            if idx in used or col != fillcol:
                continue
            if normkey(cells) == key:
                pick = idx
                break
        if pick is None:
            for idx, (cells, col) in enumerate(movers):
                if idx in used:
                    continue
                if normkey(cells) == key:
                    pick = idx
                    break
        if pick is None:
            continue
        used.add(pick)
        pairs.append((movers[pick][0], movers[pick][1], comp))

    ops, sels = [], []
    g = I.copy()

    def simulate(grid, src, col, steps):
        """ARCLE object-mode move: snapshot = grid with src zeroed, object re-pasted."""
        snap = grid.copy()
        for (r, c) in src:
            snap[r, c] = 0
        cur = list(src)
        prev = grid
        ok = True
        for _op, (dr, dc) in steps:
            cur = [(r + dr, c + dc) for r, c in cur]
            ng = snap.copy()
            for (r, c) in cur:
                if 0 <= r < h and 0 <= c < w:
                    ng[r, c] = col
            if np.array_equal(ng, prev):
                ok = False
            prev = ng
        return ok, prev, cur

    for src, col, dst in pairs:
        sr = min(r for r, _ in src)
        sc = min(c for _, c in src)
        dr_tot = min(r for r, _ in dst) - sr
        dc_tot = min(c for _, c in dst) - sc

        vsteps = [((21 if dr_tot > 0 else 20), (1 if dr_tot > 0 else -1, 0))] * abs(dr_tot)
        hsteps = [((22 if dc_tot > 0 else 23), (0, 1 if dc_tot > 0 else -1))] * abs(dc_tot)

        chosen = None
        for steps in (vsteps + hsteps, hsteps + vsteps):
            if not steps:
                break
            ok, final, cur = simulate(g, src, col, steps)
            if ok:
                chosen = (steps, final, cur)
                break

        if chosen is not None:
            steps, final, cur = chosen
            ops.append(steps[0][0])
            sels.append(sel_of(src))                     # grab the object itself
            for op, _d in steps[1:]:
                ops.append(op)
                sels.append(sel_of([]))                  # empty -> keep same object grabbed
            g = final
            # only the ORIGINAL footprint is left at 0; the path was restored by ARCLE
            hole_left = sorted(set(src) - set(cur))
            repair = [(r, c) for (r, c) in hole_left if g[r, c] != bgc]
            if repair:
                ops.append(int(bgc))
                sels.append(sel_of(repair))
                for (r, c) in repair:
                    g[r, c] = bgc
        else:
            # degenerate case (no visible change possible per step): paint directly
            ops.append(int(col))
            sels.append(sel_of(dst))
            for (r, c) in dst:
                g[r, c] = col
            leftover = [(r, c) for (r, c) in src if (r, c) not in set(dst) and g[r, c] != bgc]
            if leftover:
                ops.append(int(bgc))
                sels.append(sel_of(leftover))
                for (r, c) in leftover:
                    g[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox = whole grid rectangle (submit)
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
