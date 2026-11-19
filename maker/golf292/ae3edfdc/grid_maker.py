"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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

def _neighbors(loc):
    i, j = loc
    return {(i+a, j+b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3, 7)]
    return {"bgc": random.choice(cols)}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 7)])
    sgn = lambda v: (v > 0) - (v < 0)
    def uf(lo_hi):
        a, b = lo_hi
        if b <= a:
            return a
        lo = a + (b - a) * diff_lb
        hi = a + (b - a) * diff_ub
        lo, hi = int(round(lo)), int(round(hi))
        lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
        return random.randint(lo, hi)
    h = uf((8, max(8, max_h)))
    w = uf((8, max(8, max_w)))
    go = [[bgc] * w for _ in range(h)]
    rd = (random.randint(1, h - 2), random.randint(1, w - 2))
    excl = {rd} | _neighbors(rd)
    reminds = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1) if (i, j) not in excl]
    bd = random.choice(reminds)
    go[rd[0]][rd[1]] = 2
    go[bd[0]][bd[1]] = 1
    gd = random.sample(sorted(_neighbors(rd)), uf((1, 8)))
    od = random.sample(sorted(_neighbors(bd)), uf((1, 8)))
    for (i, j) in gd:
        go[i][j] = 3
    for (i, j) in od:
        go[i][j] = 7
    pos = lambda a, b: (sgn(b[0] - a[0]), sgn(b[1] - a[1]))
    mpr = {d: (3, pos(rd, d)) for d in gd}
    mpr.update({d: (7, pos(bd, d)) for d in od})
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = uf((1, ub))
    gi = [row[:] for row in go]
    for _ in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = loc[0] + direc[0], loc[1] + direc[1]
            if 0 <= ii < h and 0 <= jj < w and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if not options:
            break
        loc, col, direc = random.choice(options)
        del mpr[loc]
        newloc = (loc[0] + direc[0], loc[1] + direc[1])
        mpr[newloc] = (col, direc)
        gi[loc[0]][loc[1]] = bgc
        gi[newloc[0]][newloc[1]] = col
    return {"input": gi, "output": go}

def derive_operations(I, O, examples=None):
    """7-dots slide radially back to the cell next to the 1, 3-dots to the cell
    next to the 2.  Dots sharing a slide direction are grabbed together and slid
    in one stroke; a dot is dropped from the group once it is home."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    sgn = lambda v: (v > 0) - (v < 0)

    cnt = Counter(I.flatten().tolist())
    cands = [c for c in cnt if c not in (1, 2, 3, 7)]
    bgc = max(cands, key=lambda c: cnt[c]) if cands else 0

    p1 = np.argwhere(I == 1)
    p2 = np.argwhere(I == 2)
    c1 = (int(p1[0][0]), int(p1[0][1])) if len(p1) else None
    c2 = (int(p2[0][0]), int(p2[0][1])) if len(p2) else None

    dots = []
    for col, ctr in ((7, c1), (3, c2)):
        if ctr is None:
            continue
        for cell in np.argwhere(I == col):
            r, c = int(cell[0]), int(cell[1])
            dst = (ctr[0] + sgn(r - ctr[0]), ctr[1] + sgn(c - ctr[1]))
            dots.append({"src": (r, c), "dst": dst, "col": col})

    # the rule's result, computed from I alone (used only to validate a plan)
    expect = I.copy()
    for d in dots:
        expect[d["src"]] = bgc
    for col in (7, 3):
        for d in dots:
            if d["col"] == col:
                expect[d["dst"]] = col

    MOVE = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}

    def simulate(ops, sels):
        g = I.copy()
        snap, obj = None, None
        for op, s in zip(ops, sels):
            before = g.copy()
            cells = [tuple(x) for x in s["cells"]]
            if op in (20, 21, 22, 23):
                dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                if cells:
                    snap = g.copy()
                    obj = {}
                    for (r, c) in cells:
                        if g[r, c] != 0:
                            obj[(r, c)] = g[r, c]
                        snap[r, c] = 0
                obj = {(r + dr, c + dc): v for (r, c), v in obj.items()}
                g = snap.copy()
                for (r, c), v in obj.items():
                    if 0 <= r < hi and 0 <= c < wi:
                        g[r, c] = v
            else:
                for (r, c) in cells:
                    g[r, c] = op
            if np.array_equal(before, g):
                return None
        return g

    def plan_grouped(phases):
        ops, sels = [], []
        cur = {i: d["src"] for i, d in enumerate(dots)}
        for (ar, ac) in phases:
            ax = 0 if ar else 1
            sgn_ax = ar if ar else ac
            # dots whose slide still needs steps along this axis in this sense
            need = {}
            for i, d in enumerate(dots):
                delta = d["dst"][ax] - cur[i][ax]
                if delta != 0 and sgn(delta) == sgn_ax:
                    need[i] = abs(delta)
            prev = None
            step = 0
            while True:
                group = frozenset(i for i, n in need.items() if n > step)
                if not group:
                    break
                ops.append(MOVE[(ar, ac)])
                if group == prev:
                    sels.append(sel_of([]))          # same object still grabbed
                else:
                    sels.append(sel_of([cur[i] for i in group]))  # grab group
                for i in group:
                    cur[i] = (cur[i][0] + ar, cur[i][1] + ac)
                prev = group
                step += 1
        return ops, sels

    def plan_single():
        pending = [d for d in dots if d["dst"] != d["src"]]
        frozen = {d["src"] for d in dots if d["dst"] == d["src"]}
        order = []
        while pending:
            blocked = frozen | {d["src"] for d in pending}
            pick = None
            for d in pending:
                if d["dst"] not in (blocked - {d["src"]}):
                    pick = d
                    break
            if pick is None:
                pick = sorted(pending, key=lambda d: 0 if d["col"] == 7 else 1)[0]
            pending.remove(pick)
            order.append(pick)
        ops, sels = [], []
        for d in order:
            (sr, sc), (tr, tc) = d["src"], d["dst"]
            vr, vc = tr - sr, tc - sc
            steps = []
            for i in range(max(abs(vr), abs(vc))):
                if i < abs(vr):
                    steps.append(20 if vr < 0 else 21)
                if i < abs(vc):
                    steps.append(23 if vc < 0 else 22)
            for k, mop in enumerate(steps):
                ops.append(mop)
                sels.append(sel_of([(sr, sc)]) if k == 0 else sel_of([]))
            if bgc != 0:
                ops.append(bgc)
                sels.append(sel_of([(sr, sc)]))
        return ops, sels

    def finish(ops, sels):
        # re-grabbing leaves 0s where each grabbed group started: repaint every
        # 0 cell (0 is never a real colour here when bgc != 0) in one stroke
        if bgc != 0:
            g = simulate(ops, sels) if ops else I.copy()
            if g is None:
                return None
            zeros = [(int(r), int(c)) for r, c in np.argwhere(g == 0)]
            if zeros:
                ops = ops + [bgc]
                sels = sels + [sel_of(zeros)]
        return ops, sels

    best = None
    orders = [
        [(1, 0), (-1, 0), (0, 1), (0, -1)],
        [(0, 1), (0, -1), (1, 0), (-1, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(0, -1), (0, 1), (-1, 0), (1, 0)],
    ]
    for ph in orders:
        res = finish(*plan_grouped(ph))
        if res is None:
            continue
        g = simulate(*res)
        if g is not None and np.array_equal(g, expect):
            if best is None or len(res[0]) < len(best[0]):
                best = res
    if best is None:
        best = plan_single()
    ops, sels = best
    ops = list(ops) + [34]
    sels = list(sels) + [[0, 0, ho - 1, wo - 1]]
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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
