"""
ARC Task: 264363fd (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, linc = random.sample(cols, 3)
    rem = [c for c in cols if c not in (bgc, sqc, linc)]
    return {"bgc": bgc, "sqc": sqc, "linc": linc,
            "cpcol": random.choice(rem), "nbhcol": random.choice(rem)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, linc=None,
             cpcol=None, nbhcol=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        x = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * x))))

    def nbs8(p):
        r, c = p
        return {(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)}

    def shoot(s, d):
        return {(s[0] + k * d[0], s[1] + k * d[1]) for k in range(42)}

    cols = list(range(10))
    if bgc is None or sqc is None or linc is None:
        bgc, sqc, linc = random.sample(cols, 3)
    rem = [c for c in cols if c not in (bgc, sqc, linc)]
    if cpcol is None:
        cpcol = random.choice(rem)
    if nbhcol is None:
        nbhcol = random.choice(rem)
    cp = (2, 2)
    neighs = nbs8(cp)
    o1 = frozenset({(1, 2), (0, 2)})
    o2 = frozenset({(2, 1), (2, 0)})
    o3 = frozenset({(3, 2), (4, 2)})
    o4 = frozenset({(2, 3), (2, 4)})
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    nspikes = random.randint(1, 4)
    spikes = random.sample([o1, o2, o3, o4], nspikes)
    lns = set().union(*spikes)
    obj = {cp: cpcol}
    for p in lns:
        obj[p] = linc
    for p in neighs - lns:
        obj[p] = nbhcol
    loci = random.randint(0, h - 5)
    locj = random.randint(0, w - 5)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for (r, c), v in obj.items():
        gi[r + loci][c + locj] = v
    objplaced = {(r + loci, c + locj) for (r, c) in obj}
    near = set()
    for p in objplaced:
        near |= nbs8(p)
    inds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc} - near
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 100))
    succ = tr = 0
    maxtr = 10 * numsq
    while succ < numsq and tr < maxtr:
        tr += 1
        gh = random.randint(5, h // 2 + 1)
        gw = random.randint(5, w // 2 + 1)
        cands = [p for p in inds if p[0] <= h - gh and p[1] <= w - gw]
        if not cands:
            continue
        lr, lc = random.choice(sorted(cands))
        g1 = [[sqc] * gw for _ in range(gh)]
        g2 = [[sqc] * gw for _ in range(gh)]
        ginds = {(r, c) for r in range(gh) for c in range(gw)}
        gfull = set(ginds)
        bck = {(r + lr, c + lc) for (r, c) in gfull}
        if not bck <= inds:
            continue
        noccs = unifint(diff_lb, diff_ub, (1, (gh * gw) // 25))
        s2 = t2 = 0
        while s2 < noccs and t2 < 5 * noccs:
            t2 += 1
            c2 = [p for p in ginds if p[0] <= gh - 5 and p[1] <= gw - 5]
            if not c2:
                break
            l2 = random.choice(sorted(c2))
            lns2 = set()
            for sp in spikes:
                d = mpr[sp]
                lns2 |= shoot((cp[0] + l2[0] + d[0], cp[1] + l2[1] + d[1]), d)
            lns2 &= gfull
            plc = {(r + l2[0], c + l2[1]): v for (r, c), v in obj.items()}
            pi = set(plc)
            linccells = {(r, c) for r in range(gh) for c in range(gw) if g2[r][c] == linc}
            if pi <= ginds and lns2 <= (ginds | linccells) and len(lns2 - pi) > 0:
                s2 += 1
                nn = set()
                for p in pi:
                    nn |= nbs8(p)
                ginds = (ginds - pi) - nn - lns2
                g1[cp[0] + l2[0]][cp[1] + l2[1]] = cpcol
                for (r, c), v in plc.items():
                    g2[r][c] = v
                for (r, c) in lns2:
                    g2[r][c] = linc
        if s2 > 0:
            succ += 1
            ob = {(r - 1, c - 1) for r in range(lr, lr + gh + 2) for c in range(lc, lc + gw + 2)}
            inds = inds - ob
            for r in range(gh):
                for c in range(gw):
                    gi[r + lr][c + lc] = g1[r][c]
                    go[r + lr][c + lc] = g2[r][c]
    if succ == 0:
        raise ValueError("no rectangle placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def components(bg):
        seen = np.zeros((h, w), bool)
        out = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    st = [(r, c)]; seen[r, c] = True; cells = []
                    while st:
                        a, b = st.pop(); cells.append((a, b))
                        for dr, dc in D4:
                            x, y = a + dr, b + dc
                            if 0 <= x < h and 0 <= y < w and not seen[x, y] and I[x, y] != bg:
                                seen[x, y] = True; st.append((x, y))
                    out.append(cells)
        return out

    def ok(bg):
        cs = components(bg)
        if len(cs) < 2:
            return None
        cs.sort(key=len)
        if len(cs[0]) > 13:
            return None
        for comp in cs[1:]:
            rs = [r for r, _ in comp]; cc = [c for _, c in comp]
            hh = max(rs) - min(rs) + 1; ww = max(cc) - min(cc) + 1
            if hh < 5 or ww < 5 or len(comp) != hh * ww:
                return None
        return cs

    # background: the canvas the key and rectangles sit on (confirmed structurally)
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    comps = None
    for cand, _ in cnt.most_common():
        got = ok(cand)
        if got is not None:
            bgc, comps = cand, got
            break
    if comps is None:
        comps = sorted(components(bgc), key=len)

    ops, sels = [], []
    G = I.copy()
    full = [(r, c) for r in range(h) for c in range(w)]
    if len(comps) < 2:
        return [34], [sel_of(full)]

    key = comps[0]
    keyset = set(key)
    # key hub: the only key cell whose full 8-neighbourhood is part of the key
    hub = None
    for (r, c) in key:
        if all((r + i, c + j) in keyset for i in (-1, 0, 1) for j in (-1, 0, 1)):
            hub = (r, c); break
    if hub is None:
        return [34], [sel_of(full)]
    hr, hc = hub
    core = [(hr + i, hc + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    zero_offs = [(r - hr, c - hc) for (r, c) in core if I[r, c] == 0]
    arms = [d for d in D4 if (hr + 2 * d[0], hc + 2 * d[1]) in keyset]
    linc = int(I[hr + arms[0][0] * 2, hc + arms[0][1] * 2]) if arms else None

    def paint(col, cells):
        cells = [(r, c) for r, c in cells if G[r, c] != col]
        if not cells:
            return
        for r, c in cells:
            G[r, c] = col
        ops.append(int(col)); sels.append(sel_of(cells))

    rects = []
    for comp in comps[1:]:
        rs = [r for r, _ in comp]; cc = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cc), max(cc)
        sq = Counter(int(I[r, c]) for r, c in comp).most_common(1)[0][0]
        marks = sorted((r, c) for r, c in comp if I[r, c] != sq)
        rects.append((r0, c0, r1, c1, marks))
    rects.sort()
    copied = False
    for (r0, c0, r1, c1, marks) in rects:
        for (mr, mc) in marks:
            if not copied:
                # pick the key's 3x3 core up once (clipboard keeps it for every stamp)
                ops.append(28); sels.append(sel_of(core)); copied = True
            # stamp the key core centred on the marker
            before = G.copy()
            for i in range(3):
                for j in range(3):
                    v = I[hr - 1 + i, hc - 1 + j]
                    if v != 0:
                        G[mr - 1 + i, mc - 1 + j] = v
            if not np.array_equal(before, G):
                ops.append(30); sels.append(sel_of([(mr - 1, mc - 1)]))
            # 0-coloured core cells are transparent to Paste: paint them explicitly
            if zero_offs:
                paint(0, [(mr + a, mc + b) for a, b in zero_offs])
            # grow each arm out to the rectangle edge
            for d in arms:
                line = []
                r, c = mr + 2 * d[0], mc + 2 * d[1]
                while r0 <= r <= r1 and c0 <= c <= c1:
                    line.append((r, c)); r += d[0]; c += d[1]
                paint(linc, line)
    # the key has served as template: wipe it to background
    paint(bgc, key)
    ops.append(34); sels.append(sel_of(full))
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
                        f"num_examples+1 ({num_examples + 1}) for task 264363fd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 264363fd"
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
                                f"for task 264363fd"
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
                    f"Failed to build a complete episode for task 264363fd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"264363fd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
