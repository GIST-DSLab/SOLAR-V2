"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # fallback when helper module is unavailable
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ----------------------------------------------------------------------------- 1
VARIANTS = [
    {"diag": True,  "rot": False},
    {"diag": True,  "rot": True},
    {"diag": False, "rot": False},
    {"diag": False, "rot": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)                       # the intruding ("blot") color
    pool = [c for c in cols if c != fixc]            # pattern colors, fixed order
    random.shuffle(pool)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"fixc": fixc, "ccols_pool": pool, "instance_plan": plan}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, fixc, ccols_pool, diag=None, rot=None) -> dict:
    if diag is None or rot is None:
        v = random.choice(VARIANTS)
        diag = v["diag"] if diag is None else diag
        rot = v["rot"] if rot is None else rot

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    hb = max(7, min(30, max_w if rot else max_h))
    wb = max(7, min(30, max_h if rot else max_w))
    h = unifint(diff_lb, diff_ub, (7, hb))
    w = unifint(diff_lb, diff_ub, (7, wb))
    pmax = max(2, min(9, h // 3, w // 3))
    p = unifint(diff_lb, diff_ub, (2, pmax))
    ccols = list(ccols_pool[:p])

    if diag:
        g = [[ccols[(c + r) % p] for c in range(w)] for r in range(h)]
    else:
        g = [[ccols[c % p] for c in range(w)] for r in range(h)]

    def acceptable(gr):
        cnt = 0
        for r in range(h):
            for c in range(w - p + 1):
                if all(gr[r][c + j] == ccols[j] for j in range(p)):
                    cnt += 1
                    if cnt > 1:
                        break
            if cnt > 1:
                break
        if cnt <= 1:
            return False
        if not any(all(v != fixc for v in row) for row in gr):
            return False
        if not any(all(gr[r][c] != fixc for r in range(h)) for c in range(w)):
            return False
        return True

    def try_block(cur):
        oh = min(unifint(diff_lb, diff_ub, (2, 5)), h)
        ow = min(unifint(diff_lb, diff_ub, (2, 5)), w)
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        tmp = [row[:] for row in cur]
        for r in range(loci, loci + oh):
            for c in range(locj, locj + ow):
                tmp[r][c] = fixc
        return tmp

    gi = [row[:] for row in g]
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        tmp = try_block(gi)
        if acceptable(tmp):
            gi = tmp
            succ += 1
        tr += 1

    extra = 0
    while succ == 0 and extra < 400:
        tmp = try_block(gi)
        if acceptable(tmp):
            gi = tmp
            succ = 1
        extra += 1
    if succ == 0:
        raise ValueError("could not place any blot")

    go = [row[:] for row in g]
    if rot:
        gi = [list(r) for r in zip(*gi[::-1])]
        go = [list(r) for r in zip(*go[::-1])]

    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------- 3
def _min_period(seq):
    n = len(seq)
    for d in range(1, n + 1):
        if all(seq[i] == seq[i + d] for i in range(n - d)):
            return d
    return n


def _reconstruct(I, x):
    """Rebuild the underlying periodic pattern assuming `x` is the intruding color."""
    H, W = I.shape
    rows = [r for r in range(H) if not np.any(I[r] == x)]
    cols = [c for c in range(W) if not np.any(I[:, c] == x)]
    if not rows or not cols:
        return None
    R, C = rows[0], cols[0]
    P = np.zeros((H, W), dtype=int)
    hp = _min_period([int(v) for v in I[R]])
    if hp > 1:
        f = [int(v) for v in I[R][:hp]]
        if len(set(f)) != hp:
            return None
        for r in range(H):
            s = None
            for cand in range(hp):
                if f[(C - cand) % hp] == int(I[r, C]):
                    s = cand
                    break
            if s is None:
                return None
            for c in range(W):
                P[r, c] = f[(c - s) % hp]
    else:
        vp = _min_period([int(v) for v in I[:, C]])
        if vp <= 1:
            return None
        gcol = [int(v) for v in I[:vp, C]]
        if len(set(gcol)) != vp:
            return None
        for c in range(W):
            t = None
            for cand in range(vp):
                if gcol[(R - cand) % vp] == int(I[R, c]):
                    t = cand
                    break
            if t is None:
                return None
            for r in range(H):
                P[r, c] = gcol[(r - t) % vp]
    if x in set(P.flatten().tolist()):
        return None
    for r in range(H):
        for c in range(W):
            if int(I[r, c]) != int(P[r, c]) and int(I[r, c]) != x:
                return None
    return P


def _components(cells):
    cs = set(cells)
    seen = set()
    out = []
    for cell in cells:
        if cell in seen:
            continue
        comp = []
        dq = deque([cell])
        seen.add(cell)
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cs and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dq.append((nr, nc))
        out.append(sorted(comp))
    return out


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # --- 1. find the intruding color and rebuild the periodic pattern from I alone
    cands = []
    for x in sorted(set(I.flatten().tolist())):
        P2 = _reconstruct(I, x)
        if P2 is not None:
            cands.append((x, P2))
    P = None
    for x, P2 in cands:
        if np.array_equal(P2, O):
            P = P2
            break
    if P is None and cands:
        P = cands[0][1]
    if P is None:
        P = O.copy()

    noise = [(r, c) for r in range(H) for c in range(W) if int(I[r, c]) != int(P[r, c])]
    ops, sels = [], []
    if not noise:
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    nc = int(I[noise[0][0], noise[0][1]])
    grid = I.copy()

    # --- 2. if the pattern uses colour 0, the (transparent) Paste cannot write it,
    #        so wipe each blot to 0 first: base layer, then the replication on top.
    if nc != 0 and 0 in set(P.flatten().tolist()):
        for comp in _components(noise):
            ops.append(0)
            sels.append(sel_of(comp))
            for (r, c) in comp:
                grid[r, c] = 0

    def h_changes(r, c0, clip):
        for i, v in enumerate(clip):
            cc = c0 + i
            if cc >= W:
                break
            if v != 0 and int(grid[r, cc]) != v:
                return True
        return False

    def h_apply(r, c0, clip):
        for i, v in enumerate(clip):
            cc = c0 + i
            if cc >= W:
                break
            if v != 0:
                grid[r, cc] = v

    def v_changes(c, r0, clip):
        for i, v in enumerate(clip):
            rr = r0 + i
            if rr >= H:
                break
            if v != 0 and int(grid[rr, c]) != v:
                return True
        return False

    def v_apply(c, r0, clip):
        for i, v in enumerate(clip):
            rr = r0 + i
            if rr >= H:
                break
            if v != 0:
                grid[rr, c] = v

    clean_rows = [r for r in range(H) if all(int(I[r, c]) == int(P[r, c]) for c in range(W))]
    clean_cols = [c for c in range(W) if all(int(I[r, c]) == int(P[r, c]) for r in range(H))]
    R = clean_rows[0] if clean_rows else 0
    C = clean_cols[0] if clean_cols else 0

    hp = _min_period([int(v) for v in P[R]])

    if hp > 1:
        # ---- the pattern repeats horizontally: every row is a shift of the clean row R
        f = [int(v) for v in P[R][:hp]]
        shifts = {}
        dirty_rows = sorted({r for r, c in noise})
        for r in dirty_rows:
            s = 0
            for cand in range(hp):
                if all(int(P[r, c]) == f[(c - cand) % hp] for c in range(W)):
                    s = cand
                    break
            shifts[r] = s
        full = [int(v) for v in I[R]]
        for s in sorted(set(shifts.values())):
            rws = [r for r in dirty_rows if shifts[r] == s]
            # stamp the clean row, offset by this phase (covers columns s..W-1)
            a = [r for r in rws if h_changes(r, s, full)]
            if a:
                ops.append(28); sels.append([R, 0, 0, W - 1])   # full clean row rectangle
                for r in a:
                    ops.append(30); sels.append([r, s, 0, 0])
                    h_apply(r, s, full)
            if s != 0:
                t0 = hp - s
                part = [int(v) for v in I[R][t0:]]
                b = [r for r in rws if h_changes(r, 0, part)]
                if b:
                    ops.append(28); sels.append([R, t0, 0, W - 1 - t0])  # tail of clean row
                    for r in b:
                        ops.append(30); sels.append([r, 0, 0, 0])
                        h_apply(r, 0, part)
    else:
        # ---- rows are constant: the pattern repeats vertically, every column is a
        #      shift of the clean column C
        vp = _min_period([int(v) for v in P[:, C]])
        gcol = [int(v) for v in P[:vp, C]]
        shifts = {}
        dirty_cols = sorted({c for r, c in noise})
        for c in dirty_cols:
            t = 0
            for cand in range(max(1, vp)):
                if all(int(P[r, c]) == gcol[(r - cand) % vp] for r in range(H)):
                    t = cand
                    break
            shifts[c] = t
        full = [int(v) for v in I[:, C]]
        for t in sorted(set(shifts.values())):
            cls = [c for c in dirty_cols if shifts[c] == t]
            a = [c for c in cls if v_changes(c, t, full)]
            if a:
                ops.append(28); sels.append([0, C, H - 1, 0])   # full clean column rectangle
                for c in a:
                    ops.append(30); sels.append([t, c, 0, 0])
                    v_apply(c, t, full)
            if t != 0:
                u0 = vp - t
                part = [int(v) for v in I[u0:, C]]
                b = [c for c in cls if v_changes(c, 0, part)]
                if b:
                    ops.append(28); sels.append([u0, C, H - 1 - u0, 0])  # tail of clean column
                    for c in b:
                        ops.append(30); sels.append([0, c, 0, 0])
                        v_apply(c, 0, part)

    # --- 3. anything the replication genuinely left over (normally nothing)
    rem = [(r, c) for r in range(H) for c in range(W) if int(grid[r, c]) != int(P[r, c])]
    if rem:
        for col in sorted({int(P[r, c]) for r, c in rem}):
            cells = [(r, c) for (r, c) in rem if int(P[r, c]) == col]
            ops.append(col); sels.append(sel_of(cells))
            for (r, c) in cells:
                grid[r, c] = col

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
