"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

VARIANTS = [
    {"wide": False, "numlins": 1},
    {"wide": True,  "numlins": 2},
    {"wide": False, "numlins": 3},
    {"wide": True,  "numlins": 4},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, wide=None, numlins=None) -> dict:
    if wide is None or numlins is None:
        v = random.choice(VARIANTS)
        wide = v["wide"] if wide is None else wide
        numlins = v["numlins"] if numlins is None else numlins
    maxL = min(30, max_w if wide else max_h)
    maxS = min(30, max_h if wide else max_w)
    w = unifint(diff_lb, diff_ub, (2, max(2, min(10, maxS, maxL - 1))))
    h = unifint(diff_lb, diff_ub, (w + 1, max(w + 1, maxL)))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    if (len(gi[0]) > len(gi)) != wide:
        gi = dmirror(gi)
        go = dmirror(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    go = replace(go, bgc, 8)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    tall = H > W                       # the rays travel along the long axis
    L, S = (H, W) if tall else (W, H)
    P = 2 * (S - 1)                    # one full bounce (there and back) of a ray

    # background = a cell that can never be a corner marker
    bseed = (1, 0) if H >= 3 else (0, 1)
    bgc = int(I[bseed])
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    seeds = [p for p in corners if int(I[p]) != bgc]
    linc = int(I[seeds[0]])

    def to_tu(p):
        return (p[0], p[1]) if tall else (p[1], p[0])

    def to_rc(t, u):
        return (t, u) if tall else (u, t)

    # the first marker fixes the end the pattern grows from
    prim_t = to_tu(seeds[0])[0]
    fwd = prim_t == 0

    def real_t(k):                     # k = distance from the primary end
        return k if fwd else L - 1 - k

    def ray_u(seed, k):                # cross position of seed's ray at distance k
        st, su = to_tu(seed)
        kk = k if (st == 0) == fwd else L - 1 - k
        m = kk % P
        off = m if m <= S - 1 else P - m
        return off if su == 0 else S - 1 - off

    def rect(k0, k1):                  # band of the grid between distances k0..k1-1
        k1 = min(k1, L)
        a, b = sorted((real_t(k0), real_t(k1 - 1)))
        return [a, 0, b - a, S - 1] if tall else [0, a, S - 1, b - a]

    cur = I.copy()
    ops, sels = [], []

    def emit(op, sel, newgrid):
        nonlocal cur
        if not np.array_equal(newgrid, cur):
            ops.append(op); sels.append(sel)
            cur = newgrid
            return True
        return False

    # 1) the background turns 8 — one connected region, one FloodFill seed
    g = cur.copy(); g[g == bgc] = 8
    emit(18, sel_of([bseed]), g)

    # 2) inside the first period from the primary end, draw every marker's
    #    ray as straight diagonal strokes (near-end markers first)
    B = min(P, L)
    order = sorted(seeds, key=lambda s: (s != seeds[0], (to_tu(s)[0] == 0) != fwd))
    for s in order:
        path = [(k, ray_u(s, k)) for k in range(B)]
        if (to_tu(s)[0] == 0) != fwd:
            path = path[::-1]          # trace it from its own side
        strokes, curs, d = [], [path[0]], None
        for a, b in zip(path, path[1:]):
            dd = b[1] - a[1]
            if d is None or dd == d:
                curs.append(b); d = dd
            else:
                strokes.append(curs); curs = [b]; d = None
        strokes.append(curs)
        for st in strokes:
            cells = [to_rc(real_t(k), u) for k, u in st]
            cells = [p for p in cells if cur[p] != linc]
            if cells:
                g = cur.copy()
                for p in cells:
                    g[p] = linc
                emit(linc, sel_of(cells), g)

    # 3) the pattern repeats every P cells along the long axis: copy the
    #    period band and paste it period after period
    if L > P:
        if linc == 0:
            # 0 never travels through the clipboard: lay the rays' colour as a
            # base over the rest of the grid, the pasted 8s then stencil it
            a, b, hh, ww = rect(P, L)
            g = cur.copy(); g[a:a + hh + 1, b:b + ww + 1] = 0
            emit(0, rect(P, L), g)
        nfull = L // P
        rem = L - nfull * P

        def copy_paste(k_len, k_dst_list):
            nonlocal cur
            src = rect(0, k_len)
            a, b, hh, ww = src
            clip = cur[a:a + hh + 1, b:b + ww + 1].copy()
            pastes = []
            g = cur.copy()
            for kd in k_dst_list:
                da, db, _, _ = rect(kd, kd + k_len)
                g2 = g.copy()
                reg = g2[da:da + hh + 1, db:db + ww + 1]
                cl = clip[:reg.shape[0], :reg.shape[1]]   # Paste clips at the grid edge
                reg[cl != 0] = cl[cl != 0]
                if not np.array_equal(g2, g):
                    pastes.append(([da, db, 0, 0], g2))
                    g = g2
            if pastes:
                ops.append(29); sels.append(src)
                for sel, g2 in pastes:
                    ops.append(30); sels.append(sel)
                cur = g

        if fwd:
            # pattern grows towards the far end: the last band runs off the
            # grid edge and Paste clips it, so one copy serves every period
            copy_paste(P, [i * P for i in range(1, -(-L // P))])
        else:
            # growing towards row/col 0: the final partial band would start
            # off-grid, so it is copied as its own shorter band
            copy_paste(P, [i * P for i in range(1, nfull)])
            if rem:
                # the leftover strip at the start edge: if the band already on
                # the clipboard lands there correctly (the combined rays repeat
                # with a shorter period), paste it again; else copy the strip
                g_old = None
                if nfull >= 2:
                    a, b, hh, ww = rect(0, P)
                    clip = cur[a:a + hh + 1, b:b + ww + 1]
                    g_old = cur.copy()
                    reg = g_old[0:hh + 1, 0:ww + 1]
                    reg[clip != 0] = clip[clip != 0]
                a, b, hh, ww = rect(0, rem)
                clip = cur[a:a + hh + 1, b:b + ww + 1]
                g_rem = cur.copy()
                reg = g_rem[0:hh + 1, 0:ww + 1]
                reg[clip != 0] = clip[clip != 0]
                if g_old is not None and np.array_equal(g_old, g_rem):
                    ops.append(30); sels.append([0, 0, 0, 0])
                    cur = g_rem
                else:
                    copy_paste(rem, [nfull * P])

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
