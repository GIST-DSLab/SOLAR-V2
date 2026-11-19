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
from collections import deque
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


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    tall = H > W                       # rays travel along the long axis
    L, S = (H, W) if tall else (W, H)
    P = 2 * (S - 1)                    # one full bounce of a ray

    # background: a cell that can never hold a corner marker
    bseed = (1, 0) if H >= 3 else (0, 1)
    bgc = int(I[bseed])
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    seeds = [q for q in corners if int(I[q]) != bgc]
    linc = int(I[seeds[0]])

    def ray(seed):                     # zigzag from a corner marker, as (t, u) list
        st, su = (seed if tall else (seed[1], seed[0]))
        out = []
        for k in range(L):
            m = k % P
            off = m if m <= S - 1 else P - m
            out.append((st + k if st == 0 else st - k, off if su == 0 else S - 1 - off))
        return out

    rays = [ray(s) for s in seeds]
    U = set(q for r in rays for q in r)

    # smallest period (a divisor of the bounce) of the combined rays along the long axis
    def col(t):
        return frozenset(u for u in range(S) if (t, u) in U)
    p = next(q for q in range(1, P + 1) if P % q == 0
             and all(col(t) == col(t + q) for t in range(L - q)))
    blk = min(p, L)

    def plan(rev):
        # rev=False: the first band sits at long index 0 and grows upward;
        # rev=True : it sits at the far end and grows back towards index 0
        def T(d):                      # distance from the growing end -> real index
            return L - 1 - d if rev else d

        def rc(t, u):
            return (t, u) if tall else (u, t)

        def band(d0, n):               # full rectangle covering distances d0..d0+n-1
            t0 = min(T(d0), T(d0 + n - 1))
            return [t0, 0, n - 1, W - 1] if tall else [0, t0, H - 1, n - 1]

        ops, sels = [], []
        cur = I.copy()

        # 1) the background turns 8: one connected region, one FloodFill seed
        ops.append(18); sels.append(sel_of([bseed]))
        cur[cur == bgc] = 8

        # 2) inside the first band, draw each marker's ray as straight diagonal strokes
        lo, hi = (L - blk, L - 1) if rev else (0, blk - 1)
        for r in rays:
            path = [q for q in r if lo <= q[0] <= hi]
            strokes, cs, d = [], [], None
            for q in path:
                if cs and d is not None and q[1] - cs[-1][1] != d:
                    strokes.append(cs); cs = [cs[-1]]; d = None
                if cs and d is None:
                    d = q[1] - cs[-1][1]
                cs.append(q)
            if cs:
                strokes.append(cs)
            for st in strokes:
                cells = [rc(*q) for q in st if cur[rc(*q)] != linc]
                if cells:
                    ops.append(linc); sels.append(sel_of(cells))
                    for c in cells:
                        cur[c] = linc

        # 3) the band repeats every p along the long axis: grow it by copy + paste
        if blk < L:
            if linc == 0:
                # 0 never travels through the clipboard: lay the ray colour as a
                # base on the rest of the grid; the pasted 8s stencil it
                ops.append(0); sels.append(band(blk, L - blk))   # full rectangle
            # BFS over (filled length f, clipboard length k): fewest copy/paste ops
            start = (blk, 0)
            prev = {start: None}
            dq = deque([start])
            goal = None
            while dq:
                f, k = dq.popleft()
                if f == L:
                    goal = (f, k)
                    break
                nxt = [((f, k2), ("copy", k2)) for k2 in range(1, f + 1) if k2 != k]
                if k:
                    nxt += [((b + k, k), ("paste", b)) for b in range(0, f + 1, p)
                            if f < b + k <= L]
                for s2, act in nxt:
                    if s2 not in prev:
                        prev[s2] = ((f, k), act)
                        dq.append(s2)
            acts, s = [], goal
            while prev[s] is not None:
                s, act = prev[s]
                acts.append(act)
            klen = 0
            for kind, v in reversed(acts):
                if kind == "copy":
                    klen = v
                    ops.append(29); sels.append(band(0, v))          # full band rectangle
                else:
                    t0 = min(T(v), T(v + klen - 1))
                    ops.append(30); sels.append([*rc(t0, 0), 0, 0])  # paste origin

        ops.append(34)
        sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    return min((plan(False), plan(True)), key=lambda x: len(x[0]))


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
