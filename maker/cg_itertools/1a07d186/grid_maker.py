"""
ARC Task: 1a07d186 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, transpose=None, **kw):
    if transpose is None:
        transpose = random.choice([False, True])
    H = max_w if transpose else max_h
    W = max_h if transpose else max_w
    h = _unifint(diff_lb, diff_ub, (8, min(30, H)))
    w = _unifint(diff_lb, diff_ub, (8, min(30, W)))
    remcols = [c for c in range(10) if c != bgc]
    nlines = _unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = random.sample(remcols, nlines)
    remcols = [c for c in remcols if c not in linecols]
    nnoise = _unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = random.sample(remcols, nnoise)
    locopts = list(range(w))
    locs = []
    for _ in range(nlines):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if not (loc - 2 <= x <= loc + 2)]
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = [[bgc] * w for _ in range(h)]
    for loc, col in zip(locs, linecols):
        for r in range(h):
            gi[r][loc] = col
    go = [row[:] for row in gi]
    nilocs = _unifint(diff_lb, diff_ub, (1, h))
    ilocs = random.sample(range(h), nilocs)
    dotopts = [c for c in range(w) if c not in locs]
    pool = list(set(linecols) | set(noisecols))
    for ii in ilocs:
        ndots = _unifint(diff_lb, diff_ub, (1, min(nlines + nnoise, (w - nlines) // 2 - 1)))
        ndots = min(ndots, len(pool), len(dotopts))
        dl = random.sample(dotopts, ndots)
        dc = random.sample(pool, ndots)
        for j, col in zip(dl, dc):
            gi[ii][j] = col
            if col in linecols:
                lj = locs[linecols.index(col)]
                go[ii][lj + 1 if j > lj else lj - 1] = col
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the canvas colour shared by the demonstrations and I
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # frontier lines: full rows / full columns of one non-bg colour
    lines = []
    for r in range(h):
        if I[r, 0] != bgc and np.all(I[r] == I[r, 0]):
            lines.append(('row', r, int(I[r, 0])))
    for c in range(w):
        if I[0, c] != bgc and np.all(I[:, c] == I[0, c]):
            lines.append(('col', c, int(I[0, c])))
    on_line = np.zeros((h, w), bool)
    line_of = {}
    for o, i, k in lines:
        if o == 'row':
            on_line[i, :] = True
        else:
            on_line[:, i] = True
        line_of.setdefault(k, (o, i))

    dots = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & ~on_line))]

    def landing(r, c, k):
        o, i = line_of[k]
        if o == 'row':
            return (i + 1 if r > i else i - 1, c)
        return (r, i + 1 if c > i else i - 1)

    ops, sels = [], []
    G = I.copy()

    # 1. noise: dots whose colour has no line vanish -- one FloodFill per blob
    noise = {p for p in dots if int(I[p]) not in line_of}
    seen = set()
    for p in sorted(noise):
        if p in seen:
            continue
        col = int(I[p])
        comp, st = [], [p]
        seen.add(p)
        while st:
            a, b = st.pop()
            comp.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (a + da, b + db)
                if q in noise and q not in seen and int(I[q]) == col:
                    seen.add(q)
                    st.append(q)
        ops.append(10 + bgc); sels.append([p[0], p[1], 0, 0])
        for q in comp:
            G[q] = bgc

    # 2. matching dots slide toward their own line until they touch it
    pending = []
    for p in dots:
        k = int(I[p])
        if k in line_of:
            d = landing(p[0], p[1], k)
            if d != p:
                pending.append((p, d, k))

    def dist(t):
        return abs(t[0][0] - t[1][0]) + abs(t[0][1] - t[1][1])

    deferred = []  # cycle fallback: (dest, colour)
    while pending:
        starts = {t[0] for t in pending}
        ready = [t for t in pending if t[1] not in starts]
        if not ready:
            # cyclic blocking: lift one dot off, paint it at its landing later
            t = min(pending, key=dist)
            pending.remove(t)
            ops.append(bgc); sels.append(sel_of([t[0]]))
            G[t[0]] = bgc
            deferred.append((t[1], t[2]))
            continue
        t = min(ready, key=lambda t: (dist(t), t[0]))
        pending.remove(t)
        (r, c), (dr_, dc_), k = t
        if k == 0:
            # colour-0 content is transparent to Move: erase then repaint
            ops.append(bgc); sels.append(sel_of([(r, c)]))
            G[r, c] = bgc
            ops.append(0); sels.append(sel_of([(dr_, dc_)]))
            G[dr_, dc_] = 0
            continue
        if dr_ != r:
            op = 21 if dr_ > r else 20
            n = abs(dr_ - r)
        else:
            op = 22 if dc_ > c else 23
            n = abs(dc_ - c)
        ops.append(op); sels.append(sel_of([(r, c)]))   # grab the dot
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))     # keep sliding same dot
        G[r, c] = 0
        G[dr_, dc_] = k
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of([(r, c)]))  # vacated start cell
            G[r, c] = bgc

    for d, k in deferred:
        if G[d] != k:
            ops.append(k); sels.append(sel_of([d]))
            G[d] = k

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
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
                                f"for task 1a07d186"
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
                    f"Failed to build a complete episode for task 1a07d186 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1a07d186-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
