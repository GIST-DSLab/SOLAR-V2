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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        examples = [dict(v) for v in ORIENTS]
        examples += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, transpose=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + int(round(d * (b - a)))))

    if transpose is None:
        transpose = random.choice([False, True])
    hmax, wmax = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (8, max(8, hmax)))
    w = unifint(diff_lb, diff_ub, (8, max(8, wmax)))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    nlines = unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = random.sample(remcols, nlines)
    remcols = [c for c in remcols if c not in linecols]
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = random.sample(remcols, nnoisecols)
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
    gi = np.full((h, w), bgc, dtype=int)
    for loc, col in zip(locs, linecols):
        gi[:, loc] = col
    go = gi.copy()
    nilocs = unifint(diff_lb, diff_ub, (1, h))
    ilocs = random.sample(range(h), nilocs)
    dotlocopts = [j for j in range(w) if j not in locs]
    pool = sorted(set(linecols) | set(noisecols))
    for ii in ilocs:
        ndots = unifint(diff_lb, diff_ub, (1, min(nlines + nnoisecols, (w - nlines) // 2 - 1)))
        ndots = min(ndots, len(pool), len(dotlocopts))
        dotlocs = random.sample(dotlocopts, ndots)
        dotcols = random.sample(pool, ndots)
        for j, col in zip(dotlocs, dotcols):
            gi[ii, j] = col
            if col in linecols:
                lj = locs[linecols.index(col)]
                go[ii, lj + 1 if j > lj else lj - 1] = col
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    # Background: fixed per episode -> majority colour over the demonstration inputs.
    cnt = Counter()
    if examples:
        for ex in examples:
            ei = ex[0] if not isinstance(ex, dict) else ex["input"]
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    else:
        cnt.update(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 1. Lines = full-span uniform non-background rows/columns.
    lines = []
    for r in range(h):
        v = int(I[r, 0])
        if v != bgc and np.all(I[r] == v):
            lines.append(('row', r, v))
    for c in range(w):
        v = int(I[0, c])
        if v != bgc and np.all(I[:, c] == v):
            lines.append(('col', c, v))
    on_line = np.zeros((h, w), dtype=bool)
    line_of = {}
    for o, idx, k in lines:
        if o == 'row':
            on_line[idx, :] = True
        else:
            on_line[:, idx] = True
        line_of.setdefault(k, (o, idx))

    G = I.copy()
    ops, sels = [], []
    dots = [(int(r), int(c)) for r, c in zip(*np.nonzero((I != bgc) & (~on_line)))]

    # 2. Each matching dot slides along its row/column until it touches its line.
    def landing(r, c, k):
        o, idx = line_of[k]
        if o == 'row':
            return (idx + 1 if r > idx else idx - 1, c)
        return (r, idx + 1 if c > idx else idx - 1)

    pending = []
    holes = set()
    for o, idx, k in sorted(lines, key=lambda t: (t[0], t[1])):
        if line_of.get(k) != (o, idx):
            continue
        mine = [(r, c) for r, c in dots if int(I[r, c]) == k]
        mine.sort(key=lambda p: (abs(p[0] - idx) if o == 'row' else abs(p[1] - idx), p))
        pending += [(r, c, k) for r, c in mine]

    def slide(r, c, k):
        lr, lc = landing(r, c, k)
        dr, dc = lr - r, lc - c
        n = abs(dr) + abs(dc)
        if n == 0:
            return
        if k == 0:
            # ARCLE's Move treats colour 0 as empty, so a 0-dot cannot be grabbed: it is
            # drawn beside its line, and its old cell joins the vacated cells repaired below.
            if G[lr, lc] != 0:          # a vacated cell there already reads 0
                ops.append(0); sels.append(sel_of([(lr, lc)]))
                G[lr, lc] = 0
            holes.add((r, c))
            holes.discard((lr, lc))
            return
        if dr:
            op = 21 if dr > 0 else 20
        else:
            op = 22 if dc > 0 else 23
        ops.append(op); sels.append(sel_of([(r, c)]))   # grab the dot
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))      # keep sliding it
        G[r, c] = 0
        G[lr, lc] = k
        holes.add((r, c))
        holes.discard((lr, lc))

    moved = set()
    while pending:
        progressed = False
        rest = []
        for (r, c, k) in pending:
            lr, lc = landing(r, c, k)
            unmoved = {(a, b) for a, b, _ in pending if (a, b) not in moved}
            if (lr, lc) == (r, c) or (lr, lc) not in unmoved:
                slide(r, c, k)
                moved.add((r, c))
                progressed = True
            else:
                rest.append((r, c, k))
        pending = rest
        if pending and not progressed:
            # Two dots blocking each other's landing cells: slide one over the other,
            # then the overwritten dot is re-drawn at its own landing cell.
            r, c, k = pending.pop(0)
            lr, lc = landing(r, c, k)
            victim = [(a, b, kk) for (a, b, kk) in pending if (a, b) == (lr, lc)]
            slide(r, c, k)
            for (a, b, kk) in victim:
                pending.remove((a, b, kk))
                vr, vc = landing(a, b, kk)
                ops.append(kk); sels.append(sel_of([(vr, vc)]))
                G[vr, vc] = kk

    # Repair the cells the dots left at 0 (ARCLE zeroes a grabbed footprint),
    # unless another dot has since landed there.
    left = sorted(p for p in holes if G[p] == 0)
    if bgc != 0 and left:
        ops.append(bgc); sels.append(sel_of(left))
        for p in left:
            G[p] = bgc

    # 3. Noise: every colour that has no line of its own is erased, one colour at a time
    #    (whatever of it is still visible -- a dot that landed on a noise cell covered it).
    noise_cols = sorted({int(I[r, c]) for r, c in dots if int(I[r, c]) not in line_of})
    for k in noise_cols:
        cells = [(r, c) for r, c in dots if int(I[r, c]) == k and G[r, c] == k]
        if not cells:
            continue
        ops.append(bgc); sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
