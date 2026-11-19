"""
ARC Task: 82819916 (RE-ARC) — LLM-generated grid_maker
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
except Exception:  # stand-alone use
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


# ---------------------------------------------------------------- helpers

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _rot90ccw(g):
    return [list(row) for row in zip(*g)][::-1]


def _rot(g, k):
    for _ in range(k % 4):
        g = _rot90ccw(g)
    return g


# ---------------------------------------------------------------- 1. colors

_VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ass, bss = random.sample(rem, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        examples = [dict(v) for v in _VARIANTS]
        examples += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ass": ass, "bss": bss, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate

def generate(diff_lb, diff_ub, max_h, max_w, bgc, ass, bss, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    hmax = max(5, min(30, int(max_h)))
    wmax = max(5, min(30, int(max_w)))
    h = _unifint(diff_lb, diff_ub, (5, hmax))
    w = _unifint(diff_lb, diff_ub, (5, wmax))

    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    itv = list(range(w))
    na = random.randint(2, w - 2)
    alocs = sorted(random.sample(itv, na))
    aset = set(alocs)
    blocs = sorted([j for j in itv if j not in aset])
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs

    llocs = random.randint(0, h - 1)
    gi = [[bgc] * w for _ in range(h)]
    for j in alocs:
        gi[llocs][j] = ass
    for j in blocs:
        gi[llocs][j] = bss

    numl = _unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = [r for r in range(h) if r != llocs]
    for _ in range(numl):
        if not remlocs:
            break
        lloc = random.choice(remlocs)
        remlocs.remove(lloc)
        a, b = random.sample(remcols, 2)
        for j in alocs:
            gi[lloc][j] = a
        for j in blocs:
            gi[lloc][j] = b

    cutoff = min(blocs) + 1
    go = [row[:] for row in gi]

    for r in range(h):
        for j in range(cutoff, w):
            gi[r][j] = bgc
    for j in alocs:
        gi[llocs][j] = ass
    for j in blocs:
        gi[llocs][j] = bss

    gi = _rot(gi, rot)
    go = _rot(go, rot)

    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


# ---------------------------------------------------------------- 3. derive

def derive_operations(I, O, examples=None):
    """
    Rule, read from I alone:
      * exactly one full line (a row, or a column when the picture is turned)
        carries no background at all -> the KEY line.  It is bicoloured, and its
        two colours split the positions along a line into two groups.
      * every other drawn line is cut short: only a stub survives, and that stub
        always shows one cell of each group -> the two colours that line assigns.
      * each cut-short line is completed: its group-1 colour on every group-1
        position, its group-2 colour on every group-2 position.

    The route -- which line is finished first, which way the sweep runs, which
    group is laid first, whether the majority colour goes down as a base coat
    with the minority drawn over it, and whether a line identical to one already
    finished is copied instead of painted again -- is read off the input: the
    key line's place in the grid, the sizes of the two groups, where each group
    starts, and which lines repeat.  Same input -> same route; the variety comes
    from the instances differing.  O is used only for the Submit bbox.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # ---- background colour: an untouched line is uniform -------------------
    bgc = None
    for r in range(hi):
        if len(set(I[r, :].tolist())) == 1:
            bgc = int(I[r, 0])
            break
    if bgc is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                bgc = int(I[0, c])
                break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # ---- orientation: the key line is a row only when some row has no bg ----
    horiz = any(not np.any(I[r, :] == bgc) for r in range(hi))

    G = (I if horiz else I.T).copy()      # line space: G[line][position]
    cur = G.copy()                        # working grid, in the same space
    nl, L = G.shape

    def cell(i, q):
        return (i, q) if horiz else (q, i)

    def whole_line(i):                    # a true full rectangle: one line
        return [i, 0, 0, L - 1] if horiz else [0, i, L - 1, 0]

    def line_origin(i):
        return [i, 0, 0, 0] if horiz else [0, i, 0, 0]

    # ---- the key line ------------------------------------------------------
    key = None
    for i in range(nl):
        if not np.any(G[i, :] == bgc):
            key = i
            break
    if key is None:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    groups = {}
    for q in range(L):
        groups.setdefault(int(G[key, q]), []).append(q)
    gcols = sorted(groups)
    grp_of = {q: int(G[key, q]) for q in range(L)}

    # ---- the cut-short lines ----------------------------------------------
    pattern = [i for i in range(nl) if i != key and np.any(G[i, :] != bgc)]

    # route choice 1 — where the key line sits decides where the sweep starts
    span = max(1, nl - 1)
    if key <= span / 3.0:                 # key near the start: work downwards
        order, backwards = sorted(pattern), False
    elif key >= 2.0 * span / 3.0:         # key near the end: work upwards
        order, backwards = sorted(pattern, reverse=True), True
    else:                                 # key in the middle: work outwards
        order = sorted(pattern, key=lambda i: (abs(i - key), i))
        backwards = False

    # route choice 2 — a lopsided split invites a base coat + minority on top
    layered = False
    big = small = None
    if len(gcols) == 2:
        big = max(gcols, key=lambda v: (len(groups[v]), v))
        small = gcols[0] if gcols[1] == big else gcols[1]
        layered = len(groups[big]) >= 2 * len(groups[small])

    # route choice 3 — otherwise the group that starts first is laid first
    gorder = sorted(gcols, key=lambda v: min(groups[v]))
    if backwards:
        gorder = gorder[::-1]

    def paint(colour, positions, line):
        cells = [cell(line, q) for q in positions if int(cur[line, q]) != colour]
        if not cells:
            return
        ops.append(int(colour))
        sels.append(sel_of(cells))
        for q in positions:
            cur[line, q] = colour

    done = {}          # colour signature -> a finished line already on the grid
    clip_from = None   # which line the clipboard currently holds

    for p in order:
        seeds = {}
        for v in gcols:
            shown = [q for q in groups[v] if int(G[p, q]) != bgc]
            if shown:
                seeds[v] = int(G[p, shown[0]])
        if not seeds:
            continue

        sig = tuple(seeds.get(v) for v in gcols)
        complete = all(v in seeds for v in gcols)

        # route choice 4 — a line identical to a finished one is copied over
        if complete and sig in done:
            src = done[sig]
            if clip_from != src:
                ops.append(29)                       # CopyO the finished line
                sels.append(whole_line(src))
                clip_from = src
            ops.append(30)                           # Paste it onto this line
            sels.append(line_origin(p))
            for q in range(L):
                if seeds[grp_of[q]] != 0:
                    cur[p, q] = seeds[grp_of[q]]
            # colour 0 does not travel with a Paste: set those cells down here
            zeros = [cell(p, q) for q in range(L)
                     if seeds[grp_of[q]] == 0 and int(cur[p, q]) != 0]
            if zeros:
                ops.append(0)
                sels.append(sel_of(zeros))
                for q in range(L):
                    if seeds[grp_of[q]] == 0:
                        cur[p, q] = 0
            continue

        if layered and big in seeds and small in seeds:
            paint(seeds[big], list(range(L)), p)      # base coat over the line
            paint(seeds[small], groups[small], p)     # minority drawn on top
        else:
            for v in gorder:
                if v in seeds:
                    paint(seeds[v], groups[v], p)

        if complete:
            done.setdefault(sig, p)

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
                        f"num_examples+1 ({num_examples + 1}) for task 82819916"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 82819916"
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
                                f"for task 82819916"
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
                    f"Failed to build a complete episode for task 82819916 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"82819916-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
