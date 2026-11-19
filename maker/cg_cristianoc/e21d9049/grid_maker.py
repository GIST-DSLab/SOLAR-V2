"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        ph = unifint(diff_lb, diff_ub, (2, 9))
        pw = unifint(diff_lb, diff_ub, (2, 9))
        locih = randint(0, h - ph)
        locjh = randint(0, w - 1)
        locjw = randint(0, w - pw)
        lociw = randint(0, h - 1)
        # keep the two runs apart so each line is read unambiguously
        if locih - 1 <= lociw <= locih + ph:
            continue
        if locjw - 1 <= locjh <= locjw + pw:
            continue
        break
    hbar = frozenset({(choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(choice(remcols), (0, k)) for k in range(pw)})
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, (locih, locjh))
    wbar = shift(wbar, (lociw, locjw))
    col = choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    go = paint(canv, hbar | wbar)
    for k in range(h // ph + 1):
        go = paint(go, shift(hbar, (k * ph, 0)))
        go = paint(go, shift(hbar, (-k * ph, 0)))
    for k in range(w // pw + 1):
        go = paint(go, shift(wbar, (0, k * pw)))
        go = paint(go, shift(wbar, (0, -k * pw)))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Read only from I.  The input holds one short vertical run (in the column
    that carries the most non-background cells) and one short horizontal run
    (in the row that carries the most).  Each run is a period of its line: it
    repeats along that line, in both directions, out to the grid edges.

    Per line: CopyI the run, Paste it at every period origin along the line
    (the copy hanging over the far edge is clipped by ARCLE).  The stretch
    before the first whole period shows the run's tail: CopyI that tail and
    Paste it at the line's start.  A run cell coloured 0 cannot travel with a
    paste, so the destination cells it should have carried get a Color0.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    MAXD = 30
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ri = Counter(r for r, _ in fg).most_common(1)[0][0]
    cj = Counter(c for _, c in fg).most_common(1)[0][0]
    vrows = sorted(r for r, c in fg if c == cj)
    hcols = sorted(c for r, c in fg if r == ri)

    st = {"grid": np.zeros((MAXD, MAXD), dtype=int),
          "clip": np.zeros((MAXD, MAXD), dtype=int), "cd": (0, 0)}
    st["grid"][:h, :w] = I

    def apply(s, op, cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if op == 28:
            s["clip"][:, :] = 0
            ch, cw = r1 - r0 + 1, c1 - c0 + 1
            s["cd"] = (ch, cw)
            for r, c in cells:
                if I[r, c] != 0:
                    s["clip"][r - r0, c - c0] = I[r, c]
        elif op == 30:
            ch, cw = s["cd"]
            for a in range(ch):
                for b in range(cw):
                    v = s["clip"][a, b]
                    if v and r0 + a < MAXD and c0 + b < MAXD:
                        s["grid"][r0 + a, c0 + b] = v
        else:
            for r, c in cells:
                s["grid"][r, c] = op

    def cp(s):
        return {"grid": s["grid"].copy(), "clip": s["clip"].copy(), "cd": s["cd"]}

    acts = []
    clip_src = None

    def at(axis, fixed, p):
        return (p, fixed) if axis == 0 else (fixed, p)

    for axis, fixed, line, L in ((0, cj, vrows, h), (1, ri, hcols, w)):
        s0, n = line[0], len(line)
        vals = [int(I[at(axis, fixed, s0 + i)]) for i in range(n)]
        base = s0 % n
        tiles = []
        if base > 0:
            tiles.append((0, n - base, base))      # the run's tail, cut by the edge
        k = base
        while k <= L - 1:
            if k != s0:
                tiles.append((k, 0, n))            # whole periods along the line
            k += n
        for dest, lo, ln in tiles:
            src = [at(axis, fixed, s0 + lo + i) for i in range(ln)]
            before = st["grid"][:h, :w].copy()
            trial = cp(st)
            if clip_src != src:
                apply(trial, 28, src)
            apply(trial, 30, [at(axis, fixed, dest)])
            if not np.array_equal(trial["grid"][:h, :w], before):
                if clip_src != src:
                    acts.append((28, src))
                    apply(st, 28, src)
                    clip_src = src
                acts.append((30, [at(axis, fixed, dest)]))
                apply(st, 30, [at(axis, fixed, dest)])
            zeros = [at(axis, fixed, dest + i) for i in range(ln)
                     if dest + i <= L - 1 and vals[lo + i] == 0
                     and st["grid"][at(axis, fixed, dest + i)] != 0]
            if zeros:
                acts.append((0, zeros))
                apply(st, 0, zeros)

    # safety: drop any act whose removal leaves the finished grid unchanged
    def replay(a):
        s = {"grid": np.zeros((MAXD, MAXD), dtype=int),
             "clip": np.zeros((MAXD, MAXD), dtype=int), "cd": (0, 0)}
        s["grid"][:h, :w] = I
        for op, cells in a:
            apply(s, op, cells)
        return s["grid"][:h, :w]

    target = replay(acts)
    changed = True
    while changed:
        changed = False
        for i in range(len(acts)):
            if np.array_equal(replay(acts[:i] + acts[i + 1:]), target):
                del acts[i]
                changed = True
                break

    ops = [op for op, _ in acts] + [34]
    sels = [sel_of(cells) for _, cells in acts] + [[0, 0, h - 1, w - 1]]  # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
