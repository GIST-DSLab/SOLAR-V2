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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kw) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return lo + round(d * (hi - lo))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = unifint(10, max(10, max_h))
        w = unifint(10, max(10, max_w))
        ph = unifint(2, 9)
        pw = unifint(2, 9)
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        vcov = locih <= lociw < locih + ph
        hcov = locjw <= locjh < locjw + pw
        if vcov != hcov:
            continue          # one run sits on the other's line alone: ambiguous
        break
    col = random.choice(remcols)
    vb = {}
    for k in range(ph):
        r = locih + k
        vb[(r, locjh)] = col if abs(r - lociw) % ph == 0 else random.choice(remcols)
    hb = {}
    for k in range(pw):
        c = locjw + k
        hb[(lociw, c)] = col if abs(c - locjh) % pw == 0 else random.choice(remcols)
    gi = np.full((h, w), bgc, dtype=int)
    for (r, c), v in list(vb.items()) + list(hb.items()):
        gi[r, c] = v
    go = gi.copy()
    for k in range(h // ph + 1):
        for s in (k * ph, -k * ph):
            for (r, c), v in vb.items():
                if 0 <= r + s < h:
                    go[r + s, c] = v
    for k in range(w // pw + 1):
        for s in (k * pw, -k * pw):
            for (r, c), v in hb.items():
                if 0 <= c + s < w:
                    go[r, c + s] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: a line is OR-ed with ITSELF shifted by
    the period.  The unit being copied is the whole stretch of the line that is
    already laid (read off the working grid with CopyO), not the original run:
    each Paste puts that stretch one-or-more periods further along, so the
    tiled stretch doubles outward until it meets the edge.  0 is invisible to
    Copy/Paste, so cells a paste should have carried as 0 get a Color0."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = I != bgc
    cj = int(np.argmax(fg.sum(axis=0)))
    ri = int(np.argmax(fg.sum(axis=1)))

    def longest(line):
        best, cur, start = (0, -1), 0, 0
        for i, v in enumerate(line):
            if v:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best[1] - best[0] + 1:
                    best = (start, i)
            else:
                cur = 0
        return best

    g = I.copy()
    clip = None
    ops, sels = [], []

    for axis in (0, 1):
        if axis == 0:
            a, b = longest(fg[:, cj]); L = h
            at = lambda p: (p, cj)
        else:
            a, b = longest(fg[ri, :]); L = w
            at = lambda p: (ri, p)
        n = b - a + 1
        steps = []
        # grow forward (down / right), then backward (up / left)
        while b < L - 1 or a > 0:
            D = n * ((b - a + 1) // n)
            if b < L - 1:
                dest = list(range(b + 1, min(L - 1, b + D) + 1))
                b = dest[-1]
            else:
                dest = list(range(max(0, a - D), a))
                a = dest[0]
            steps.append((dest, D))
        # replay the planned steps on the working grid
        a0, b0 = longest(fg[:, cj]) if axis == 0 else longest(fg[ri, :])

        def paste(grid, cl, p0):
            out = grid.copy()
            for i, v in enumerate(cl[1]):
                p = p0 + i
                if p < L and v != 0:
                    out[at(p)] = v
            return out

        for dest, D in steps:
            forward = dest[0] > b0
            if forward:
                # the whole stretch one shift back; whatever hangs over the
                # far edge is clipped by ARCLE
                src = list(range(dest[0] - D, dest[0]))
            else:
                src = [p + D for p in dest]
            vals = tuple(int(g[at(p)]) for p in src)
            new = (axis, vals)
            want = paste(g, new, dest[0])
            if not np.array_equal(want, g):
                if clip is None or clip[0] != axis or not np.array_equal(paste(g, clip, dest[0]), want):
                    ops.append(29); sels.append(sel_of([at(p) for p in src]))
                    clip = new
                ops.append(30); sels.append(sel_of([at(dest[0])]))
                g = want
            zeros = [at(p) for p, q in zip(dest, src)
                     if g[at(q)] == 0 and g[at(p)] != 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))
                for c in zeros:
                    g[c] = 0
            if forward:
                b0 = dest[-1]
            else:
                a0 = dest[0]

    # the two lines share one cell; a step whose only effect is a cell the
    # other line's tiling lays anyway is dropped
    def replay(acts):
        gg = np.zeros((30, 30), dtype=int)
        gg[:h, :w] = I
        cb = None
        for op, cells in acts:
            if op == 29:
                rs = [r for r, _ in cells]; cs = [c for _, c in cells]
                r0, c0 = min(rs), min(cs)
                cb = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=int)
                for r, c in cells:
                    cb[r - r0, c - c0] = gg[r, c]
            elif op == 30:
                if cb is None:
                    continue
                r0, c0 = cells[0]
                sub = gg[r0:r0 + cb.shape[0], c0:c0 + cb.shape[1]]
                pc = cb[:sub.shape[0], :sub.shape[1]]
                sub[pc > 0] = pc[pc > 0]
            else:
                for r, c in cells:
                    gg[r, c] = op
        return gg[:h, :w].copy()

    acts = [(op, [tuple(x) for x in s["cells"]]) for op, s in zip(ops, sels)]
    target = replay(acts)
    dropped = True
    while dropped:
        dropped = False
        for i in range(len(acts)):
            if np.array_equal(replay(acts[:i] + acts[i + 1:]), target):
                del acts[i]
                dropped = True
                break
    ops = [op for op, _ in acts]
    sels = [sel_of(cells) for _, cells in acts]
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # whole grid
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
