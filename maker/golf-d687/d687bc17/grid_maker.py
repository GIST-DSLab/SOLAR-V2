"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    v = int(round(a + (b - a) * d))
    return max(a, min(b, v))


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, **kwargs) -> dict:
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = c1
    for i in range(h):
        gi[i][0] = c2
    for i in range(h):
        gi[i][w - 1] = c3
    for j in range(w):
        gi[h - 1][j] = c4
    for (i, j) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        gi[i][j] = bgc
    go = [row[:] for row in gi]

    cands = [(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]
    ndots = _unifint(diff_lb, diff_ub, (1, min(len(cands), 2 * h + 2 * w)))
    dots = random.sample(cands, ndots)
    dots = [(random.choice((c1, c2, c3, c4)), ij) for ij in dots]

    # top colour c1 -> row 1, per column
    n1 = [ij for c, ij in dots if c == c1]
    cov = sorted(set(j for _, j in n1))
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for i, j in n1 if j == jj])
        gi[loci][jj] = c1
        go[1][jj] = c1
    # left colour c2 -> col 1, per row
    n2 = [ij for c, ij in dots if c == c2]
    cov = sorted(set(i for i, _ in n2))
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for i, j in n2 if i == ii])
        gi[ii][locj] = c2
        go[ii][1] = c2
    # bottom colour c4 -> row h-2, per column
    n3 = [ij for c, ij in dots if c == c4]
    cov = sorted(set(j for _, j in n3))
    if len(cov) == w - 4 and w > 5:
        cov.remove(random.choice(cov))
    for jj in cov:
        loci = random.choice([i for i, j in n3 if j == jj])
        gi[loci][jj] = c4
        go[h - 2][jj] = c4
    # right colour c3 -> col w-2, per row
    n4 = [ij for c, ij in dots if c == c3]
    cov = sorted(set(i for i, _ in n4))
    if len(cov) == h - 4 and h > 5:
        cov.remove(random.choice(cov))
    for ii in cov:
        locj = random.choice([j for i, j in n4 if i == ii])
        gi[ii][locj] = c3
        go[ii][w - 2] = c3

    noisecands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    noisecols = [c for c in cols if c not in (bgc, c1, c2, c3, c4)]
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
    nnoise = min(nnoise, len(noisecands))
    for (i, j) in random.sample(noisecands, nnoise):
        gi[i][j] = random.choice(noisecols)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Rule: every border side (top/left/bottom/right) has its own colour; each dot of
    # that colour inside the inner box slides straight toward its side and parks one
    # cell in from the border; everything else (noise, corners) becomes background.
    # Large strokes: (1) wipe all noise-coloured cells in one Color op, (2) per side,
    # slide the dots as ONE growing group: the farthest row/col of dots starts moving,
    # and each time the group reaches the next row/col holding dots of that colour it
    # re-grabs them too, so a side costs (max distance) Moves instead of one chain per
    # dot, (3) one Color(bgc) on the inner rectangle wipes the start-cell holes.
    import itertools
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    c1 = int(I[0, 1]); c2 = int(I[1, 0]); c3 = int(I[1, w - 1]); c4 = int(I[h - 1, 1])
    bgc = None
    if examples:
        for ex in examples:
            bgc = int(np.asarray(ex[1])[0, 0])
            break
    if bgc is None:
        inner = [int(v) for v in I[1:h - 1, 1:w - 1].flatten()]
        bgc = Counter(inner).most_common(1)[0][0]
    keep = {bgc, c1, c2, c3, c4}
    r0, r1, q0, q1 = 2, h - 3, 2, w - 3            # inner box holding the dots

    # side spec: colour, axis (0 = rows move, 1 = cols move), direction, target line
    sides = [(c1, 0, -1, 1, 20), (c2, 1, -1, 1, 23), (c4, 0, 1, h - 2, 21), (c3, 1, 1, w - 2, 22)]

    def side_dots(col):
        return [(r, c) for r in range(r0, r1 + 1) for c in range(q0, q1 + 1) if int(I[r, c]) == col]

    def plan_side(spec, mode):
        col, ax, d, tgt, op = spec
        dots = side_dots(col)
        out = []
        if not dots:
            return out
        if col == 0:
            # zero cells cannot be grabbed by Move: paint the projection onto the target line
            dest = sorted({(tgt, c) if ax == 0 else (r, tgt) for r, c in dots})
            out.append((0, dest))
            return out
        key = (lambda p: p[ax])
        if mode == "group":
            levels = sorted({key(p) for p in dots}, key=lambda v: d * v)  # farthest first
            cur, pos = [], None
            for L in levels:
                if pos is not None:
                    while pos != L:
                        out.append((op, None)); pos += d
                        cur = [(r + d, c) if ax == 0 else (r, c + d) for r, c in cur]
                cur = cur + [p for p in dots if key(p) == L]
                out.append((op, list(cur))); pos = L + d
                cur = [(r + d, c) if ax == 0 else (r, c + d) for r, c in cur]
            while pos != tgt:
                out.append((op, None)); pos += d
        else:
            for p in sorted(dots, key=lambda p: d * key(p)):
                steps = abs(tgt - key(p))
                out.append((op, [p]))
                out.extend([(op, None)] * (steps - 1))
        return out

    def simulate(seq):
        g = I.copy()
        snap, obj, off = None, None, None
        for op, cells in seq:
            if op in (20, 21, 22, 23):
                dr, dc = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
                if cells is not None:
                    obj = [(r, c, int(g[r, c])) for r, c in cells if g[r, c] != 0]
                    snap = g.copy()
                    for r, c in cells:
                        snap[r, c] = 0
                    off = (0, 0)
                off = (off[0] + dr, off[1] + dc)
                g = snap.copy()
                for r, c, v in obj:
                    rr, cc = r + off[0], c + off[1]
                    if 0 <= rr < h and 0 <= cc < w:
                        g[rr, cc] = v
            else:
                for r, c in cells:
                    g[r, c] = op
        return g

    # expected result, derived from I by the rule (used only to validate a plan)
    exp = np.full_like(I, bgc)
    exp[0, 1:w - 1] = c1; exp[h - 1, 1:w - 1] = c4; exp[1:h - 1, 0] = c2; exp[1:h - 1, w - 1] = c3
    for col, ax, d, tgt, op in sides:
        for r, c in side_dots(col):
            if ax == 0:
                exp[tgt, c] = col
            else:
                exp[r, tgt] = col

    noise = [(r, c) for r in range(h) for c in range(w) if int(I[r, c]) not in keep]
    head = [(bgc, noise)] if noise else []
    box = [(r, c) for r in range(r0, r1 + 1) for c in range(q0, q1 + 1)]

    best = None
    for order in itertools.permutations(range(4)):
        for modes in itertools.product(("group", "dot"), repeat=4):
            seq = list(head)
            for i in order:
                seq += plan_side(sides[i], modes[i])
            g = simulate(seq)
            if any(g[r, c] != bgc for r, c in box):
                seq.append((bgc, box)); g = simulate(seq)
            if np.array_equal(g, exp) and (best is None or len(seq) < len(best)):
                best = seq
        if best is not None and all(len(best) <= len(best) for _ in [0]):
            pass
    ops, sels = [], []
    for op, cells in best:
        ops.append(op)
        sels.append(sel_of([]) if cells is None else sel_of(cells))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
