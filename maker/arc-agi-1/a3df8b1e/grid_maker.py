"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    g = globals().get("unifint")
    if g is not None:
        return g(diff_lb, diff_ub, bounds)
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _id(g):
    return [list(r) for r in g]


def _dm(g):                                   # dmirror  (transpose)
    return [list(r) for r in zip(*g)]


def _vm(g):                                   # vmirror  (left<->right)
    return [list(r)[::-1] for r in g]


def _hm(g):                                   # hmirror  (up<->down)
    return [list(r) for r in g[::-1]]


def _r180(g):
    return [list(r)[::-1] for r in g[::-1]]


def _r90(g):                                  # clockwise
    return [list(r) for r in zip(*g[::-1])]


def _r270(g):                                 # counter-clockwise
    return [list(r) for r in zip(*g)][::-1]


def _cm(g):                                   # cmirror (anti-transpose)
    return _dm(_r180(g))


# numlins (how many corners emit a bouncing ray) and the final orientation are the
# discrete structural cases of this task -> plan them per instance.
def _variants():
    ns = [1, 2, 3, 4]
    ps = [True, False, True, False]
    random.shuffle(ps)
    return [{"numlins": n, "portrait_out": p} for n, p in zip(ns, ps)]


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the ray colour must be non-zero: ARCLE's object ops (Flip) treat 0 as "nothing"
    linc = random.choice([c for c in cols if c != bgc and c != 0])

    variants = _variants()
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc,
             numlins=None, portrait_out=None) -> dict:
    if numlins is None:
        numlins = random.choice([1, 2, 3, 4])
    if portrait_out is None:
        portrait_out = random.choice([True, False])

    # transposing mirrors may swap h/w, so keep both dims inside both limits
    lim = min(max_h, max_w)
    wub = min(10, lim - 1)
    if wub < 2:
        wub = 2
    w = _unifint(diff_lb, diff_ub, (2, wub))
    h = _unifint(diff_lb, diff_ub, (w + 1, max(w + 1, lim)))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    r, c = h - 1, 0
    gi[r][c] = linc
    go[r][c] = linc
    direc = 1
    while True:
        r -= 1
        c += direc
        if c == 0 or c == w - 1:
            direc *= -1
        if r < 0:
            break
        go[r][c] = linc

    mfs = [_id, _dm, _cm, _vm, _hm, _r90, _r180, _r270]
    for fn in random.sample(mfs, random.choice([1, 2])):
        gi = fn(gi)
        go = fn(go)

    if (len(gi) > len(gi[0])) != bool(portrait_out):
        gi = _dm(gi)
        go = _dm(go)

    gix = [row[:] for row in gi]
    gox = [row[:] for row in go]

    def overlay(dst, src):
        for i, row in enumerate(src):
            for j, v in enumerate(row):
                if v == linc:
                    dst[i][j] = linc

    if numlins > 1:
        overlay(gi, _hm(gix))
        overlay(go, _hm(gox))
    if numlins > 2:
        overlay(gi, _vm(gix))
        overlay(go, _vm(gox))
    if numlins > 3:
        overlay(gi, _r180(gix))
        overlay(go, _r180(gox))

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    """
    Rule: every marked corner shoots a diagonal ray that bounces between the two
    walls of the SHORT dimension while travelling along the LONG one.  The marked
    corners always form a mirror family (p, hmirror p, vmirror p, rot180 p), so the
    extra rays ARE reflections of the first one -> they are produced with FlipV/FlipH
    on the ray object itself (its bbox is the whole grid, so the flip mirrors it about
    the grid centre line).  ARCLE's flip MOVES the grabbed ray, so after each flip the
    original ray is laid down again with one Color op.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    cset = set(corners)
    cnt = Counter(int(I[r, c]) for r in range(h) for c in range(w) if (r, c) not in cset)
    bgc = cnt.most_common(1)[0][0]                       # bg = mostcolor of non-corner cells
    fg = [v for v in sorted(set(int(x) for x in I.reshape(-1))) if v != bgc]
    col = fg[0] if fg else bgc
    marks = [q for q in corners if int(I[q[0], q[1]]) == col]

    def ray(q):
        r0, c0 = q
        cells = [(r0, c0)]
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1
        r, c = r0, c0
        if h > w:                                        # portrait: bounce off left/right
            while True:
                r += dr
                c += dc
                if c == 0 or c == w - 1:
                    dc *= -1
                if not (0 <= r < h):
                    break
                cells.append((r, c))
        else:                                            # landscape: bounce off top/bottom
            while True:
                r += dr
                c += dc
                if r == 0 or r == h - 1:
                    dr *= -1
                if not (0 <= c < w):
                    break
                cells.append((r, c))
        return frozenset(cells)

    def mir(cells, kind):
        if kind == "ud":
            return frozenset((h - 1 - r, c) for (r, c) in cells)
        return frozenset((r, w - 1 - c) for (r, c) in cells)

    cur = I.copy()
    ops, sels = [], []

    def do_color(cells, val):
        pts = sorted(p for p in cells if int(cur[p[0], p[1]]) != val)
        if not pts:
            return False
        ops.append(int(val))
        sels.append(sel_of(pts))
        for (r, c) in pts:
            cur[r, c] = val
        return True

    def do_flip(cells, kind):
        pts = sorted(cells)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        vals = {p: int(cur[p[0], p[1]]) for p in pts}
        new = cur.copy()
        for (r, c) in pts:                               # ARCLE zeroes the grabbed cells
            new[r, c] = 0
        for (r, c), v in vals.items():
            nr, nc = (r0 + r1 - r, c) if kind == "ud" else (r, c0 + c1 - c)
            if v != 0:
                new[nr, nc] = v
        if np.array_equal(new, cur):
            return False
        ops.append(27 if kind == "ud" else 26)           # 27 = FlipV (up/down), 26 = FlipH
        sels.append(sel_of(pts))
        cur[:, :] = new
        return True

    targets = []
    if marks:
        p0 = marks[0]
        cand = [p0,
                (h - 1 - p0[0], p0[1]),
                (p0[0], w - 1 - p0[1]),
                (h - 1 - p0[0], w - 1 - p0[1])]
        for q in cand:
            if q in marks:
                R = ray(q)
                if R not in targets:
                    targets.append(R)

    drawn = []
    drawn_cells = set()
    for idx, X in enumerate(targets):
        if X <= drawn_cells:                             # already fully on the grid
            drawn.append(X)
            drawn_cells |= X
            continue
        if idx == 0:
            do_color(X, col)                             # draw the first bouncing ray
        else:
            found = None
            for Y in drawn:
                for kind in ("ud", "lr"):
                    if mir(Y, kind) == X:
                        found = (Y, kind)
                        break
                if found:
                    break
            if found:
                Y, kind = found
                if do_flip(Y, kind):                     # reflect the ray onto its mirror corner
                    do_color(Y, col)                     # lay the original ray back down
                else:
                    do_color(X, col)
            else:
                do_color(X, col)
        drawn.append(X)
        drawn_cells |= X

    if not np.array_equal(cur, O):                       # safety net (normally inactive)
        rem = {}
        for r in range(h):
            for c in range(w):
                if cur[r, c] != O[r, c]:
                    rem.setdefault(int(O[r, c]), []).append((r, c))
        for v, pts in rem.items():
            do_color(pts, v)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                    # whole-grid rectangle: Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
