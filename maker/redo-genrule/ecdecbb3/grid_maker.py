"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# helpers (shared by generate() and derive_operations())
# ----------------------------------------------------------------------------

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    ilb = a + int((b - a) * diff_lb)
    iub = a + int((b - a) * diff_ub)
    ilb = max(a, min(ilb, b))
    iub = max(ilb, min(iub, b))
    return random.randint(ilb, iub)


def _connect(p, q):
    (ai, aj), (bi, bj) = p, q
    if ai == bi:
        return [(ai, j) for j in range(min(aj, bj), max(aj, bj) + 1)]
    if aj == bj:
        return [(i, aj) for i in range(min(ai, bi), max(ai, bi) + 1)]
    return []


def _rule_parts(grid, dotc, linc):
    """Faithful reimplementation of verify_ecdecbb3's per-dot geometry.

    The line-coloured full rows/columns are 'frontiers'; the two grid borders
    (one step outside the grid) count as frontiers too.  For every dot:
      * project it onto its nearest frontier and onto its second nearest one
        (a projection collapses onto the dot itself when the frontier is
        already adjacent -- that is what gravitate+crement does),
      * if the dot lies inside the bounding box of the lines, the ray is the
        segment joining the two projections,
      * otherwise the ray joins the dot to whichever projection lies inside
        that bounding box.
    Returns [(dot, ray_cells_on_grid, crossed_line_cells), ...].
    """
    g = [list(r) for r in grid]
    h, w = len(g), len(g[0])
    hrows = [r for r in range(h) if all(g[r][c] == linc for c in range(w))]
    vcols = [c for c in range(w) if all(g[r][c] == linc for r in range(h))]
    horizontal = len(hrows) > 0 and w > 1
    if horizontal:
        cands = sorted(set(hrows) | {-1, h})
        lo, hi = min(hrows), max(hrows)
    else:
        cands = sorted(set(vcols) | {-1, w})
        lo, hi = min(vcols), max(vcols)

    def dist(cd, i, j):
        return abs(i - cd) if horizontal else abs(j - cd)

    def proj(cd, i, j):
        if dist(cd, i, j) <= 1:          # already adjacent -> stays put
            return (i, j)
        return (cd, j) if horizontal else (i, cd)

    def inbd(pt):
        r, c = pt
        if horizontal:
            return lo <= r <= hi and 0 <= c < w
        return 0 <= r < h and lo <= c <= hi

    dots = [(i, j) for i in range(h) for j in range(w) if g[i][j] == dotc]
    lines = {(i, j) for i in range(h) for j in range(w) if g[i][j] == linc}

    parts = []
    for (i, j) in dots:
        order = sorted(cands, key=lambda cd: dist(cd, i, j))
        p1 = proj(order[0], i, j)
        p2 = proj(order[1], i, j) if len(order) > 1 else p1
        if inbd((i, j)):
            seg = _connect(p1, p2)
        else:
            tgt = p1 if inbd(p1) else p2
            seg = _connect((i, j), tgt)
        on = [(r, c) for (r, c) in seg if 0 <= r < h and 0 <= c < w]
        inter = [pt for pt in on if pt in lines]
        parts.append(((i, j), on, inter))
    return parts


def _apply_rule(grid, dotc, linc):
    g = [list(r) for r in grid]
    h, w = len(g), len(g[0])
    parts = _rule_parts(g, dotc, linc)
    for (_d, seg, _it) in parts:                      # all rays first
        for (r, c) in seg:
            g[r][c] = dotc
    inters = set()
    for (_d, _seg, it) in parts:
        inters |= set(it)
    for (r, c) in sorted(inters):                     # then every crossing frame
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        g[rr][cc] = linc
    return g


def _canonical_row(i, locs, cands):
    """True when nearest/second-nearest frontier are exactly the enclosing
    lines (or the dot sits outside the lines' span, where the rule is
    unambiguous).  Keeps generated instances free of border-frontier oddities
    and free of argmin ties."""
    if i < min(locs) or i > max(locs):
        return True
    sp = max(l for l in locs if l < i)
    ep = min(l for l in locs if l > i)
    dmax = max(i - sp, ep - i)
    return all(abs(i - c) > dmax for c in cands if c != sp and c != ep)


def _loose_row(i, locs, cands):
    if i < min(locs) or i > max(locs):
        return True
    d = sorted(abs(i - c) for c in cands)
    return len(d) < 3 or d[1] < d[2]


def _try_build(diff_lb, diff_ub, hb, wb, bgc, dotc, linc, canonical):
    h = _unifint(diff_lb, diff_ub, (4, hb))
    w = _unifint(diff_lb, diff_ub, (4, wb))
    g = [[bgc] * w for _ in range(h)]
    nl = _unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if not inds:
            break
        idx = random.choice(inds)
        locs.append(idx)
        bad = {idx - 2, idx - 1, idx, idx + 1, idx + 2}
        inds = [x for x in inds if x not in bad]
    if not locs:
        return None
    locs = sorted(locs)
    for r in locs:
        for c in range(w):
            g[r][c] = linc
    iopts = [i for i in range(h)
             if i not in locs and (i - 1) not in locs and (i + 1) not in locs]
    cands = sorted(set(locs) | {-1, h})
    if canonical:
        iopts = [i for i in iopts if _canonical_row(i, locs, cands)]
    else:
        iopts = [i for i in iopts if _loose_row(i, locs, cands)]
    if not iopts:
        return None
    jopts = list(range(w))
    ndots = _unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    placed = 0
    for _ in range(ndots):
        if not jopts:
            break
        i = random.choice(iopts)
        j = random.choice(jopts)
        g[i][j] = dotc
        placed += 1
        jopts = [x for x in jopts if x not in (j - 1, j, j + 1)]
    if placed == 0:
        return None
    flat = [v for row in g for v in row]
    cnt = Counter(flat)
    if len(cnt) != 3:
        return None
    if cnt[dotc] >= cnt[linc] or cnt[bgc] <= cnt[linc]:
        return None                                   # leastcolor/bg must be safe
    return g


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------

_VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, linc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        examples = [dict(v) for v in _VARIANTS]
        examples += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, linc=None,
             transpose=None, **kwargs) -> dict:
    if bgc is None or dotc is None or linc is None:
        bgc, dotc, linc = random.sample(list(range(10)), 3)
    if transpose is None:
        transpose = random.choice((True, False))

    hb = max(4, min(30, max_w if transpose else max_h))
    wb = max(4, min(30, max_h if transpose else max_w))

    gi = None
    for attempt in range(400):
        gi = _try_build(diff_lb, diff_ub, hb, wb, bgc, dotc, linc,
                        canonical=(attempt < 300))
        if gi is not None:
            break
    if gi is None:                                    # last-resort minimal grid
        gi = [[bgc] * 8 for _ in range(9)]
        for c in range(8):
            gi[4][c] = linc
        gi[1][2] = dotc

    if transpose:
        gi = [list(r) for r in zip(*gi)]

    go = _apply_rule(gi, dotc, linc)
    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------

def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    ranked = Counter(I.flatten().tolist()).most_common()
    bgc = ranked[0][0]                                # canvas colour: most cells
    dotc = ranked[-1][0]                              # dots: fewest cells
    rest = [c for c, _ in ranked if c != bgc and c != dotc]
    linc = rest[0] if rest else bgc                   # the frontier lines

    parts = _rule_parts(I.tolist(), dotc, linc)

    cur = I.copy()
    ops, sels = [], []

    # one dot at a time: grow its ray to the frontier(s), then thicken the
    # crossing(s) that ray makes with the line(s).
    for (dot, seg, inter) in sorted(parts, key=lambda p: p[0]):
        ray = [(r, c) for (r, c) in seg if cur[r, c] != dotc]
        if ray:
            ops.append(int(dotc))
            sels.append(sel_of(ray))
            for (r, c) in ray:
                cur[r, c] = dotc
        for (ir, ic) in sorted(inter):
            frame = [(ir + dr, ic + dc)
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if (dr or dc) and 0 <= ir + dr < h and 0 <= ic + dc < w]
            frame = [(r, c) for (r, c) in frame if cur[r, c] != linc]
            if frame:
                ops.append(int(linc))
                sels.append(sel_of(frame))
                for (r, c) in frame:
                    cur[r, c] = linc

    # safety net (never fires for instances produced by generate(): the same
    # rule builds them) -- repair per target colour rather than per pixel.
    if not np.array_equal(cur, O):
        for color in sorted(set(O[cur != O].tolist())):
            cells = [(r, c) for r in range(h) for c in range(w)
                     if cur[r, c] != O[r, c] and O[r, c] == color]
            if cells:
                ops.append(int(color))
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    cur[r, c] = color

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                 # whole grid: full rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
