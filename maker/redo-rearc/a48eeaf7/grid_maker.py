"""
ARC Task: a48eeaf7 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """bgc / sqc / dotc are all sampled by the original generator -> fix them per episode."""
    cols = list(range(10))
    bgc, sqc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, dotc=None) -> dict:
    if bgc is None or sqc is None or dotc is None:
        cols = list(range(10))
        bgc, sqc, dotc = random.sample(cols, 3)

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        return random.randint(lb + int((ub - lb) * diff_lb), lb + int((ub - lb) * diff_ub))

    hub = max(8, min(30, int(max_h)))
    wub = max(8, min(30, int(max_w)))
    h = unifint(8, hub)
    w = unifint(8, wub)
    ih = unifint(2, h // 2)
    iw = unifint(2, w // 2)
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
            go[r][c] = sqc

    A = [(x, locj - 1) for x in range(loci, loci + ih)]
    Ap = [(x, random.randint(0, locj - 2)) for x in range(loci, loci + ih)]
    B = [(x, locj + iw) for x in range(loci, loci + ih)]
    Bp = [(x, random.randint(locj + iw + 1, w - 1)) for x in range(loci, loci + ih)]
    C = [(loci - 1, x) for x in range(locj, locj + iw)]
    Cp = [(random.randint(0, loci - 2), x) for x in range(locj, locj + iw)]
    D = [(loci + ih, x) for x in range(locj, locj + iw)]
    Dp = [(random.randint(loci + ih + 1, h - 1), x) for x in range(locj, locj + iw)]

    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    num = unifint(1, len(srarr))
    locs = set(random.sample(range(len(srarr)), num))
    for j in sorted(locs):
        sr, sc = srarr[j]
        dr, dc = dearr[j]
        gi[sr][sc] = dotc
        go[dr][dc] = dotc

    ncorn = unifint(0, 4)

    def ray(start, d):
        pts = []
        r, c = start
        while 0 <= r < h and 0 <= c < w:
            pts.append((r, c))
            r += d[0]
            c += d[1]
        return pts

    corner_defs = [
        ((loci - 1, locj - 1), (loci - 2, locj - 2), (-1, -1)),
        ((loci - 1, locj + iw), (loci - 2, locj + iw + 1), (-1, 1)),
        ((loci + ih, locj - 1), (loci + ih + 1, locj - 2), (1, -1)),
        ((loci + ih, locj + iw), (loci + ih + 1, locj + iw + 1), (1, 1)),
    ]
    for k in range(min(4, ncorn)):
        tgt, st, d = corner_defs[k]
        go[tgt[0]][tgt[1]] = dotc
        pts = ray(st, d)
        if pts:
            rr, cc = random.choice(pts)
            gi[rr][cc] = dotc

    def rot90cw(g):
        return [list(row) for row in zip(*g[::-1])]

    for _ in range(random.choice([0, 1, 2, 3])):
        gi = rot90cw(gi)
        go = rot90cw(go)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Every scattered dot travels to the nearest cell of the ring (outbox) that
    surrounds the solid rectangle.  Each dot is SLID there with Move ops (grab once,
    then empty selections), and its vacated cell is repaired with one Color(bgc).
    When the dot colour is 0 ARCLE cannot grab it (object buffer keeps only nonzero
    cells), so those dots are erased at the source and drawn at the destination."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    ops, sels = [], []
    g = I.copy()

    # ---- colour roles -------------------------------------------------------
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    others = [c for c in cnt if c != bgc]

    def bbox_of(color):
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == color))]
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return cells, (min(rs), max(rs), min(cs), max(cs))

    best = None
    for c in others:
        cells, (r0, r1, c0, c1) = bbox_of(c)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(cells) >= 4:
            if best is None or len(cells) > best[0]:
                best = (len(cells), c, (r0, r1, c0, c1))
    if best is None and others:
        c = max(others, key=lambda x: cnt[x])
        cells, box = bbox_of(c)
        best = (len(cells), c, box)
    if best is None:
        ops.append(34)
        sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
        return ops, sels

    sqc = best[1]
    r0, r1, c0, c1 = best[2]
    dot_cols = [c for c in others if c != sqc]
    if not dot_cols:
        ops.append(34)
        sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
        return ops, sels
    dotc = dot_cols[0]

    # ---- the ring (outbox) around the rectangle ----------------------------
    ring = set()
    for r in range(r0 - 1, r1 + 2):
        ring.add((r, c0 - 1))
        ring.add((r, c1 + 1))
    for c in range(c0 - 1, c1 + 2):
        ring.add((r0 - 1, c))
        ring.add((r1 + 1, c))
    ring = sorted(p for p in ring if 0 <= p[0] < h and 0 <= p[1] < w)

    sources = [(int(r), int(c)) for r, c in zip(*np.where(I == dotc))]

    def nearest(src):
        return min(ring, key=lambda p: (abs(p[0] - src[0]) + abs(p[1] - src[1]), p[0], p[1]))

    pairs = [(s, nearest(s)) for s in sources]

    corners = {(r0 - 1, c0 - 1): 0, (r0 - 1, c1 + 1): 1,
               (r1 + 1, c0 - 1): 2, (r1 + 1, c1 + 1): 3}

    def order_key(item):
        (sr, sc), (tr, tc) = item
        if (tr, tc) in corners:
            return (1, corners[(tr, tc)], tr, tc)
        if tr == r0 - 1:
            return (0, 0, tc, tr)      # top edge, left -> right
        if tc == c1 + 1:
            return (0, 1, tr, tc)      # right edge, top -> bottom
        if tr == r1 + 1:
            return (0, 2, tc, tr)      # bottom edge, left -> right
        return (0, 3, tr, tc)          # left edge, top -> bottom

    pairs.sort(key=order_key)

    # ---- emitters -----------------------------------------------------------
    def paint(cells, color):
        cells = [(r, c) for (r, c) in cells if int(g[r, c]) != int(color)]
        if not cells:
            return
        ops.append(int(color))
        sels.append(sel_of(cells))
        for (r, c) in cells:
            g[r, c] = color

    def slide(src, tgt):
        """Grab the dot once, then walk it with empty selections. Returns False if
        the object cannot be grabbed (colour 0) or a step would be invisible."""
        sr, sc = src
        tr, tc = tgt
        color = int(g[sr, sc])
        if color == 0:
            return False
        dr, dc = tr - sr, tc - sc
        steps = []
        if dr:
            steps += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
        if dc:
            steps += [(22 if dc > 0 else 23, (0, 1 if dc > 0 else -1))] * abs(dc)
        if not steps:
            return True
        snap = g.copy()
        snap[sr, sc] = 0                      # ARCLE zeroes the grabbed cell
        prev = g
        cur = (sr, sc)
        frames = []
        for op, d in steps:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            ng = snap.copy()
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w:
                ng[nxt[0], nxt[1]] = color
            if np.array_equal(ng, prev):
                return False                  # invisible step -> use the paint fallback
            frames.append((op, ng))
            prev = ng
            cur = nxt
        for i, (op, ng) in enumerate(frames):
            ops.append(op)
            sels.append(sel_of([src]) if i == 0 else sel_of([]))
        g[:, :] = frames[-1][1]
        # the only 0 left is the dot's original footprint
        if cur != (sr, sc) and int(g[sr, sc]) != int(bgc):
            ops.append(int(bgc))
            sels.append(sel_of([(sr, sc)]))
            g[sr, sc] = bgc
        return True

    for src, tgt in pairs:
        if src == tgt:
            continue
        if not slide(src, tgt):
            # dot colour is 0 (ungrabbable): erase it where it is, draw it on the ring
            paint([src], bgc)
            paint([tgt], dotc)

    # ---- last-resort consistency guard (should never fire) ------------------
    if not np.array_equal(g, O):
        leftovers = {}
        for r in range(h):
            for c in range(w):
                if int(g[r, c]) != int(O[r, c]):
                    leftovers.setdefault(int(O[r, c]), []).append((r, c))
        for color, cells in leftovers.items():
            paint(cells, color)

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
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
                                f"for task a48eeaf7"
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
                    f"Failed to build a complete episode for task a48eeaf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a48eeaf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
