"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in sorted({(int(a), int(b)) for a, b in cells})]}


ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1, c2, c3, c4 = random.sample(rem, 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": k} for k in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": k} for k in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, rot=None) -> dict:
    from collections import Counter, deque
    if rot is None:
        rot = random.choice(ROTS)
    rot = int(rot) % 4

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    mh = max(7, min(int(max_h), 30))
    mw = max(7, min(int(max_w), 30))
    lim_h, lim_w = (mh, mw) if rot % 2 == 0 else (mw, mh)

    for _attempt in range(400):
        h = unifint(diff_lb, diff_ub, (7, lim_h))
        w = unifint(diff_lb, diff_ub, (7, lim_w))
        loci1 = random.randint(1, h - 4)
        locj1 = random.randint(1, w - 4)
        loci1 -= unifint(diff_lb, diff_ub, (0, loci1 - 1))
        locj1 -= unifint(diff_lb, diff_ub, (0, locj1 - 1))
        loci2 = unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
        locj2 = unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))

        g = [[bgc] * w for _ in range(h)]
        fronts = [('h', loci1, c1), ('h', loci2, c2), ('v', locj1, c3), ('v', locj2, c4)]
        random.shuffle(fronts)
        for kind, idx, col in fronts:
            if kind == 'h':
                for j in range(w):
                    g[idx][j] = col
            else:
                for i in range(h):
                    g[i][idx] = col

        cands = [(i, j) for i in range(h) for j in range(w) if g[i][j] == bgc]
        nn = len(cands)
        if nn < 8:
            continue
        nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        for (i, j) in random.sample(cands, nnoise):
            g[i][j] = c1

        inner = [(i, j) for i in range(loci1 + 1, loci2) for j in range(locj1 + 1, locj2)]
        inner_set = set(inner)

        # every background region outside the framed rectangle must reach the
        # grid border, else it counts as an enclosed region and moves the crop
        def bg_components():
            seen = [[False] * w for _ in range(h)]
            comps = []
            for si in range(h):
                for sj in range(w):
                    if g[si][sj] != bgc or seen[si][sj]:
                        continue
                    dq = deque([(si, sj)])
                    seen[si][sj] = True
                    comp = []
                    edge = False
                    while dq:
                        i, j = dq.popleft()
                        comp.append((i, j))
                        if i == 0 or j == 0 or i == h - 1 or j == w - 1:
                            edge = True
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and not seen[ni][nj] and g[ni][nj] == bgc:
                                seen[ni][nj] = True
                                dq.append((ni, nj))
                    comps.append((comp, edge))
            return comps

        for comp, edge in bg_components():
            if not edge and comp[0] not in inner_set:
                for (i, j) in comp:
                    g[i][j] = c1

        # the framed rectangle must keep background on each of its four inner
        # edges, and must carry at least one noise cell
        edges = [
            [(loci1 + 1, j) for j in range(locj1 + 1, locj2)],
            [(loci2 - 1, j) for j in range(locj1 + 1, locj2)],
            [(i, locj1 + 1) for i in range(loci1 + 1, loci2)],
            [(i, locj2 - 1) for i in range(loci1 + 1, loci2)],
        ]
        for ln in edges:
            if not any(g[i][j] == bgc for (i, j) in ln):
                i, j = random.choice(ln)
                g[i][j] = bgc
        if len(inner) == 1:
            g[inner[0][0]][inner[0][1]] = bgc
        elif not any(g[i][j] == c1 for (i, j) in inner):
            free = []
            for (i, j) in inner:
                if g[i][j] != bgc:
                    continue
                keep = True
                for ln in edges:
                    if (i, j) in ln and sum(1 for (a, b) in ln if g[a][b] == bgc) < 2:
                        keep = False
                if keep:
                    free.append((i, j))
            if not free:
                continue
            i, j = random.choice(free)
            g[i][j] = c1

        # validation: background stays the clear majority colour, only the four
        # painted lines read as lines, and the enclosed-background bbox is
        # exactly the framed rectangle's interior
        bad = False
        tally = Counter(v for row in g for v in row)
        if any(n + 1 >= tally[bgc] for v, n in tally.items() if v != bgc):
            continue
        for r in range(h):
            if r in (loci1, loci2):
                continue
            for v, n in Counter(g[r]).items():
                if v != bgc and n >= w - 2:
                    bad = True
        for c in range(w):
            if c in (locj1, locj2):
                continue
            col = [g[i][c] for i in range(h)]
            for v, n in Counter(col).items():
                if v != bgc and n >= h - 2:
                    bad = True
        if bad:
            continue
        encl = [cell for comp, edge in bg_components() if not edge for cell in comp]
        if not encl:
            continue
        if (min(i for i, _ in encl), max(i for i, _ in encl),
                min(j for _, j in encl), max(j for _, j in encl)) != \
                (loci1 + 1, loci2 - 1, locj1 + 1, locj2 - 1):
            continue

        H = loci2 - loci1 + 1
        W = locj2 - locj1 + 1
        go = [row[locj1:locj2 + 1] for row in g[loci1:loci2 + 1]]
        for j in range(W):
            far = -1
            for i in range(H):
                if go[i][j] == c1:
                    far = i
            for i in range(far + 1):
                go[i][j] = c1

        gi = np.rot90(np.array(g, dtype=int), rot)
        go = np.rot90(np.array(go, dtype=int), rot)
        return {"input": gi.tolist(), "output": go.tolist()}

    raise RuntimeError("generation failed")


def _read_scene(I):
    """Read the framed rectangle off the INPUT alone.

    Four painted lines — two full rows, two full columns, four different
    colours — box in a rectangle; everything else is background plus speckles.
    A line is a row/column carrying one non-background colour everywhere
    except where the perpendicular pair crosses it, so at most two breaks.
    The rectangle that counts is the one the lines enclose: background inside
    it cannot walk out to the grid border.
    """
    from collections import Counter, deque
    h, w = I.shape
    for bgc, _cnt in Counter(I.flatten().tolist()).most_common():
        rows, cols = [], []
        for r in range(h):
            for v, n in Counter(I[r, :].tolist()).items():
                if v != bgc and n >= w - 2:
                    rows.append((r, v))
        for c in range(w):
            for v, n in Counter(I[:, c].tolist()).items():
                if v != bgc and n >= h - 2:
                    cols.append((c, v))
        if len(rows) < 2 or len(cols) < 2:
            continue
        # background that the grid border can reach by walking on background
        out = [[False] * w for _ in range(h)]
        dq = deque()
        for i in range(h):
            for j in range(w):
                if (i in (0, h - 1) or j in (0, w - 1)) and I[i, j] == bgc and not out[i][j]:
                    out[i][j] = True
                    dq.append((i, j))
        while dq:
            i, j = dq.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and not out[ni][nj] and I[ni, nj] == bgc:
                    out[ni][nj] = True
                    dq.append((ni, nj))
        best = None
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                r1, vt = rows[a]
                r2, vb = rows[b]
                if r2 - r1 < 2:
                    continue
                for x in range(len(cols)):
                    for y in range(x + 1, len(cols)):
                        cc1, vl = cols[x]
                        cc2, vr = cols[y]
                        if cc2 - cc1 < 2:
                            continue
                        if len({vt, vb, vl, vr}) != 4 or bgc in (vt, vb, vl, vr):
                            continue
                        ok = True
                        for rr, vv in ((r1, vt), (r2, vb)):
                            for j in range(w):
                                if j not in (cc1, cc2) and I[rr, j] != vv:
                                    ok = False
                        for ccx, vv in ((cc1, vl), (cc2, vr)):
                            for i in range(h):
                                if i not in (r1, r2) and I[i, ccx] != vv:
                                    ok = False
                        if not ok:
                            continue
                        # inside: background plus speckles of a single colour,
                        # and that colour is one of the four lines'
                        pal = set(I[r1 + 1:r2, cc1 + 1:cc2].flatten().tolist()) - {bgc}
                        if len(pal) > 1 or (pal and next(iter(pal)) not in (vt, vb, vl, vr)):
                            continue
                        shut = sum(1 for i in range(r1 + 1, r2) for j in range(cc1 + 1, cc2)
                                   if I[i, j] == bgc and not out[i][j])
                        if shut == 0:
                            continue
                        score = (shut, (r2 - r1) * (cc2 - cc1))
                        if best is None or score > best[0]:
                            best = (score, (bgc, (r1, r2, cc1, cc2), (vt, vb, vl, vr)))
        if best is not None:
            return best[1]
    return None


def derive_operations(I, O=None, examples=None):
    """Cut out the framed rectangle and grow every speckle inside it into a
    ray that runs to the frame side painted in the speckles' own colour.

    Only the input is read: the frame, the speckle colour, the direction and
    the reach of each ray all come off I, so a different answer grid would
    not change a single operation.  What does change between instances is the
    ROUTE — where the sweep starts, whether the rectangle is cut out before or
    after the rays are drawn, and whether neighbouring rays of equal reach are
    filled as one block — and each of those is decided by the input too.

    Nothing here is carried by a Copy, a Paste or a Move, so a picture that
    uses colour 0 survives: CropGrid keeps a 0 cell as 0, and a ray whose own
    colour is 0 is laid down by an explicit Color0 on its cells.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, (r1, r2, cc1, cc2), (vt, vb, vl, vr) = _read_scene(I)
    H, W = r2 - r1 + 1, cc2 - cc1 + 1
    sub = I[r1:r2 + 1, cc1:cc2 + 1]

    # the speckles inside the frame are all one colour, and it is the colour
    # of exactly one of the four sides: that side is where the rays point
    pal = sorted(set(sub[1:H - 1, 1:W - 1].flatten().tolist()) - {bgc})
    nc = pal[0] if len(pal) == 1 else None
    side = {vt: 'up', vb: 'down', vl: 'left', vr: 'right'}.get(nc) if nc is not None else None
    vertical = side in ('up', 'down')

    def seg_cells(k, lo, hi, k2=None):
        """cells of one ray (k2 is None) or of a block of rays k..k2"""
        k2 = k if k2 is None else k2
        if vertical:
            return [(t, j) for j in range(k, k2 + 1) for t in range(lo, hi + 1)]
        return [(i, t) for i in range(k, k2 + 1) for t in range(lo, hi + 1)]

    # one ray per line that holds a speckle: from the frame side out to the
    # farthest speckle on that line
    rays = []
    if side is not None:
        lines = range(1, W - 1) if vertical else range(1, H - 1)
        span = (H - 2) if vertical else (W - 2)
        for k in lines:
            hits = [t for t in range(1, span + 1)
                    if (sub[t, k] if vertical else sub[k, t]) == nc]
            if not hits:
                continue
            lo, hi = (1, max(hits)) if side in ('up', 'left') else (min(hits), span)
            if all(sub[a, b] == nc for a, b in seg_cells(k, lo, hi)):
                continue          # this ray is solid already, nothing to draw
            rays.append((k, lo, hi))

    # ── route choices, all read off the input ────────────────────────────────
    n_lines = (W - 2) if vertical else (H - 2)
    # 1. a rectangle that is a small corner of a big cluttered picture is cut
    #    out first and worked on small; one that takes up a good share of the
    #    picture is drawn on where it stands and cut out at the end
    crop_first = 6 * H * W <= h * w
    # 2. start the sweep at the half of the frame the longest ray sits in and
    #    work across to the other end
    forward = True
    if rays:
        deepest = max(rays, key=lambda t: (t[2] - t[1], -t[0]))[0]
        forward = (deepest - 1) <= (n_lines - deepest)
    # 3. when the rays are dense, neighbouring ones of equal reach are filled
    #    as a single block; when they are sparse, one ray at a time
    dense = 2 * len(rays) >= n_lines

    blocks = []
    for k, lo, hi in rays:
        if dense and blocks and blocks[-1][1] == lo and blocks[-1][2] == hi and blocks[-1][3] + 1 == k:
            blocks[-1][3] = k
        else:
            blocks.append([k, lo, hi, k])
    if not forward:
        blocks.reverse()

    ops, sels = [], []
    if crop_first:
        # keep only the framed rectangle; the selection is exactly that full
        # rectangle (background included), so a bbox is exact here
        ops.append(33)
        sels.append([r1, cc1, H - 1, W - 1])
    dr, dc = (0, 0) if crop_first else (r1, cc1)
    for k, lo, hi, k2 in blocks:
        cells = [(a + dr, b + dc) for a, b in seg_cells(k, lo, hi, k2)]
        if k2 > k:
            # a solid block of equal-reach rays: the bbox is exactly its cells
            rs = [a for a, _ in cells]
            cs = [b for _, b in cells]
            ops.append(int(nc))
            sels.append([min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)])
        else:
            ops.append(int(nc))
            sels.append(sel_of(cells))
    if not crop_first:
        ops.append(33)
        sels.append([r1, cc1, H - 1, W - 1])
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
