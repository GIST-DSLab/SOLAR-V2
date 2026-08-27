"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    def find_frame(bgc):
        # the four painted lines: full rows / columns of one non-background
        # colour, broken at most where the perpendicular pair crosses them
        cand_r = []
        for r in range(h):
            for v, n in Counter(I[r].tolist()).items():
                if v != bgc and n >= w - 2:
                    cand_r.append((r, v))
                    break
        cand_c = []
        for c in range(w):
            for v, n in Counter(I[:, c].tolist()).items():
                if v != bgc and n >= h - 2:
                    cand_c.append((c, v))
                    break
        best = None
        for a in range(len(cand_r)):
            for b in range(a + 1, len(cand_r)):
                r1, vt = cand_r[a]
                r2, vb = cand_r[b]
                if r2 - r1 < 2:
                    continue
                for x in range(len(cand_c)):
                    for y in range(x + 1, len(cand_c)):
                        cc1, vl = cand_c[x]
                        cc2, vr = cand_c[y]
                        if cc2 - cc1 < 2:
                            continue
                        four = [vt, vb, vl, vr]
                        if len(set(four)) != 4:
                            continue
                        ok = True
                        for rr, vv in ((r1, vt), (r2, vb)):
                            for j in range(w):
                                if I[rr, j] != vv and j not in (cc1, cc2):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            for ccx, vv in ((cc1, vl), (cc2, vr)):
                                for i in range(h):
                                    if I[i, ccx] != vv and i not in (r1, r2):
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if not ok:
                            continue
                        pal = set(I[r1 + 1:r2, cc1 + 1:cc2].flatten().tolist()) - {bgc}
                        if len(pal) > 1:
                            continue
                        if len(pal) == 1 and next(iter(pal)) not in four:
                            continue
                        score = (1 if O.shape == (r2 - r1 + 1, cc2 - cc1 + 1) else 0,
                                 (r2 - r1) * (cc2 - cc1))
                        if best is None or score > best[0]:
                            best = (score, (r1, r2, cc1, cc2, vt, vb, vl, vr))
        return best

    bgc, frame = None, None
    for cand_bg, _n in Counter(I.flatten().tolist()).most_common():
        frame = find_frame(cand_bg)
        if frame is not None:
            bgc = cand_bg
            break
    r1, r2, cc1, cc2, vt, vb, vl, vr = frame[1]
    H = r2 - r1 + 1
    W = cc2 - cc1 + 1
    sub = I[r1:r2 + 1, cc1:cc2 + 1]

    ops, sels = [], []
    # keep only the rectangle framed by the four lines; the selection is
    # exactly that full rectangle (background included), so a bbox is exact
    ops.append(33)
    sels.append([r1, cc1, H - 1, W - 1])

    pal = set(sub[1:H - 1, 1:W - 1].flatten().tolist()) - {bgc}
    if len(pal) == 1:
        nc = next(iter(pal))
        # the speckles grow into a ray aimed at whichever frame side carries
        # their own colour: one ray per column (or row) of the rectangle,
        # reaching from that side out to its farthest speckle
        side = None
        if vt == nc:
            side = 'up'
        elif vb == nc:
            side = 'down'
        elif vl == nc:
            side = 'left'
        elif vr == nc:
            side = 'right'
        if side is not None:
            cur = sub.copy()
            if side in ('up', 'down'):
                for j in range(1, W - 1):
                    hits = [i for i in range(1, H - 1) if sub[i, j] == nc]
                    if not hits:
                        continue
                    if side == 'up':
                        seg = [(i, j) for i in range(1, max(hits) + 1)]
                    else:
                        seg = [(i, j) for i in range(min(hits), H - 1)]
                    if all(cur[a, b] == nc for a, b in seg):
                        continue
                    ops.append(int(nc))
                    sels.append(sel_of(seg))
                    for a, b in seg:
                        cur[a, b] = nc
            else:
                for i in range(1, H - 1):
                    hits = [j for j in range(1, W - 1) if sub[i, j] == nc]
                    if not hits:
                        continue
                    if side == 'left':
                        seg = [(i, j) for j in range(1, max(hits) + 1)]
                    else:
                        seg = [(i, j) for j in range(min(hits), W - 1)]
                    if all(cur[a, b] == nc for a, b in seg):
                        continue
                    ops.append(int(nc))
                    sels.append(sel_of(seg))
                    for a, b in seg:
                        cur[a, b] = nc

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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
