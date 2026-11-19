"""
ARC Task: b782dc8a (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    """All four colours of the maze picture are sampled once per episode:
    the lattice colour, the gap colour, the dot colour and the neighbour
    colour.  Any of them may be 0 -- the route below only ever uses Color
    ops, so a 0-coloured role is painted with an explicit Color0."""
    import random
    cols = list(range(10))
    pathcol, wallcol, dotcol, ncol = random.sample(cols, 4)
    return {"pathcol": pathcol, "wallcol": wallcol,
            "dotcol": dotcol, "ncol": ncol}


def generate(diff_lb, diff_ub, max_h, max_w,
             pathcol=None, wallcol=None, dotcol=None, ncol=None) -> dict:
    import random
    import numpy as np

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if pathcol is None or wallcol is None or dotcol is None or ncol is None:
        pathcol, wallcol, dotcol, ncol = random.sample(list(range(10)), 4)

    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    walls = {'N': True, 'S': True, 'E': True, 'W': True}
    dirs4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

    rotk = random.choice((0, 1, 2, 3))
    if rotk % 2 == 0:
        h_ub = min(15, (max_h + 1) // 2)
        w_ub = min(15, (max_w + 1) // 2)
    else:
        h_ub = min(15, (max_w + 1) // 2)
        w_ub = min(15, (max_h + 1) // 2)
    h_ub = max(3, h_ub)
    w_ub = max(3, w_ub)

    while True:
        h = unifint(diff_lb, diff_ub, (3, h_ub))
        w = unifint(diff_lb, diff_ub, (3, w_ub))

        maze = [[{'x': x, 'y': y, 'walls': dict(walls)} for y in range(h)]
                for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        broke = False
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    neighbour = maze[x2][y2]
                    if all(neighbour['walls'].values()):
                        nbhs.append((direc, neighbour))
            if not nbhs:
                if not stck:
                    broke = True
                    break
                cc = stck.pop()
                continue
            direc, next_cell = random.choice(nbhs)
            cc['walls'][direc] = False
            next_cell['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = next_cell
            nv += 1
        if broke:
            continue

        grid = [[pathcol for _ in range(w * 2)]]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]['walls']['E'] else wallcol)
            grid.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]['walls']['S'] else wallcol)
                row.append(pathcol)
            grid.append(row)
        gi = [list(r[1:-1]) for r in grid[1:-1]]

        H, W = len(gi), len(gi[0])
        if H < 3 or W < 3:
            continue

        # connected components of the lattice colour, keep the big ones
        seen = [[False] * W for _ in range(H)]
        objs = []
        for r in range(H):
            for c in range(W):
                if gi[r][c] == pathcol and not seen[r][c]:
                    comp = []
                    stack = [(r, c)]
                    seen[r][c] = True
                    while stack:
                        rr, cc2 = stack.pop()
                        comp.append((rr, cc2))
                        for dr, dc in dirs4:
                            nr, nc = rr + dr, cc2 + dc
                            if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] \
                                    and gi[nr][nc] == pathcol:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                    if len(comp) > 4:
                        objs.append(comp)
        if not objs:
            continue
        objs.sort(key=len)
        idx = unifint(diff_lb, diff_ub, (0, len(objs) - 1))
        obj = objs[idx]

        cr, cc0 = random.choice(obj)
        gin = [row[:] for row in gi]
        gin[cr][cc0] = dotcol
        for dr, dc in dirs4:
            nr, nc = cr + dr, cc0 + dc
            if 0 <= nr < H and 0 <= nc < W and gin[nr][nc] == pathcol:
                gin[nr][nc] = ncol

        go = [row[:] for row in gin]
        for (r, c) in obj:
            if (abs(r - cr) + abs(c - cc0)) % 2 == 0:
                go[r][c] = dotcol
            else:
                go[r][c] = ncol

        ai = np.rot90(np.array(gin, dtype=int), k=rotk)
        ao = np.rot90(np.array(go, dtype=int), k=rotk)
        if ai.shape[0] > max_h or ai.shape[1] > max_w:
            continue

        return {"input": tuple(tuple(int(v) for v in row) for row in ai.tolist()),
                "output": tuple(tuple(int(v) for v in row) for row in ao.tolist())}


def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone): the picture is a maze lattice.  One lattice cell
    carries a dot, and the lattice cells touching the dot carry a second
    marker colour.  The two marker colours then alternate over the WHOLE
    lattice component the dot sits in: every cell at even manhattan distance
    from the dot takes the dot's colour, every odd one the neighbour colour.

    O is never consulted.  Which route is taken through that one rule -- bulk
    strokes, a base coat with the alternates drawn over it, branch by branch,
    or a cell-by-cell walk outward, and in which direction -- is decided by
    the dot's position, the colours and the size of the component, so the same
    picture always gets the same route and different pictures differ.
    """
    import numpy as np
    from collections import Counter, deque
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def emit(col, cells):
        if not cells:
            return
        ops.append(int(col))
        sels.append(sel_of(sorted(set(cells))))

    def finish():
        ops.append(34)
        sels.append(full)
        return ops, sels

    # ---- the lattice ---------------------------------------------------
    # the maze is drawn on a fixed parity lattice: (odd,odd) positions always
    # hold the lattice colour, (even,even) positions always hold the gaps.
    lat = Counter(int(I[r, c]) for r in range(1, h, 2) for c in range(1, w, 2))
    gap = Counter(int(I[r, c]) for r in range(0, h, 2) for c in range(0, w, 2))
    if not lat or not gap:
        return finish()
    corridor_col = lat.most_common(1)[0][0]
    wall_col = gap.most_common(1)[0][0]

    cnt = Counter(int(v) for v in I.flatten().tolist())
    marks = sorted([c for c in cnt if c != corridor_col and c != wall_col],
                   key=lambda c: (cnt[c], c))
    if len(marks) < 2:
        return finish()
    # the dot is the rarer marker; on a 1-1 tie the two markers are adjacent,
    # so starting from either and keeping ITS own colour for the even cells
    # gives the identical alternation.
    seed_col, other_col = marks[0], marks[1]

    seed_cells = [(r, c) for r in range(h) for c in range(w)
                  if int(I[r, c]) == seed_col]
    if not seed_cells:
        return finish()
    seed = None
    for (r, c) in seed_cells:
        for dr, dc in D4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(I[nr, nc]) == other_col:
                seed = (r, c)
                break
        if seed is not None:
            break
    if seed is None:
        seed = seed_cells[0]
    sr, sc = seed

    # ---- the one lattice component the dot lives in ---------------------
    member = {corridor_col, seed_col, other_col}
    compset = {seed}
    dq = deque([seed])
    while dq:
        r, c = dq.popleft()
        for dr, dc in D4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in compset \
                    and int(I[nr, nc]) in member:
                compset.add((nr, nc))
                dq.append((nr, nc))

    par = (sr + sc) % 2
    targets = [p for p in sorted(compset) if int(I[p[0], p[1]]) == corridor_col]
    if not targets:
        return finish()
    evens = [p for p in targets if (p[0] + p[1]) % 2 == par]   # -> seed_col
    odds = [p for p in targets if (p[0] + p[1]) % 2 != par]    # -> other_col

    # ---- which route this particular picture gets -----------------------
    style = (sr + sc + len(targets)) % 4
    swap = (corridor_col + wall_col) % 2
    rot = (sr + wall_col) % 4
    base_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dirs = base_dirs[rot:] + base_dirs[:rot]
    if (sc + seed_col) % 2:
        dirs = list(reversed(dirs))

    if style == 0 and evens and odds:
        # lay the whole corridor down in one colour, then dot the alternates
        # on top of it
        if swap:
            emit(other_col, targets)
            emit(seed_col, evens)
        else:
            emit(seed_col, targets)
            emit(other_col, odds)

    elif style == 2:
        # branch by branch, out of the dot: each branch gets its two strokes
        rest = compset - {seed}
        seen2 = set()
        arms = []
        for p in sorted(rest):
            if p in seen2:
                continue
            seen2.add(p)
            stack = [p]
            arm = []
            while stack:
                q = stack.pop()
                arm.append(q)
                for dr, dc in D4:
                    n = (q[0] + dr, q[1] + dc)
                    if n in rest and n not in seen2:
                        seen2.add(n)
                        stack.append(n)
            root = None
            for q in arm:
                if abs(q[0] - sr) + abs(q[1] - sc) == 1:
                    root = q
                    break
            if root is None:
                root = min(arm, key=lambda q: (abs(q[0] - sr) + abs(q[1] - sc), q))
            d = (root[0] - sr, root[1] - sc)
            key = dirs.index(d) if d in dirs else len(dirs)
            arms.append((key, root, arm))
        arms.sort(key=lambda t: (t[0], t[1]))
        tgt = set(targets)
        for _, _, arm in arms:
            a_even = [p for p in sorted(arm) if p in tgt and (p[0] + p[1]) % 2 == par]
            a_odd = [p for p in sorted(arm) if p in tgt and (p[0] + p[1]) % 2 != par]
            emit(seed_col, a_even)      # the cells nearest the dot first
            emit(other_col, a_odd)

    elif style == 3:
        # walk the corridor outward from the dot, marking cell by cell
        use_dfs = bool((sc + corridor_col) % 2)
        seenw = {seed}
        order = []
        if use_dfs:
            stack = [seed]
            while stack:
                p = stack.pop()
                order.append(p)
                for d in reversed(dirs):
                    q = (p[0] + d[0], p[1] + d[1])
                    if q in compset and q not in seenw:
                        seenw.add(q)
                        stack.append(q)
        else:
            dq = deque([seed])
            while dq:
                p = dq.popleft()
                order.append(p)
                for d in dirs:
                    q = (p[0] + d[0], p[1] + d[1])
                    if q in compset and q not in seenw:
                        seenw.add(q)
                        dq.append(q)
        tgt = set(targets)
        for p in order:
            if p in tgt:
                col = seed_col if (p[0] + p[1]) % 2 == par else other_col
                emit(col, [p])

    else:
        # two strokes, one per alternating class
        strokes = [(seed_col, evens), (other_col, odds)]
        if swap:
            strokes.reverse()
        for col, cells in strokes:
            emit(col, cells)

    return finish()


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
                        f"num_examples+1 ({num_examples + 1}) for task b782dc8a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b782dc8a"
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
                                f"for task b782dc8a"
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
                    f"Failed to build a complete episode for task b782dc8a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b782dc8a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
