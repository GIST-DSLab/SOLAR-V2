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
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """pathcol / wallcol / dotcol / ncol are all sampled randomly by the original
    generator, so all four are fixed once per episode."""
    cols = list(range(10))
    pathcol, wallcol, dotcol, ncol = random.sample(cols, 4)
    return {"pathcol": pathcol, "wallcol": wallcol, "dotcol": dotcol, "ncol": ncol}


def generate(diff_lb, diff_ub, max_h, max_w, pathcol, wallcol, dotcol, ncol) -> dict:

    def unifint(bounds):
        a, b = bounds
        lo = max(a, min(b, int(a + (b - a) * diff_lb)))
        hi = max(a, min(b, int(a + (b - a) * diff_ub)))
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def dnbs(r, c, H, W):
        return [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= rr < H and 0 <= cc < W]

    def comps_of(grid, col):
        H = len(grid)
        W = len(grid[0])
        seen = [[False] * W for _ in range(H)]
        out = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == col and not seen[r][c]:
                    seen[r][c] = True
                    stack = [(r, c)]
                    comp = []
                    while stack:
                        rr, cc = stack.pop()
                        comp.append((rr, cc))
                        for nr, nc2 in dnbs(rr, cc, H, W):
                            if not seen[nr][nc2] and grid[nr][nc2] == col:
                                seen[nr][nc2] = True
                                stack.append((nr, nc2))
                    out.append(comp)
        return out

    def flood(G, start, allowed):
        H, W = G.shape
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in dnbs(r, c, H, W):
                if p not in seen and int(G[p]) in allowed:
                    seen.add(p)
                    stack.append(p)
        return seen

    def identify(G):
        # the maze lattice has wallcol at every (even row, even col) cell, and the
        # grid dims are odd so the random rotation preserves that class
        H, W = G.shape
        wcol = int(G[0, 0])
        pal = sorted(set(int(v) for v in G.reshape(-1)))
        others = [c for c in pal if c != wcol]
        if len(others) != 3:
            return None
        found = []
        for dc in others:
            dpos = [(r, c) for r in range(H) for c in range(W) if int(G[r, c]) == dc]
            if len(dpos) != 1:
                continue
            d = dpos[0]
            nb = dnbs(d[0], d[1], H, W)
            for nc in others:
                if nc == dc:
                    continue
                pc = [x for x in others if x != dc and x != nc][0]
                npos = [(r, c) for r in range(H) for c in range(W) if int(G[r, c]) == nc]
                if not (1 <= len(npos) <= 4):
                    continue
                if any(p not in nb for p in npos):
                    continue
                if any(int(G[p]) == pc for p in nb):
                    continue
                region = flood(G, d, {pc, dc, nc})
                if len(region) <= 4:
                    continue
                found.append((dc, nc, d, region))
        if len(found) != 1:
            return None
        return found[0]

    def apply_rule(G):
        res = identify(G)
        if res is None:
            return None
        dc, nc, d, region = res
        out = G.copy()
        for (r, c) in region:
            out[r, c] = dc if (abs(r - d[0]) + abs(c - d[1])) % 2 == 0 else nc
        return out

    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    hlim = min(15, (max_h + 1) // 2, (max_w + 1) // 2)
    if hlim < 3:
        hlim = 3

    while True:
        h = unifint((3, hlim))
        w = unifint((3, hlim))
        maze = [[{'x': x, 'y': y, 'walls': {'N': True, 'S': True, 'E': True, 'W': True}}
                 for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        ok = True
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    nbr = maze[x2][y2]
                    if all(nbr['walls'].values()):
                        nbhs.append((direc, nbr))
            if not nbhs:
                if not stck:
                    ok = False
                    break
                cc = stck.pop()
                continue
            direc, nxt = random.choice(nbhs)
            cc['walls'][direc] = False
            nxt['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = nxt
            nv += 1
        if not ok:
            continue

        rows = [[pathcol] * (w * 2)]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]['walls']['E'] else wallcol)
            rows.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]['walls']['S'] else wallcol)
                row.append(pathcol)
            rows.append(row)
        gi = [r[1:-1] for r in rows[1:-1]]

        comps = [cp for cp in comps_of(gi, pathcol) if len(cp) > 4]
        if not comps:
            continue
        comps.sort(key=len)
        obj = comps[unifint((0, len(comps) - 1))]
        cell = random.choice(sorted(obj))

        G = np.array(gi, dtype=int)
        H, W = G.shape
        G[cell] = dotcol
        for p in dnbs(cell[0], cell[1], H, W):
            if int(G[p]) == pathcol:
                G[p] = ncol
        GO = G.copy()
        for (r, c) in obj:
            GO[r, c] = dotcol if (abs(r - cell[0]) + abs(c - cell[1])) % 2 == 0 else ncol
        if np.array_equal(G, GO):
            continue

        k = random.randint(0, 3)
        Gi = np.ascontiguousarray(np.rot90(G, k))
        Go = np.ascontiguousarray(np.rot90(GO, k))
        if Gi.shape[0] > max_h or Gi.shape[1] > max_w:
            continue
        chk = apply_rule(Gi)
        if chk is None or not np.array_equal(chk, Go):
            continue
        return {"input": Gi.tolist(), "output": Go.tolist()}


def derive_operations(I, O):
    """Rule (read from I only): the grid is a maze whose (even,even) lattice cells are
    the wall colour.  One corridor cell carries the dot colour; every corridor cell
    4-adjacent to it carries the neighbour colour.  The whole corridor component that
    contains the dot gets checkerboarded: cells at even manhattan distance from the dot
    become the dot colour, cells at odd distance become the neighbour colour."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    def dnbs(r, c):
        return [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= rr < H and 0 <= cc < W]

    def flood(start, allowed):
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in dnbs(r, c):
                if p not in seen and int(I[p]) in allowed:
                    seen.add(p)
                    stack.append(p)
        return seen

    def identify():
        wcol = int(I[0, 0])                      # (0,0) is on the wall lattice
        pal = sorted(set(int(v) for v in I.reshape(-1)))
        others = [c for c in pal if c != wcol]
        if len(others) != 3:
            return None
        found = []
        for dc in others:
            dpos = [(r, c) for r in range(H) for c in range(W) if int(I[r, c]) == dc]
            if len(dpos) != 1:
                continue
            d = dpos[0]
            nb = dnbs(d[0], d[1])
            for nc in others:
                if nc == dc:
                    continue
                pc = [x for x in others if x != dc and x != nc][0]
                npos = [(r, c) for r in range(H) for c in range(W) if int(I[r, c]) == nc]
                if not (1 <= len(npos) <= 4):
                    continue
                if any(p not in nb for p in npos):
                    continue
                # every corridor neighbour of the dot was recoloured, so the dot has
                # no remaining path-coloured neighbour -- this fixes the orientation
                if any(int(I[p]) == pc for p in nb):
                    continue
                region = flood(d, {pc, dc, nc})
                if len(region) <= 4:
                    continue
                found.append((dc, nc, d, region))
        if not found:
            return None
        return found[0]

    ops, sels = [], []
    res = identify()
    if res is not None:
        dc, nc, d, region = res
        key = lambda p: (abs(p[0] - d[0]) + abs(p[1] - d[1]), p[0], p[1])
        even = sorted([p for p in region
                       if (abs(p[0] - d[0]) + abs(p[1] - d[1])) % 2 == 0 and int(I[p]) != dc],
                      key=key)
        odd = sorted([p for p in region
                      if (abs(p[0] - d[0]) + abs(p[1] - d[1])) % 2 == 1 and int(I[p]) != nc],
                     key=key)
        if even:
            ops.append(int(dc))
            sels.append(sel_of(even))
        if odd:
            ops.append(int(nc))
            sels.append(sel_of(odd))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])   # bbox == whole grid, exactly the cells intended
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
