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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return int(round(a + (b - a) * d)) if b > a else a


def sample_colors(num_examples=None) -> dict:
    pathcol, wallcol, dotcol, ncol = random.sample(range(10), 4)
    return {"pathcol": pathcol, "wallcol": wallcol, "dotcol": dotcol, "ncol": ncol}


def _components(g, col):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == col and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; comp = []
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and not seen[x, y] and g[x, y] == col:
                            seen[x, y] = True; q.append((x, y))
                comps.append(comp)
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, pathcol=None, wallcol=None, dotcol=None, ncol=None) -> dict:
    if None in (pathcol, wallcol, dotcol, ncol):
        pathcol, wallcol, dotcol, ncol = random.sample(range(10), 4)
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    walls = {'N': True, 'S': True, 'E': True, 'W': True}
    ub = max(3, min(15, (min(max_h, max_w) + 1) // 2))
    while True:
        h = _unifint(diff_lb, diff_ub, (3, ub))
        w = _unifint(diff_lb, diff_ub, (3, ub))
        maze = [[{'x': x, 'y': y, 'walls': {**walls}} for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    nb = maze[x2][y2]
                    if all(nb['walls'].values()):
                        nbhs.append((direc, nb))
            if not nbhs:
                cc = stck.pop()
                continue
            direc, nxt = random.choice(nbhs)
            cc['walls'][direc] = False
            nxt['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = nxt
            nv += 1
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
        gi = np.array([r[1:-1] for r in grid[1:-1]], dtype=int)
        objs = [o for o in _components(gi, pathcol) if len(o) > 4]
        if not objs:
            continue
        objs.sort(key=len)
        idx = _unifint(diff_lb, diff_ub, (0, len(objs) - 1))
        obj = objs[idx]
        cell = random.choice(obj)
        go = gi.copy()
        gi[cell] = dotcol
        H, W = gi.shape
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = cell[0] + da, cell[1] + db
            if 0 <= x < H and 0 <= y < W and gi[x, y] == pathcol:
                gi[x, y] = ncol
        for (a, b) in obj:
            go[a, b] = dotcol if (abs(a - cell[0]) + abs(b - cell[1])) % 2 == 0 else ncol
        break
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _roles(I, examples):
    # The (0,0) cell-centre lattice colour is never filled (generator structure).
    wallcol = int(I[0, 0])
    pathcol = None
    for ex in (examples or []):
        Ie = np.asarray(ex[0], dtype=int); Oe = np.asarray(ex[1], dtype=int)
        if Ie.shape != Oe.shape:
            continue
        diff = Ie != Oe
        if diff.any():
            cand = Counter(Ie[diff].tolist()).most_common(1)[0][0]
            if cand in set(I.flatten().tolist()) and cand != wallcol:
                pathcol = int(cand)
                break
    pal = Counter(I.flatten().tolist())
    others = [c for c in pal if c != wallcol]
    if pathcol is None:
        pathcol = max(others, key=lambda c: pal[c])
    markers = [c for c in others if c != pathcol]
    return wallcol, pathcol, markers


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    wallcol, pathcol, markers = _roles(I, examples)
    cnt = Counter(I.flatten().tolist())
    dotcol = min(markers, key=lambda c: cnt[c])
    ncol = [c for c in markers if c != dotcol][0] if len(markers) > 1 else dotcol
    dot = tuple(np.argwhere(I == dotcol)[0])
    starts = [tuple(p) for p in np.argwhere(np.isin(I, markers))]
    # path component reachable from the marked cells through pathcol cells
    seen = set(starts); q = deque(starts); region = []
    while q:
        a, b = q.popleft()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = a + da, b + db
            if 0 <= x < H and 0 <= y < W and (x, y) not in seen and I[x, y] == pathcol:
                seen.add((x, y)); q.append((x, y)); region.append((x, y))
    par = (dot[0] + dot[1]) % 2
    even = [p for p in region if (p[0] + p[1]) % 2 == par]
    odd = [p for p in region if (p[0] + p[1]) % 2 != par]
    ops, sels = [], []
    if even:
        ops.append(int(dotcol)); sels.append(sel_of(even))
    if odd:
        ops.append(int(ncol)); sels.append(sel_of(odd))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
