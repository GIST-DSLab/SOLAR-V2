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


def sample_colors(num_examples=None) -> dict:
    pathcol, wallcol, dotcol, ncol = random.sample(range(10), 4)
    return {"pathcol": pathcol, "wallcol": wallcol, "dotcol": dotcol, "ncol": ncol}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = int(a + diff_lb * (b - a))
    hi = int(a + diff_ub * (b - a))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _components(g, col):
    h, w = len(g), len(g[0])
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r][c] != col or (r, c) in seen:
                continue
            comp = []
            dq = deque([(r, c)])
            seen.add((r, c))
            while dq:
                a, b = dq.popleft()
                comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen and g[na][nb] == col:
                        seen.add((na, nb))
                        dq.append((na, nb))
            comps.append(comp)
    return comps


def generate(diff_lb, diff_ub, max_h=30, max_w=30, pathcol=0, wallcol=1, dotcol=2, ncol=3, **kw) -> dict:
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    cap = max(3, min(15, (min(max_h, max_w) + 1) // 2))
    while True:
        h = _unifint(diff_lb, diff_ub, (3, cap))
        w = _unifint(diff_lb, diff_ub, (3, cap))
        maze = [[{'x': x, 'y': y, 'walls': {'N': True, 'S': True, 'E': True, 'W': True}}
                 for y in range(h)] for x in range(w)]
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
        gi = [list(r[1:-1]) for r in grid[1:-1]]
        objs = [o for o in _components(gi, pathcol) if len(o) > 4]
        if not objs:
            continue
        objs.sort(key=len)
        idx = _unifint(diff_lb, diff_ub, (0, len(objs) - 1))
        obj = objs[idx]
        cell = random.choice(obj)
        gi[cell[0]][cell[1]] = dotcol
        H, W = len(gi), len(gi[0])
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < H and 0 <= nc < W and gi[nr][nc] == pathcol:
                gi[nr][nc] = ncol
        go = [list(r) for r in gi]
        for (r, c) in obj:
            go[r][c] = dotcol if (abs(r - cell[0]) + abs(c - cell[1])) % 2 == 0 else ncol
        break
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(np.array(gi), k)
    go = np.rot90(np.array(go), k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule: a maze of corridors (one colour) and walls; one corridor carries a
    marker -- a seed cell and the differently coloured cell(s) touching it.
    The two marker colours alternate outward along that whole corridor.
    Walk the corridor breadth-first from the seed, painting each still
    unmarked corridor cell with the seed colour at even steps and the
    neighbour colour at odd steps.  Everything is read from I (and the
    corridor-colour convention from the demonstrations); O is never read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    cnt = Counter(I.flatten().tolist())
    cols = sorted(cnt, key=lambda c: (cnt[c], c))
    seed_col = cols[0]
    other_col = sorted([c for c in cnt if c != seed_col], key=lambda c: (cnt[c], c))[0]
    marker = {seed_col, other_col}
    mcells = [(r, c) for r in range(h) for c in range(w) if int(I[r, c]) in marker]

    # corridor colour: the colour the demonstrations repaint
    path_col = None
    if examples:
        votes = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            m = ei != eo
            votes.update(ei[m].tolist())
        for c, _ in votes.most_common():
            if c in cnt and c not in marker:
                path_col = int(c)
                break
    if path_col is None:
        # fallback: rarest colour among the cells surrounding the marker
        mset = set(mcells)
        ring = Counter()
        for r, c in mcells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mset:
                        ring[int(I[nr, nc])] += 1
        path_col = sorted(ring, key=lambda c: (ring[c], c))[0]

    seed = next((r, c) for r, c in mcells if int(I[r, c]) == seed_col)
    ok = marker | {path_col}
    dist = {seed: 0}
    walk = [seed]
    dq = deque([seed])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and int(I[nr, nc]) in ok:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
                walk.append((nr, nc))

    # one object (the marked corridor), painted outward from the seed
    for (r, c) in walk:
        if int(I[r, c]) != path_col:
            continue
        ops.append(int(seed_col if dist[(r, c)] % 2 == 0 else other_col))
        sels.append(sel_of([(r, c)]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
