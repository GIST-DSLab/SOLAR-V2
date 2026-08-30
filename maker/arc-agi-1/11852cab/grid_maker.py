"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}

# the four D4-orbits of a 5x5 block (all cells with even (i+j))
_RINGS = (
    ((2, 2),),
    ((1, 1), (1, 3), (3, 1), (3, 3)),
    ((0, 2), (2, 0), (2, 4), (4, 2)),
    ((0, 0), (0, 4), (4, 0), (4, 4)),
)
_RING_OF = {}
for _k, _rg in enumerate(_RINGS):
    for _p in _rg:
        _RING_OF[_p] = _k


# ------------------------------------------------------- block detection (I only)
def _is_block(g, r0, c0, bgc):
    """The verifier's window test, measured on a grid alone."""
    h, w = len(g), len(g[0])
    if r0 < 1 or c0 < 1 or r0 + 5 > h - 1 or c0 + 5 > w - 1:
        return False
    # the 7x7 frame around the 5x5 must be pure background
    for j in range(c0 - 1, c0 + 6):
        if g[r0 - 1][j] != bgc or g[r0 + 5][j] != bgc:
            return False
    for i in range(r0 - 1, r0 + 6):
        if g[i][c0 - 1] != bgc or g[i][c0 + 5] != bgc:
            return False
    cells = {}
    for i in range(5):
        for j in range(5):
            v = g[r0 + i][c0 + j]
            if v == bgc:
                continue
            if (i + j) % 2 == 1:          # odd cells must be background
                return False
            cells[(i, j)] = v
    if (2, 2) not in cells:               # centre must be present
        return False
    rs = [i for i, _ in cells]
    cs = [j for _, j in cells]
    if min(rs) != 0 or max(rs) != 4 or min(cs) != 0 or max(cs) != 4:
        return False                      # content must span the whole 5x5
    ringcol = {}                          # every ring is monochrome
    nring = {}
    for (i, j), v in cells.items():
        k = _RING_OF[(i, j)]
        if ringcol.setdefault(k, v) != v:
            return False
        nring[k] = nring.get(k, 0) + 1
    # at least one ring around the centre is already whole
    if not any(nring.get(k, 0) == 4 for k in (1, 2, 3)):
        return False
    return True


def _block_score(g, r0, c0, bgc):
    """How strongly a window looks like a placed block (tie-break only)."""
    inner = sum(1 for (i, j) in ((1, 1), (1, 3), (3, 1), (3, 3))
                if g[r0 + i][c0 + j] != bgc)
    nfg = sum(1 for i in range(5) for j in range(5)
              if g[r0 + i][c0 + j] != bgc)
    return (10 * (1 if inner else 0)) + nfg


def _find_blocks(g, bgc):
    """All 5x5 blocks; placed blocks never overlap, so overlapping candidates
    are resolved by keeping the largest mutually disjoint family."""
    h, w = len(g), len(g[0])
    cands = [(r0, c0) for r0 in range(1, max(1, h - 5))
             for c0 in range(1, max(1, w - 5)) if _is_block(g, r0, c0, bgc)]
    n = len(cands)
    adj = [set() for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if abs(cands[a][0] - cands[b][0]) < 5 and \
               abs(cands[a][1] - cands[b][1]) < 5:
                adj[a].add(b)
                adj[b].add(a)
    score = [_block_score(g, r0, c0, bgc) for (r0, c0) in cands]
    free = [a for a in range(n) if not adj[a]]
    rest = [a for a in range(n) if adj[a]]
    seen = {}

    def best(nodes):
        if not nodes:
            return (0, 0, ())
        if nodes in seen:
            return seen[nodes]
        v = max(nodes, key=lambda a: (len(adj[a] & set(nodes)), score[a]))
        without = best(tuple(a for a in nodes if a != v))
        c, s, ch = best(tuple(a for a in nodes if a != v and a not in adj[v]))
        with_v = (c + 1, s + score[v], ch + (v,))
        res = max(with_v, without, key=lambda t: (t[0], t[1]))
        seen[nodes] = res
        return res

    if len(rest) <= 22:
        chosen = list(best(tuple(rest))[2])
    else:                                   # greedy fallback
        chosen, taken = [], set()
        for a in sorted(rest, key=lambda a: -score[a]):
            if a not in taken:
                chosen.append(a)
                taken |= adj[a] | {a}
    return sorted(cands[a] for a in free + chosen)


def _closure(g, bgc):
    """Complete every block: each ring is filled out to its whole orbit."""
    out = [list(row) for row in g]
    for (r0, c0) in _find_blocks(g, bgc):
        for ring in _RINGS:
            col = None
            for (i, j) in ring:
                if g[r0 + i][c0 + j] != bgc:
                    col = g[r0 + i][c0 + j]
            if col is None:
                continue
            for (i, j) in ring:
                out[r0 + i][c0 + j] = col
    return out


# ------------------------------------------------------------------ sample_colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(rem, numc)
    return {"bgc": bgc, "ccols": ccols}


# ----------------------------------------------------------------------- generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    def unifint(bounds):
        a, b = bounds
        lo = max(a, int(a + (b - a) * diff_lb))
        hi = min(b, int(a + (b - a) * diff_ub))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    mh = max(7, min(30, int(max_h)))
    mw = max(7, min(30, int(max_w)))

    def build_block():
        for _ in range(40):
            ringcols = [random.choice(ccols) for _ in range(4)]
            idx = random.randint(1, 3)
            cells = {}
            for p in _RINGS[0]:
                cells[p] = ringcols[0]
            for p in _RINGS[idx]:
                cells[p] = ringcols[idx]
            rest = [k for k in (1, 2, 3) if k != idx]
            chosen = random.sample(rest, unifint((1, 2)))
            keeps = [4 - unifint((0, 3)) for _ in chosen]
            if all(k == 4 for k in keeps):
                # a ring must be left incomplete, else there is nothing to do
                keeps[random.randrange(len(keeps))] = 3
            for k, keep in zip(chosen, keeps):
                for p in random.sample(list(_RINGS[k]), keep):
                    cells[p] = ringcols[k]
            rs = [i for i, _ in cells]
            cs = [j for _, j in cells]
            if min(rs) != 0 or max(rs) != 4 or min(cs) != 0 or max(cs) != 4:
                continue
            if not any(p in cells for p in _RINGS[1]):
                continue
            return cells
        return None

    gi = [[bgc] * mw for _ in range(mh)]
    for _attempt in range(50):
        h = unifint((7, mh))
        w = unifint((7, mw))
        gi = [[bgc] * w for _ in range(h)]
        avail = set((i, j) for i in range(1, h - 1) for j in range(1, w - 1))
        nobjs = unifint((1, max(1, (h * w) // 36)))
        succ, tr, maxtr = 0, 0, 10 * nobjs + 10
        while succ < nobjs and tr < maxtr:
            tr += 1
            cands = [(i, j) for (i, j) in avail if i <= h - 6 and j <= w - 6]
            if not cands:
                break
            r0, c0 = random.choice(sorted(cands))
            box = set((r0 + i, c0 + j) for i in range(5) for j in range(5))
            if not box <= avail:
                continue
            cells = build_block()
            if cells is None:
                continue
            for (i, j), col in cells.items():
                gi[r0 + i][c0 + j] = col
            avail -= set((r0 + i, c0 + j)
                         for i in range(-1, 6) for j in range(-1, 6))
            succ += 1
        if succ == 0:
            continue
        go = _closure(gi, bgc)
        if go == gi:
            continue
        return {'input': tuple(tuple(r) for r in gi),
                'output': tuple(tuple(r) for r in go)}
    return {'input': tuple(tuple(r) for r in gi),
            'output': tuple(tuple(r) for r in _closure(gi, bgc))}


# --------------------------------------------------------------- derive_operations
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = {}
    for v in I.flatten().tolist():
        cnt[v] = cnt.get(v, 0) + 1
    bgc = max(sorted(cnt), key=lambda c: cnt[c])

    grid = [list(map(int, row)) for row in I.tolist()]
    ops, sels = [], []

    # every block is located in I alone; nothing below looks at O
    for (r0, c0) in _find_blocks(grid, bgc):
        rect = [r0, c0, 4, 4]          # exactly the whole 5x5 block, bg included
        # Reflect the block across its vertical axis, then its horizontal axis,
        # then its main diagonal (rotate CCW + flip up/down). After each
        # reflection the marks the block carried just before it are drawn back
        # on top, so the block accumulates every mirror image of itself and its
        # partial rings close up.
        for opgroup in ((26,), (27,), (24, 27)):
            before = [row[c0:c0 + 5] for row in grid[r0:r0 + 5]]
            for op in opgroup:
                sub = [row[c0:c0 + 5] for row in grid[r0:r0 + 5]]
                if op == 26:                                  # flip left/right
                    new = [list(reversed(r)) for r in sub]
                elif op == 27:                                # flip up/down
                    new = [list(r) for r in reversed(sub)]
                else:                                         # 24 = rotate CCW
                    new = [[sub[j][4 - i] for j in range(5)] for i in range(5)]
                if new == sub:
                    continue                       # nothing would change: skip
                ops.append(op)
                sels.append(rect)
                for i in range(5):
                    for j in range(5):
                        grid[r0 + i][c0 + j] = new[i][j]
            bycol = {}
            for i in range(5):
                for j in range(5):
                    v = before[i][j]
                    if v != bgc and grid[r0 + i][c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(int(v))
                sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    grid[r][c] = v

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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
