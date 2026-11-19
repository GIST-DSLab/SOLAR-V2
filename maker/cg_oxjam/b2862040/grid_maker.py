"""
ARC Task: b2862040 (RE-ARC) — LLM-generated grid_maker
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


# Comparison with the code-golf reading:
# The golf entry sweeps in from the grid border (rotating the grid 4 times per pass)
# to mark every background cell it can reach from the edge. Background it never
# reaches is enclosed, and any shape touching that enclosed background turns 8.
# That is the same split the maker uses. "Border-unreachable background" is exactly
# the set of background components that do not touch the border, so both readings
# give the same pockets. Each closed shape is still one unit, recoloured whole.
# The golf version marks the outside with a temporary colour. In ARCLE that marked
# outside region would merge with the new 8-shapes during the restore flood, and
# it adds nothing visible. Since the readings agree, the maker's route is kept:
# one FloodFill8 per closed shape, taken pocket by pocket.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def _box(oh, ow):
    return {(i, j) for i in range(oh) for j in range(ow)
            if i == 0 or j == 0 or i == oh - 1 or j == ow - 1}


def _outbox(patch):
    rs = [i for i, _ in patch]
    cs = [j for _, j in patch]
    r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
    return {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)
            if i == r0 or i == r1 or j == c0 or j == c1}


def _dn(i, j):
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def _n8(i, j):
    return [(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def _components(g):
    h, w = len(g), len(g[0])
    cid = [[-1] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if cid[r][c] != -1:
                continue
            col = g[r][c]
            k = len(comps)
            cells = []
            q = deque([(r, c)])
            cid[r][c] = k
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in _dn(y, x):
                    if 0 <= ny < h and 0 <= nx < w and cid[ny][nx] == -1 and g[ny][nx] == col:
                        cid[ny][nx] = k
                        q.append((ny, nx))
            comps.append((col, cells))
    return comps, cid


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    while True:
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        nobjs = _unifint(diff_lb, diff_ub, (1, (h * w) // 16))
        succ, tr, maxtr = 0, 0, 10 * nobjs
        gi = [[bgc] * w for _ in range(h)]
        inds = {(i, j) for i in range(h) for j in range(w)}
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6)
            ow = random.randint(3, 6)
            obj = _box(oh, ow)
            if random.choice((True, False)):
                nkeep = _unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                corners = {(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)}
                obj.discard(random.choice(sorted(obj - corners)))
                for _ in range(nrem - 1):
                    xx = [p for p in obj if len(set(_dn(*p)) & obj) == 1]
                    if not xx:
                        break
                    obj.discard(random.choice(sorted(xx)))
            npert = _unifint(diff_lb, diff_ub, (0, oh + ow))
            ob1 = _outbox(obj)
            ob2 = _outbox(ob1)
            ob3 = _outbox(ob2)
            objcands = ob1 | ob2 | ob3
            obj = set(obj)
            for _ in range(npert):
                nb = set()
                for p in obj:
                    nb |= set(_dn(*p))
                opts = sorted((objcands - obj) & nb & objcands)
                if not opts:
                    break
                obj.add(random.choice(opts))
            mr = min(i for i, _ in obj)
            mc = min(j for _, j in obj)
            obj = {(i - mr, j - mc) for i, j in obj}
            oh = max(i for i, _ in obj) + 1
            ow = max(j for _, j in obj) + 1
            cands = [p for p in inds if p[0] <= h - oh and p[1] <= w - ow]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            plcd = {(i + loc[0], j + loc[1]) for i, j in obj}
            if plcd.issubset(inds):
                col = random.choice(remcols)
                for i, j in plcd:
                    gi[i][j] = col
                succ += 1
                rem = set(plcd)
                for p in plcd:
                    rem |= set(_n8(*p))
                inds = inds - rem
        comps, cid = _components(gi)
        pockets = [cells for col, cells in comps if col == bgc and
                   not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)]
        if not pockets:
            continue
        pset = set(p for cells in pockets for p in cells)
        go = [row[:] for row in gi]
        for col, cells in comps:
            if col == bgc:
                continue
            if any(q in pset for p in cells for q in _dn(*p)):
                for i, j in cells:
                    go[i][j] = 8
        return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule, read from I alone:
      - bgc is the dominant colour of I.
      - Pockets are the 4-connected bgc components that do not touch the grid border,
        i.e. the background a closed shape encloses.
      - Every 4-connected non-bgc shape orthogonally adjacent to a pocket is a
        closed shape, and the whole shape is recoloured to 8 with one FloodFill8.
    O is used only to size the final Submit selection.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    comp_id = -np.ones((hi, wi), dtype=int)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if comp_id[r, c] != -1:
                continue
            col = I[r, c]
            cid = len(comps)
            cells = []
            q = deque([(r, c)])
            comp_id[r, c] = cid
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < hi and 0 <= nx < wi and comp_id[ny, nx] == -1 \
                            and I[ny, nx] == col:
                        comp_id[ny, nx] = cid
                        q.append((ny, nx))
            comps.append((col, sorted(cells)))

    pockets = []
    for cid, (col, cells) in enumerate(comps):
        if col != bgc:
            continue
        if any(r == 0 or c == 0 or r == hi - 1 or c == wi - 1 for r, c in cells):
            continue
        pockets.append(cells)

    ops, sels = [], []
    done = set()
    for pcells in pockets:
        targets = []
        for (r, c) in pcells:
            for ny, nx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] != bgc:
                    tid = int(comp_id[ny, nx])
                    if tid not in done and tid not in targets:
                        targets.append(tid)
        for tid in targets:
            done.add(tid)
            seed_r, seed_c = comps[tid][1][0]
            ops.append(18)                              # FloodFill8 on the closed shape
            sels.append(sel_of([(seed_r, seed_c)]))     # exactly one seed cell

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b2862040"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b2862040"
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
                                f"for task b2862040"
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
                    f"Failed to build a complete episode for task b2862040 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b2862040-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
