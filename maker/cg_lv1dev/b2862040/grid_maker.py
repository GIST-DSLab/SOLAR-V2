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
import numpy as np
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _box(oh, ow):
    cells = set()
    for r in range(oh):
        cells.add((r, 0)); cells.add((r, ow - 1))
    for c in range(ow):
        cells.add((0, c)); cells.add((oh - 1, c))
    return cells


def _dn(r, c):
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _nb8(r, c):
    return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}


def _outbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
    out = set()
    for r in range(r0, r1 + 1):
        out.add((r, c0)); out.add((r, c1))
    for c in range(c0, c1 + 1):
        out.add((r0, c)); out.add((r1, c))
    return out


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    lo_i = int(round(lo)); hi_i = int(round(hi))
    lo_i = max(a, min(b, lo_i)); hi_i = max(lo_i, min(b, hi_i))
    return random.randint(lo_i, hi_i)


def _components(g):
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r][c]:
                continue
            col = g[r][c]
            q = deque([(r, c)]); seen[r][c] = True; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for ny, nx in _dn(y, x):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] == col:
                        seen[ny][nx] = True; q.append((ny, nx))
            comps.append((col, cells))
    return comps


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    while True:
        h = _unif(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unif(diff_lb, diff_ub, (10, max(10, max_w)))
        nobjs = _unif(diff_lb, diff_ub, (1, (h * w) // 16))
        succ, tr, maxtr = 0, 0, 10 * nobjs
        gi = [[bgc] * w for _ in range(h)]
        inds = {(r, c) for r in range(h) for c in range(w)}
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6); ow = random.randint(3, 6)
            obj = _box(oh, ow)
            if random.choice((True, False)):
                nkeep = _unif(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                corners = {(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)}
                obj.discard(random.choice(sorted(obj - corners)))
                for _ in range(nrem - 1):
                    xx = [ij for ij in obj if len(_dn(*ij) & obj) == 1]
                    if not xx:
                        break
                    obj.discard(random.choice(sorted(xx)))
            npert = _unif(diff_lb, diff_ub, (0, oh + ow))
            o1 = _outbox(obj); o2 = _outbox(o1); o3 = _outbox(o2)
            objcands = o1 | o2 | o3
            obj = set(obj)
            for _ in range(npert):
                nbrs = set()
                for ij in obj:
                    nbrs |= _dn(*ij)
                cand = (objcands - obj) & nbrs
                if not cand:
                    break
                obj.add(random.choice(sorted(cand)))
            mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
            obj = {(r - mr, c - mc) for r, c in obj}
            oh = max(r for r, _ in obj) + 1; ow = max(c for _, c in obj) + 1
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            lr, lc = random.choice(cands)
            plcd = {(r + lr, c + lc) for r, c in obj}
            if plcd.issubset(inds):
                col = random.choice(remcols)
                for r, c in plcd:
                    gi[r][c] = col
                succ += 1
                ring = set()
                for ij in plcd:
                    ring |= _nb8(*ij)
                inds = inds - plcd - ring
        comps = _components(gi)
        pockets = set()
        for col, cells in comps:
            if col == bgc and not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                pockets |= set(cells)
        if pockets:
            go = [row[:] for row in gi]
            for col, cells in comps:
                if col == bgc:
                    continue
                if any(_dn(r, c) & pockets for r, c in cells):
                    for r, c in cells:
                        go[r][c] = 8
            return {"input": gi, "output": go}


def derive_operations(I, O):
    # Rule from I alone: bgc = dominant colour; enclosed bg pockets = bg components
    # not touching the border; every same-colour 4-connected object adjacent to a
    # pocket is recoloured to 8 via one FloodFill8 seeded on that object.
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
            col = I[r, c]; cid = len(comps)
            q = deque([(r, c)]); comp_id[r, c] = cid; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < hi and 0 <= nx < wi and comp_id[ny, nx] == -1 and I[ny, nx] == col:
                        comp_id[ny, nx] = cid; q.append((ny, nx))
            comps.append((col, sorted(cells)))

    ops, sels = [], []
    done = set()
    for col, pcells in comps:
        if col != bgc:
            continue
        if any(r == 0 or c == 0 or r == hi - 1 or c == wi - 1 for r, c in pcells):
            continue
        targets = []
        for r, c in pcells:
            for ny, nx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] != bgc:
                    tid = int(comp_id[ny, nx])
                    if tid not in done and tid not in targets:
                        targets.append(tid)
        for tid in targets:
            done.add(tid)
            if comps[tid][0] == 8:
                continue
            ops.append(18)
            sels.append(sel_of([comps[tid][1][0]]))

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
