"""
ARC Task: 72322fa7 (RE-ARC) — LLM-generated grid_maker
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
    # The rule does not depend on colour, only on background. Key colours are sampled per instance.
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + round((b - a) * lb)
    hi = a + round((b - a) * ub)
    return random.randint(min(lo, hi), max(lo, hi))


def _nb8(r, c):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _landing(I, bgc):
    """From I only: for each key colour cb, the set of cells where cb must be stamped.
    A key is a diagonally connected 2-colour object. Wherever one of its single-colour
    parts occurs alone, the partner part lands at the matching offset."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for ny, nx in _nb8(y, x):
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True; q.append((ny, nx))
                comps.append(cells)
    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    landing, order = {}, []
    for cells in keys:
        kcols = sorted({int(I[y, x]) for y, x in cells})
        for ca in kcols:
            cb = [k for k in kcols if k != ca][0]
            part = [(y, x) for y, x in cells if int(I[y, x]) == ca]
            partner = [(y, x) for y, x in cells if int(I[y, x]) == cb]
            ar = min(y for y, _ in part); ac = min(x for _, x in part)
            prel = [(y - ar, x - ac) for y, x in part]
            brel = [(y - ar, x - ac) for y, x in partner]
            ph = max(y for y, _ in prel); pw = max(x for _, x in prel)
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == ca for y, x in prel):
                        for y, x in brel:
                            ty, tx = i + y, j + x
                            if 0 <= ty < h and 0 <= tx < w and I[ty, tx] != cb:
                                if cb not in landing:
                                    landing[cb] = set(); order.append(cb)
                                landing[cb].add((ty, tx))
    return landing, order


def _apply_rule(gi, bgc):
    g = np.array(gi, dtype=int)
    for cb, cells in _landing(g, bgc)[0].items():
        for r, c in cells:
            g[r, c] = cb
    return g.tolist()


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in range(10) if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, 4))
    ccols = random.sample(remcols, 2 * nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    objs = []
    for ca, cb in cpairs:
        oh = _unifint(diff_lb, diff_ub, (1, 4))
        ow = _unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(r, c) for r in range(oh) for c in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(2, oh * ow)
        for _ in range(ncells - 1):
            nbs = set()
            for (r, c) in obj:
                nbs |= set(_nb8(r, c))
            obj.add(random.choice(sorted((bounds - obj) & nbs)))
        mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
        objt = sorted((r - mr, c - mc) for r, c in obj)
        apart = set(random.sample(objt, random.randint(1, len(objt) - 1)))
        cobj = [(ca if p in apart else cb, p) for p in objt]
        cands = [(i, j) for (i, j) in inds if all((i + r, j + c) in inds for r, c in objt)]
        if not cands:
            continue
        i, j = random.choice(sorted(cands))
        placed = set()
        for col, (r, c) in cobj:
            gi[i + r][j + c] = col
            placed.add((i + r, j + c))
        ex = set(placed)
        for (r, c) in placed:
            ex |= set(_nb8(r, c))
        inds -= ex
        objs.append(cobj)
    if not objs:
        raise ValueError("no objs")
    avgs = sum(len(o) for o in objs) / len(objs)
    ub = max(1, int((h * w) // (avgs * 2)))
    noccs = _unifint(diff_lb, diff_ub, (1, ub))
    go = [row[:] for row in gi]
    succ = tr = 0
    while tr < 5 * noccs and succ < noccs:
        tr += 1
        cobj = random.choice(objs)
        cols = list({c for c, _ in cobj})
        oh = max(r for _, (r, c) in cobj) + 1
        ow = max(c for _, (r, c) in cobj) + 1
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        i, j = random.choice(sorted(cands))
        cells = [(i + r, j + c) for _, (r, c) in cobj]
        if all(p in inds for p in cells):
            succ += 1
            ex = set(cells)
            for (r, c) in cells:
                ex |= set(_nb8(r, c))
            inds -= ex
            col = random.choice(cols)
            for cc, (r, c) in cobj:
                go[i + r][j + c] = cc
                if cc == col:
                    gi[i + r][j + c] = cc
    # reject instances where a part shape also matches elsewhere (rule on I != O)
    if _apply_rule(gi, bgc) != go:
        raise ValueError("ambiguous instance: rule on input does not reproduce output")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (from I only): each 2-colour key object splits into two single-colour parts.
    Wherever one part occurs alone in I, the partner part is stamped at the matching offset.
    Unit of work: ONE COLOUR AT A TIME (not one occurrence at a time). For each key
    colour cb there is a single Color(cb) op. Its selection is every cell the cb part
    lands on, across ALL incomplete copies at once, minus cells that are already cb in I.
    O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            cnt[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = cnt.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand

    landing, order = _landing(I, bgc)

    ops, sels = [], []
    for cb in order:
        ops.append(int(cb)); sels.append(sel_of(sorted(landing[cb])))
    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
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
                                f"for task 72322fa7"
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
                    f"Failed to build a complete episode for task 72322fa7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72322fa7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
