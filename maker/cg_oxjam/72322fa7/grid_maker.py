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


def _unifint(lb, ub, bounds):
    a, b = bounds
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(max(a, lo), min(b, hi))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    random.shuffle(rem)
    return {"bgc": bgc, "ccols": rem[:8]}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    if bgc is None:
        bgc = random.randint(0, 9)
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
        random.shuffle(ccols)
    h = _unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    nobjs = _unifint(diff_lb, diff_ub, (1, 4))
    cc = list(ccols[:2 * nobjs])
    cpairs = list(zip(cc[:nobjs], cc[nobjs:]))

    def nbrs(cell):
        r, c = cell
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)}

    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    objs = []
    for ca, cb in cpairs:
        oh = _unifint(diff_lb, diff_ub, (1, 4))
        ow = _unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(2, oh * ow)
        for _ in range(ncells - 1):
            nb = set()
            for cell in obj:
                nb |= nbrs(cell)
            obj.add(random.choice(sorted((bounds - obj) & nb)))
        mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
        objn = sorted((r - mr, c - mc) for r, c in obj)
        apart = random.sample(objn, random.randint(1, len(objn) - 1))
        aset = set(apart)
        o = [(ca if cell in aset else cb, cell) for cell in objn]
        cands = [(i, j) for (i, j) in inds
                 if all((i + r, j + c) in inds for r, c in objn)]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        plcdi = set()
        for col, (r, c) in o:
            gi[li + r][lj + c] = col
            plcdi.add((li + r, lj + c))
        nb = set()
        for cell in plcdi:
            nb |= nbrs(cell)
        inds = (inds - plcdi) - nb
        objs.append(o)
    if not objs:
        raise ValueError("no key placed")
    avgs = sum(len(o) for o in objs) / len(objs)
    ub = max(1, int((h * w) // (avgs * 2)))
    noccs = _unifint(diff_lb, diff_ub, (1, ub))
    go = [row[:] for row in gi]
    succ = tr = 0
    while tr < 5 * noccs and succ < noccs:
        tr += 1
        o = random.choice(objs)
        pal = sorted({col for col, _ in o})
        oh = max(r for _, (r, _) in o) + 1
        ow = max(c for _, (_, c) in o) + 1
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        plcdi = {(li + r, lj + c) for _, (r, c) in o}
        if plcdi <= inds:
            succ += 1
            nb = set()
            for cell in plcdi:
                nb |= nbrs(cell)
            inds = (inds - plcdi) - nb
            keep = random.choice(pal)
            for col, (r, c) in o:
                go[li + r][lj + c] = col
                if col == keep:
                    gi[li + r][lj + c] = col
    # reject the rare layouts where a lone half matches spuriously (rule ambiguous)
    ops_, sels_ = derive_operations(gi, go)
    chk = np.array(gi, dtype=int)
    for o_, s_ in zip(ops_, sels_):
        if o_ == 34:
            break
        for r_, c_ in s_["cells"]:
            chk[r_, c_] = o_
    if not (chk == np.array(go, dtype=int)).all():
        raise ValueError("ambiguous instance")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading: each two-coloured object is a KEY; every lone single-colour piece in I
    that matches one half of a key is an incomplete copy. For each incomplete copy,
    stamp the missing half (one Color op of the partner colour on the partner's cells,
    placed at the key's relative offset). This agrees with the maker / golf reading,
    so the operations are unchanged; only bgc is taken from the demonstrations when
    available. O is never inspected to decide what or where to paint.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            cnt.update(ei.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # connected components of non-background cells (8-connectivity)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True; q.append((ny, nx))
                comps.append(cells)

    # keys = components made of exactly two colours
    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    ops, sels = [], []
    grid = I.copy()
    for cells in keys:
        kcols = sorted({int(I[y, x]) for y, x in cells})
        for ca in kcols:
            part = [(y, x) for y, x in cells if int(I[y, x]) == ca]
            partner = [(y, x) for y, x in cells if int(I[y, x]) != ca]
            cb = int(I[partner[0]])
            ar = min(y for y, _ in part); ac = min(x for _, x in part)
            prel = [(y - ar, x - ac) for y, x in part]
            brel = [(y - ar, x - ac) for y, x in partner]
            ph = max(y for y, _ in prel); pw = max(x for _, x in prel)
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == ca for y, x in prel):
                        tgt = [(i + y, j + x) for y, x in brel
                               if 0 <= i + y < h and 0 <= j + x < w]
                        need = [(y, x) for y, x in tgt if grid[y, x] != cb]
                        if need:
                            # one op per incomplete copy: stamp its missing half
                            ops.append(cb); sels.append(sel_of(need))
                            for y, x in need:
                                grid[y, x] = cb
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
