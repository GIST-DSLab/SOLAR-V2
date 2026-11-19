"""
ARC Task: 63613498 (RE-ARC) — LLM-generated grid_maker
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


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + int(round(diff_lb * (hi - lo)))
    b = lo + int(round(diff_ub * (hi - lo)))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if b < a:
        a, b = b, a
    return random.randint(a, b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sepc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, sepc)]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "sepc": sepc, "ccols": ccols}


def _grow(objh, objw, ncells):
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    obj = {random.choice(sorted(bounds))}
    for _ in range(ncells - 1):
        nb = set()
        for (i, j) in obj:
            nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
        cand = sorted((bounds - obj) & nb)
        if not cand:
            break
        obj.add(random.choice(cand))
    return obj


def _norm(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return {(p[0] - r0, p[1] - c0) for p in cells}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sepc=None, ccols=None) -> dict:
    if bgc is None or sepc is None or ccols is None:
        d = sample_colors()
        bgc, sepc, ccols = d["bgc"], d["sepc"], d["ccols"]
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unif(diff_lb, diff_ub, 10, max_h)
    w = _unif(diff_lb, diff_ub, 10, max_w)
    objh = _unif(diff_lb, diff_ub, 1, h // 3)
    objw = _unif(diff_lb, diff_ub, 1, w // 3)
    ncells = _unif(diff_lb, diff_ub, 1, objh * objw)
    obj = _grow(objh, objw, ncells)
    gi = np.full((h, w), bgc, dtype=int)
    objc = random.choice(ccols)
    for (i, j) in obj:
        gi[i, j] = objc
    sep = {(objh + 1, j) for j in range(objw + 2)} | {(i, objw + 1) for i in range(objh + 2)}
    for (i, j) in sep:
        gi[i, j] = sepc
    inds = {(i, j) for i in range(h) for j in range(w)}
    inds -= {(i, j) for i in range(objh + 2) for j in range(objw + 2)}
    nobjs = _unif(diff_lb, diff_ub, 1, (h * w) // 20)
    succ, tr, maxtr = 0, 0, 5 * nobjs
    baseobj = _norm(obj)
    obj = _norm(obj)
    go = gi.copy()
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        if tr > 2000:
            raise RuntimeError("placement failed")
        oh = max(p[0] for p in obj) + 1
        ow = max(p[1] for p in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        plcd = {(i + loc[0], j + loc[1]) for (i, j) in obj}
        if plcd <= inds:
            col = random.choice(ccols)
            for (i, j) in plcd:
                gi[i, j] = col
                go[i, j] = sepc if succ == 0 else col
            succ += 1
            nb = set()
            for (i, j) in plcd:
                nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
            inds = (inds - plcd) - nb
        oh2 = random.randint(1, h // 3)
        ow2 = random.randint(2 if oh2 == 1 else 1, w // 3)
        if random.choice((True, False)):
            oh2, ow2 = ow2, oh2
        nc = _unif(diff_lb, diff_ub, 1, oh2 * ow2)
        obj = _norm(_grow(oh2, ow2, nc))
        if obj == baseobj:
            bh = max(p[0] for p in obj) + 1
            bw = max(p[1] for p in obj) + 1
            if len(obj) < oh2 * ow2:
                bounds = {(i, j) for i in range(oh2) for j in range(ow2)}
                nb = set()
                for (i, j) in obj:
                    nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
                obj.add(random.choice(sorted((bounds - obj) & nb)))
            else:
                corners = [(0, 0), (0, bw - 1), (bh - 1, 0), (bh - 1, bw - 1)]
                obj.discard(random.choice(corners))
            obj = _norm(obj)
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): the L-shaped separator
    # touching the border fences off a template shape in its corner; the one
    # object elsewhere with that exact shape takes on the separator's colour.
    # Both carve the task the same way (one twin object, one recolour), so the
    # trajectory is a single FloodFill of the twin object with the separator colour.
    A = np.asarray(I, dtype=int)
    h, w = A.shape

    # background: generator paints the canvas with bgc; confirm with examples
    cnt = Counter(A.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    if bgc not in set(A.flatten().tolist()):
        bgc = Counter(A.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if A[r, c] == bgc or seen[r, c]:
                continue
            col = int(A[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and A[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append((col, frozenset(cells)))

    def norm(cells):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        return frozenset((p[0] - r0, p[1] - c0) for p in cells)

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs), max(cs)

    cands = []
    for col, cells in comps:
        r0, c0, r1, c1 = bbox(cells)
        if len(cells) != (r1 - r0 + 1) + (c1 - c0 + 1) - 1:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        delta = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if (r, c) not in cells]
        if not delta or len({int(A[r, c]) for r, c in delta}) != 2:
            continue
        cands.append((col, cells, delta))
    cands.sort(key=lambda t: -len(t[1]))

    ops, sels = [], []
    for sepc, sepcells, delta in cands:
        tmpl = frozenset((r, c) for r, c in delta if A[r, c] != bgc)
        if not tmpl:
            continue
        key = norm(tmpl)
        twins = [cells for col, cells in comps if cells != tmpl and norm(cells) == key]
        if len(twins) != 1:
            continue
        seed = sorted(twins[0])[0]
        # recolour the whole twin object (one connected same-colour region)
        ops.append(10 + int(sepc))
        sels.append(sel_of([seed]))
        break

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
                        f"num_examples+1 ({num_examples + 1}) for task 63613498"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 63613498"
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
                                f"for task 63613498"
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
                    f"Failed to build a complete episode for task 63613498 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"63613498-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
