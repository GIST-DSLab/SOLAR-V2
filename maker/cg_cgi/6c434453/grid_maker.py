"""
ARC Task: 6c434453 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
import numpy as np


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def _box(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    out = set()
    for r in range(r0, r1 + 1):
        out.add((r, c0)); out.add((r, c1))
    for c in range(c0, c1 + 1):
        out.add((r0, c)); out.add((r1, c))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kw) -> dict:
    def unifint(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    if bgc is None:
        bgc = random.choice([c for c in range(10) if c != 2])
    if not ccols:
        ccols = random.sample([c for c in range(10) if c not in (2, bgc)], 3)
    h = unifint(10, min(30, max_h))
    w = unifint(10, min(30, max_w))
    nobjs = unifint(2, (h * w) // 16)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        if random.choice((True, False)):
            oh = random.choice((3, 5)); ow = random.choice((3, 5))
            obji = _box({(0, 0), (oh - 1, ow - 1)})
        else:
            oh = random.randint(1, 5); ow = random.randint(1, 5)
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            ncells = random.randint(1, oh * ow)
            obji = {random.choice(sorted(bounds))}
            for _ in range(ncells - 1):
                nb = set()
                for (a, b) in obji:
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb.add((a + da, b + db))
                cand = sorted((bounds - obji) & nb)
                obji.add(random.choice(cand))
            mr = min(a for a, _ in obji); mc = min(b for _, b in obji)
            obji = {(a - mr, b - mc) for a, b in obji}
        oh = max(a for a, _ in obji) + 1
        ow = max(b for _, b in obji) + 1
        isbox = obji == _box(obji)
        flag = isbox and {oh, ow} <= {3, 5}
        if isbox and min(oh, ow) >= 3 and not flag:
            continue  # ambiguous hollow box of unsupported size; skip
        if flag:
            objo = {(a, ow // 2) for a in range(oh)} | {(oh // 2, b) for b in range(ow)}
            tocover = {(a, b) for a in range(oh) for b in range(ow)}
        else:
            objo = obji
            tocover = obji
        loc = (random.randint(0, h - oh), random.randint(0, w - ow))
        plcdi = {(a + loc[0], b + loc[1]) for a, b in obji}
        if plcdi <= inds:
            succ += 1
            tc = {(a + loc[0], b + loc[1]) for a, b in tocover}
            nb = set(tc)
            for (a, b) in tc:
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb.add((a + da, b + db))
            inds -= nb
            col = random.choice(ccols)
            for a, b in plcdi:
                gi[a][b] = col
            for a, b in objo:
                go[a + loc[0]][b + loc[1]] = 2 if flag else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    # Find the units of the rule: single-colour components that are exactly a hollow
    # rectangle frame (both sides >= 3).
    visited = np.zeros_like(I, dtype=bool)
    frames = []
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = I[r, c]
            q = deque([(r, c)]); visited[r, c] = True; comp = []
            while q:
                y, x = q.popleft(); comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == color:
                        visited[ny, nx] = True; q.append((ny, nx))
            rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3 and set(comp) == _box(comp):
                frames.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))

    ops, sels = [], []
    for r0, c0, oh, ow in frames:
        # 1. erase the frame object (one connected same-colour region) to background
        ops.append(10 + bgc); sels.append(sel_of([(r0, c0)]))
        # 2. draw the horizontal bar through the frame's centre row
        cr, cc = r0 + oh // 2, c0 + ow // 2
        ops.append(2); sels.append(sel_of([(cr, x) for x in range(c0, c0 + ow)]))
        # 3. draw the vertical bar through the frame's centre column
        ops.append(2); sels.append(sel_of([(y, cc) for y in range(r0, r0 + oh)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6c434453"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6c434453"
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
                                f"for task 6c434453"
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
                    f"Failed to build a complete episode for task 6c434453 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6c434453-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
