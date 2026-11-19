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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def _box(oh, ow):
    cells = set()
    for r in range(oh):
        cells.add((r, 0)); cells.add((r, ow - 1))
    for c in range(ow):
        cells.add((0, c)); cells.add((oh - 1, c))
    return cells


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, **kwargs) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    h = max(1, min(h, max_h)); w = max(1, min(w, max_w))
    nobjs = unifint(2, max(2, (h * w) // 16))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}

    def dnb(cells):
        out = set()
        for r, c in cells:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        if random.choice((True, False)):
            oh = random.choice((3, 5)); ow = random.choice((3, 5))
            obji = _box(oh, ow)
        else:
            oh = random.randint(1, 5); ow = random.randint(1, 5)
            bounds = {(r, c) for r in range(oh) for c in range(ow)}
            ncells = random.randint(1, oh * ow)
            obji = {random.choice(sorted(bounds))}
            for _ in range(ncells - 1):
                cand = sorted((bounds - obji) & dnb(obji))
                obji.add(random.choice(cand))
            mr = min(r for r, _ in obji); mc = min(c for _, c in obji)
            obji = {(r - mr, c - mc) for r, c in obji}
        oh = max(r for r, _ in obji) + 1
        ow = max(c for _, c in obji) + 1
        flag = obji == _box(oh, ow) and {oh, ow}.issubset({3, 5})
        if flag:
            objo = {(r, ow // 2) for r in range(oh)} | {(oh // 2, c) for c in range(ow)}
            tocover = {(r, c) for r in range(oh) for c in range(ow)}
        else:
            objo = obji
            tocover = obji
        cands = [(r, c) for (r, c) in inds if r <= h - oh and c <= w - ow]
        if not cands:
            continue
        lr, lc = random.choice(cands)
        plcdi = {(r + lr, c + lc) for r, c in obji}
        if plcdi.issubset(inds):
            succ += 1
            tc = {(r + lr, c + lc) for r, c in tocover}
            inds = (inds - tc) - dnb(tc)
            col = random.choice(ccols)
            for r, c in plcdi:
                gi[r][c] = col
            for r, c in objo:
                go[r + lr][c + lc] = 2 if flag else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour dominant across the episode's inputs (all share bgc)
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_in, _ in examples:
            cnt.update(np.asarray(ex_in, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # find hollow-ring objects (single-colour, 4-connected, exactly a box outline, sides 3 or 5)
    visited = np.zeros_like(I, dtype=bool)
    rings = []
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            q = deque([(r, c)]); visited[r, c] = True; comp = []
            while q:
                y, x = q.popleft(); comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == col:
                        visited[ny, nx] = True; q.append((ny, nx))
            r0 = min(p[0] for p in comp); c0 = min(p[1] for p in comp)
            oh = max(p[0] for p in comp) - r0 + 1
            ow = max(p[1] for p in comp) - c0 + 1
            if oh in (3, 5) and ow in (3, 5):
                box = {(r0 + y, c0 + x) for (y, x) in _box(oh, ow)}
                if set(comp) == box:
                    rings.append((r0, c0, oh, ow, sorted(comp)))

    ops, sels = [], []
    for r0, c0, oh, ow, comp in rings:
        # 1. erase the ring as one connected object: flood-fill it with background
        sr, sc = comp[0]
        ops.append(10 + bgc); sels.append([sr, sc, 0, 0])
        # 2. draw the plus through the ring's centre, clipped to its bounding box
        cr, cc = r0 + oh // 2, c0 + ow // 2
        plus = {(cr, c0 + x) for x in range(ow)} | {(r0 + y, cc) for y in range(oh)}
        ops.append(2); sels.append(sel_of(sorted(plus)))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
