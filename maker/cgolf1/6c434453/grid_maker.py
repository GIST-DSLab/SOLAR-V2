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


def sample_colors(num_examples=None) -> dict:
    # Colour 2 is reserved for the output crosses, so the background is never 2.
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lo, hi):
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return max(lo, min(hi, int(round(random.uniform(a, b)))))

    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    h = min(h, max_h)
    w = min(w, max_w)
    remcols = [c for c in cols if c != bgc]
    ncols = unifint(1, 8)
    ccols = random.sample(remcols, min(ncols, len(remcols)))
    nobjs = unifint(2, max(2, (h * w) // 16))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def box_of(oh, ow):
        s = set()
        for i in range(oh):
            s.add((i, 0)); s.add((i, ow - 1))
        for j in range(ow):
            s.add((0, j)); s.add((oh - 1, j))
        return s

    def dnb(cells):
        out = set()
        for (i, j) in cells:
            out |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        return out

    while succ < nobjs and tr < maxtr:
        tr += 1
        if random.choice((True, False)):
            oh = random.choice((3, 5)); ow = random.choice((3, 5))
            obji = box_of(oh, ow)
        else:
            oh = random.randint(1, 5); ow = random.randint(1, 5)
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            ncells = random.randint(1, oh * ow)
            obji = {random.choice(sorted(bounds))}
            for _ in range(ncells - 1):
                cand = sorted((bounds - obji) & dnb(obji))
                if not cand:
                    break
                obji.add(random.choice(cand))
            mi = min(i for i, j in obji); mj = min(j for i, j in obji)
            obji = {(i - mi, j - mj) for i, j in obji}
        oh = max(i for i, j in obji) + 1
        ow = max(j for i, j in obji) + 1
        flag = obji == box_of(oh, ow) and {oh, ow}.issubset({3, 5})
        if flag:
            objo = {(i, ow // 2) for i in range(oh)} | {(oh // 2, j) for j in range(ow)}
            tocover = {(i, j) for i in range(oh) for j in range(ow)}
        else:
            objo = obji
            tocover = obji
        if h - oh < 0 or w - ow < 0:
            continue
        li = random.randint(0, h - oh); lj = random.randint(0, w - ow)
        plcdi = {(i + li, j + lj) for i, j in obji}
        if plcdi.issubset(inds):
            succ += 1
            tc = {(i + li, j + lj) for i, j in tocover}
            inds = (inds - tc) - dnb(tc)
            col = random.choice(ccols)
            for (i, j) in plcdi:
                gi[i][j] = col
            for (i, j) in objo:
                go[i + li][j + lj] = 2 if flag else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Both readings agree: each hollow 3/5-sided ring is one unit. Clear its window,
    # then draw the plus through its centre in colour 2. The steps follow that reading.
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour that dominates every input in the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            cnt.update(ei.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])

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
            r0 = min(p[0] for p in comp); r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp); c1 = max(p[1] for p in comp)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if oh in (3, 5) and ow in (3, 5):
                ring = set()
                for y in range(r0, r1 + 1):
                    ring.add((y, c0)); ring.add((y, c1))
                for x in range(c0, c1 + 1):
                    ring.add((r0, x)); ring.add((r1, x))
                if set(comp) == ring:
                    frames.append((r0, c0, oh, ow, sorted(ring)))

    ops, sels = [], []
    for r0, c0, oh, ow, ring in frames:
        # erase the ring object (its window becomes plain background)
        ops.append(bgc); sels.append(sel_of(ring))
        # horizontal bar of the plus through the centre row
        ops.append(2); sels.append(sel_of([(r0 + oh // 2, c0 + x) for x in range(ow)]))
        # vertical bar through the centre column
        ops.append(2); sels.append(sel_of([(r0 + y, c0 + ow // 2) for y in range(oh)]))

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
