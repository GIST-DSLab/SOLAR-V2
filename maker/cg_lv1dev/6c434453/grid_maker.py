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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def dneigh(cell):
        r, c = cell
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def box_of(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        b = set()
        for y in range(r0, r1 + 1):
            b.add((y, c0)); b.add((y, c1))
        for x in range(c0, c1 + 1):
            b.add((r0, x)); b.add((r1, x))
        return b

    def normalize(cells):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        return {(r - r0, c - c0) for r, c in cells}

    if bgc is None:
        cols = [c for c in range(10) if c != 2]
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in range(10) if c not in (2, bgc)]
        ccols = random.sample(remcols, random.randint(1, len(remcols)))

    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 16)))
    succ, tr, maxtr = 0, 0, 5 * nobjs
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    while succ < nobjs and tr < maxtr:
        tr += 1
        if random.choice((True, False)):
            oh = random.choice((3, 5))
            ow = random.choice((3, 5))
            obji = box_of({(0, 0), (oh - 1, ow - 1)})
        else:
            oh = random.randint(1, 5)
            ow = random.randint(1, 5)
            bounds = {(r, c) for r in range(oh) for c in range(ow)}
            ncells = random.randint(1, oh * ow)
            obji = {random.choice(sorted(bounds))}
            for _ in range(ncells - 1):
                nb = set()
                for p in obji:
                    nb |= dneigh(p)
                cand = sorted((bounds - obji) & nb)
                obji.add(random.choice(cand))
            obji = normalize(obji)
        oh = max(p[0] for p in obji) + 1
        ow = max(p[1] for p in obji) + 1
        flag = obji == box_of(obji) and {oh, ow}.issubset({3, 5})
        if flag:
            objo = {(r, ow // 2) for r in range(oh)} | {(oh // 2, c) for c in range(ow)}
            tocover = {(r, c) for r in range(oh) for c in range(ow)}
        else:
            objo = obji
            tocover = obji
        cands = sorted(ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow)
        if not cands:
            continue
        lr, lc = random.choice(cands)
        plcdi = {(r + lr, c + lc) for r, c in obji}
        if plcdi.issubset(inds):
            succ += 1
            tci = {(r + lr, c + lc) for r, c in tocover}
            nb = set()
            for p in tci:
                nb |= dneigh(p)
            inds = (inds - tci) - nb
            col = random.choice(ccols)
            for r, c in plcdi:
                gi[r][c] = col
            for r, c in objo:
                go[r + lr][c + lc] = 2 if flag else col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the generator paints the canvas with bgc first; it is the
    # dominant colour (objects are sparse and separated by background).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Find 4-connected single-colour objects.
    visited = np.zeros_like(I, dtype=bool)
    frames = []
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = I[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == color:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            # Generator rule: a hollow box whose sides are 3 or 5 is a "frame".
            if oh not in (3, 5) or ow not in (3, 5):
                continue
            outline = set()
            for y in range(r0, r1 + 1):
                outline.add((y, c0)); outline.add((y, c1))
            for x in range(c0, c1 + 1):
                outline.add((r0, x)); outline.add((r1, x))
            if set(comp) == outline:
                frames.append((r0, c0, oh, ow, sorted(comp)))

    ops, sels = [], []
    # Each frame is one unit that gets swapped for a plus stamp of the same
    # footprint: erase the frame object, then stamp the plus (centre row and
    # centre column of the frame's backdrop) in one stroke.
    for r0, c0, oh, ow, comp in frames:
        ops.append(bgc); sels.append(sel_of(comp))            # erase the frame
        plus = {(r0 + oh // 2, c0 + x) for x in range(ow)} | \
               {(r0 + y, c0 + ow // 2) for y in range(oh)}
        ops.append(2); sels.append(sel_of(sorted(plus)))      # stamp the plus

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
