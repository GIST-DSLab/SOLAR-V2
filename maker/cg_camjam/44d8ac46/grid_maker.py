"""
ARC Task: 44d8ac46 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = _unifint(diff_lb, diff_ub, (1, 10))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        tr += 1
        if not remcols or not indss:
            break
        oh = random.randint(5, 7)
        ow = random.randint(5, 7)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            continue
        loci, locj = random.choice(subs)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        col = random.choice(remcols)
        if not bd.issubset(indss):
            continue
        if random.choice((True, False)):
            dim = random.randint(1, min(oh, ow) - 2)
            iloci = random.randint(1, oh - dim - 1)
            ilocj = random.randint(1, ow - dim - 1)
            inpart = {(loci + iloci + a, locj + ilocj + b)
                      for a in range(dim) for b in range(dim)}
        else:
            cnds = {(loci + a, locj + b) for a in range(1, oh - 1) for b in range(1, ow - 1)}
            ch = random.choice(sorted(cnds))
            inpart = {ch}
            kk = _unifint(diff_lb, diff_ub, (1, len(cnds)))
            for _ in range(kk - 1):
                nb = set()
                for (y, x) in inpart:
                    nb |= {(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)}
                cand = sorted((cnds - inpart) & nb)
                if not cand:
                    break
                inpart.add(random.choice(cand))
        rs = [p[0] for p in inpart]
        cs = [p[1] for p in inpart]
        hh = max(rs) - min(rs) + 1
        ww = max(cs) - min(cs) + 1
        incol = 2 if (hh == ww and len(inpart) == hh * ww) else bgc
        for (y, x) in bd:
            gi[y][x] = col
            go[y][x] = col
        for (y, x) in inpart:
            gi[y][x] = bgc
            go[y][x] = incol
        succ += 1
        outb = {(loci + a, locj + b) for a in range(-1, oh + 1) for b in range(-1, ow + 1)}
        indss = indss - bd - outb
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Boxes never touch the last row/column, so that corner is always background.
    bgc = int(I[hi - 1, wi - 1])

    ops, sels = [], []

    # Find the 4-connected regions of background cells. A hole is one of these
    # regions; the unit is the hole, and it gets one FloodFill.
    seen = np.zeros((hi, wi), dtype=bool)
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] != bgc:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == bgc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            r0 = min(p[0] for p in comp)
            r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            c1 = max(p[1] for p in comp)
            s = r1 - r0 + 1
            # The region must be a solid s×s square of background.
            if s != c1 - c0 + 1 or len(comp) != s * s:
                continue
            # The ring around it must lie inside the grid and be one single frame colour.
            if r0 - 1 < 0 or c0 - 1 < 0 or r1 + 1 >= hi or c1 + 1 >= wi:
                continue
            ring = [(y, x) for y in range(r0 - 1, r1 + 2) for x in range(c0 - 1, c1 + 2)
                    if not (r0 <= y <= r1 and c0 <= x <= c1)]
            ring_cols = {int(I[y, x]) for (y, x) in ring}
            if len(ring_cols) != 1 or bgc in ring_cols:
                continue
            # An enclosed square hole: flood it with 2 from its top-left cell.
            ops.append(12)
            sels.append(sel_of([(r0, c0)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
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
                                f"for task 44d8ac46"
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
                    f"Failed to build a complete episode for task 44d8ac46 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44d8ac46-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
