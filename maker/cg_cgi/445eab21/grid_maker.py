"""
ARC Task: 445eab21 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is sampled as an episode-level role; box colours
    # are drawn per instance and the rule reads the winning colour from I.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + d * (b - a)))))

    if bgc is None:
        bgc = random.choice(range(10))
    hi_h = max(10, min(30, max_h))
    hi_w = max(10, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (10, hi_h))
        w = unifint(diff_lb, diff_ub, (10, hi_w))
        remcols = [c for c in range(10) if c != bgc]
        gi = [[bgc] * w for _ in range(h)]
        num = unifint(diff_lb, diff_ub, (1, 9))
        indss = {(i, j) for i in range(h) for j in range(w)}
        maxtrials = 4 * num
        succ, tr = 0, 0
        bigcol, area = 0, 0
        while succ < num and tr <= maxtrials:
            if len(remcols) == 0 or len(indss) == 0:
                break
            oh = random.randint(3, 7)
            ow = random.randint(3, 7)
            if oh * ow == area:
                continue
            subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
            if not subs:
                tr += 1
                continue
            loci, locj = random.choice(subs)
            bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
            col = random.choice(remcols)
            if bd.issubset(indss):
                remcols.remove(col)
                for (i, j) in bd:
                    if i in (loci, loci + oh - 1) or j in (locj, locj + ow - 1):
                        gi[i][j] = col
                succ += 1
                indss -= bd
                if oh * ow > area:
                    bigcol, area = col, oh * ow
            tr += 1
        if succ == 0:
            continue
        go = [[bigcol] * 2 for _ in range(2)]
        return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): each box is one unit,
    # scored by its height*width (golf: count of y in its row * count of y in
    # its column = box width * box height). The answer is a 2x2 patch of the
    # largest box's colour. Route: crop the canvas onto that largest box,
    # crop down to its 2x2 top-left corner (three outline cells + one interior
    # cell), then paint the interior cell with the box colour.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    best = None  # (area, col, r0, c0, r1, c1)
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            rs, cs = [], []
            while q:
                y, x = q.popleft()
                rs.append(y); cs.append(x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            a = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or a > best[0]:
                best = (a, col, r0, c0, r1, c1)

    _, col, r0, c0, r1, c1 = best
    ops, sels = [], []

    # 1. Crop the canvas to the winning box (its full bounding rectangle).
    #    Skip when the box already sits at the origin: then the corner crop
    #    alone yields the same 2x2.
    if (r0, c0) != (0, 0):
        ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    # 2. Crop to the box's 2x2 top-left corner (full rectangle).
    ops.append(33); sels.append([0, 0, 1, 1])
    # 3. The corner's inner cell is the box interior; paint it the box colour.
    ops.append(col); sels.append(sel_of([(1, 1)]))

    ops.append(34); sels.append([0, 0, 1, 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 445eab21"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 445eab21"
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
                                f"for task 445eab21"
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
                    f"Failed to build a complete episode for task 445eab21 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"445eab21-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
