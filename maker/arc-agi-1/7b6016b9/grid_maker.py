"""
ARC Task: 7b6016b9 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # generator samples bgc, fgc from colors excluding 2 and 3 (output-reserved)
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(lb, ub, bounds):
    lo, hi = bounds
    a = lo + int(round((hi - lo) * lb))
    b = lo + int(round((hi - lo) * ub))
    if a > b:
        a, b = b, a
    a = max(lo, a)
    b = min(hi, b)
    return random.randint(a, b)


def _bgc_components(grid, bgc):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(comp)
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    while True:
        h = _unifint(diff_lb, diff_ub, (5, max_h))
        w = _unifint(diff_lb, diff_ub, (5, max_w))
        numl = _unifint(diff_lb, diff_ub, (4, min(h, w)))
        gi = [[bgc] * w for _ in range(h)]

        iopts = list(range(1, h - 1))
        jopts = list(range(1, w - 1))
        numlh = random.randint(numl // 3, numl // 3 * 2)
        numlw = numl - numlh

        for _ in range(numlh):
            if not iopts:
                continue
            loci = random.choice(iopts)
            for x in (loci, loci + 1, loci - 1):
                if x in iopts:
                    iopts.remove(x)
            a, b = random.sample(range(w), 2)
            a = random.randint(0, a)
            b = random.randint(b, w - 1)
            lo, hh = min(a, b), max(a, b)
            for c in range(lo, hh + 1):
                gi[loci][c] = fgc

        for _ in range(numlw):
            if not jopts:
                continue
            locj = random.choice(jopts)
            for x in (locj, locj + 1, locj - 1):
                if x in jopts:
                    jopts.remove(x)
            a, b = random.sample(range(h), 2)
            a = random.randint(0, a)
            b = random.randint(b, h - 1)
            lo, hh = min(a, b), max(a, b)
            for r in range(lo, hh + 1):
                gi[r][locj] = fgc

        grid = np.array(gi)
        comps = _bgc_components(grid, bgc)
        interior = [comp for comp in comps
                    if not any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for (r, c) in comp)]
        tofill = set()
        for comp in interior:
            tofill.update(comp)
        if len(tofill) == 0:
            continue

        # seal interior regions: fill 8-neighborhood ring with fgc (modifies input)
        tofix = set()
        for (r, c) in tofill:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        tofix.add((nr, nc))
        tofix -= tofill
        for (r, c) in tofix:
            gi[r][c] = fgc

        go = [row[:] for row in gi]
        for (r, c) in tofill:
            go[r][c] = 2
        for r in range(h):
            for c in range(w):
                if go[r][c] == bgc:
                    go[r][c] = 3

        return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = majority color in I (only bgc + fgc present)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    seen = np.zeros((hi, wi), bool)

    # each connected bgc region: enclosed (no border cell) -> 2, else -> 3.
    # classification computed from I (does the region touch the grid edge), not read from O.
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                bord = False
                while stack:
                    cr, cc = stack.pop()
                    if cr == 0 or cr == hi - 1 or cc == 0 or cc == wi - 1:
                        bord = True
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                target = 3 if bord else 2
                ops.append(10 + target)      # FloodFill<target> from seed
                sels.append([r, c, 0, 0])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
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
                                f"for task 7b6016b9"
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
                    f"Failed to build a complete episode for task 7b6016b9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7b6016b9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
