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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    while True:
        h = max(5, min(unifint(diff_lb, diff_ub, (5, max_h)), max_h))
        w = max(5, min(unifint(diff_lb, diff_ub, (5, max_w)), max_w))
        numl = unifint(diff_lb, diff_ub, (4, min(h, w)))
        numl = max(4, min(numl, min(h, w)))
        gi = [[bgc] * w for _ in range(h)]
        iopts = list(range(1, h - 1))
        jopts = list(range(1, w - 1))
        numlh = random.randint(numl // 3, numl // 3 * 2)
        numlw = numl - numlh
        for _ in range(numlh):
            if not iopts:
                continue
            loci = random.choice(iopts)
            iopts = [x for x in iopts if x not in (loci, loci - 1, loci + 1)]
            a, b = random.sample(range(w), 2)
            a = random.randint(0, a)
            b = random.randint(b, w - 1)
            lo, hi_ = min(a, b), max(a, b)
            for j in range(lo, hi_ + 1):
                gi[loci][j] = fgc
        for _ in range(numlw):
            if not jopts:
                continue
            locj = random.choice(jopts)
            jopts = [x for x in jopts if x not in (locj, locj - 1, locj + 1)]
            a, b = random.sample(range(h), 2)
            a = random.randint(0, a)
            b = random.randint(b, h - 1)
            lo, hi_ = min(a, b), max(a, b)
            for i in range(lo, hi_ + 1):
                gi[i][locj] = fgc
        # enclosed bg regions (4-connected, not touching the border)
        seen = [[False] * w for _ in range(h)]
        tofill = set()
        for r in range(h):
            for c in range(w):
                if gi[r][c] == bgc and not seen[r][c]:
                    stack = [(r, c)]
                    seen[r][c] = True
                    comp = []
                    bord = False
                    while stack:
                        cr, cc = stack.pop()
                        comp.append((cr, cc))
                        if cr in (0, h - 1) or cc in (0, w - 1):
                            bord = True
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and gi[nr][nc] == bgc:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                    if not bord:
                        tofill.update(comp)
        if tofill:
            break
    tofix = set()
    for (r, c) in tofill:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in tofill:
                    tofix.add((nr, nc))
    for (r, c) in tofix:
        gi[r][c] = fgc
    go = [row[:] for row in gi]
    for (r, c) in tofill:
        go[r][c] = 2
    go = [[3 if v == bgc else v for v in row] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background = dominant colour along the grid border (as in the verifier).
    border = [(r, c) for r in range(hi) for c in range(wi)
              if r in (0, hi - 1) or c in (0, wi - 1)]
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    ops, sels = [], []

    # Reading: every background cell is "enclosed" (2) by default,
    # then colour 3 leaks in from the outside edge and overwrites what it reaches.
    # 1) Lay the base: all background-coloured cells -> 2.
    bg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
    ops.append(2); sels.append(sel_of(bg_cells))
    G = I.copy()
    for r, c in bg_cells:
        G[r, c] = 2

    # 2) Let 3 flood in from the border: one FloodFill3 per still-2 region
    #    that touches the edge, seeded at its first border cell.
    for (r, c) in border:
        if G[r, c] != 2:
            continue
        ops.append(13); sels.append([r, c, 0, 0])
        stack = [(r, c)]
        G[r, c] = 3
        while stack:
            cr, cc = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < hi and 0 <= nc < wi and G[nr, nc] == 2:
                    G[nr, nc] = 3
                    stack.append((nr, nc))

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
