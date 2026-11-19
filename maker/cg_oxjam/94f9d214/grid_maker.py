"""
ARC Task: 94f9d214 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=1, acol=3, bcol=7) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    while True:
        flip = random.choice((True, False))
        if flip:
            H, W = max_w, max_h // 2
        else:
            H, W = max_h, max_w // 2
        h = unifint(2, min(30, H))
        w = unifint(2, min(14, W))
        inds = [(i, j) for i in range(h) for j in range(w)]
        numadev = unifint(0, (h * w) // 2)
        numbdev = unifint(0, (h * w) // 2)
        numa = random.choice((numadev, h * w - numadev))
        numb = random.choice((numadev, h * w - numbdev))
        numa = min(max(1, numa), h * w - 1)
        numb = min(max(1, numb), h * w - 1)
        aset = random.sample(inds, numa)
        bset = random.sample(inds, numb)
        A = np.full((h, w), bgc, dtype=int)
        B = A.copy()
        go = A.copy()
        for r, c in aset:
            A[r, c] = acol
        for r, c in bset:
            B[r, c] = bcol
        for r, c in set(inds) - set(aset) - set(bset):
            go[r, c] = 2
        gi = np.hstack([A, B])
        if flip:
            gi = gi.T
            go = go.T

        def ok(a, b):
            pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
            return len(pa) == 2 and len(pb) == 2 and len(pa & pb) == 1

        Hh, Ww = gi.shape
        alts = []
        if Ww % 2 == 0:
            alts.append(ok(gi[:, :Ww // 2], gi[:, Ww // 2:]))
        if Hh % 2 == 0:
            alts.append(ok(gi[:Hh // 2], gi[Hh // 2:]))
        if sum(alts) != 1:
            continue  # split direction must be unambiguous from the input
        return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- split I into halves A (top/left) and B; the valid split is the one where
    #     each half has exactly two colours sharing one (the background) ---
    cands = []
    if wi % 2 == 0:
        cands.append((I[:, :wi // 2], I[:, wi // 2:]))
    if hi % 2 == 0:
        cands.append((I[:hi // 2, :], I[hi // 2:, :]))
    pick = None
    for a, b in cands:
        pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
        if len(pa) == 2 and len(pb) == 2 and len(pa & pb) == 1:
            pick = (a, b, (pa & pb).pop())
            break
    if pick is None:
        a, b = cands[0]
        sh = set(a.flatten().tolist()) & set(b.flatten().tolist())
        pick = (a, b, min(sh) if sh else int(np.bincount(a.flatten()).argmax()))
    A, B, bgc = pick
    ho, wo = A.shape
    G = A.copy()
    ops, sels = [], []
    acol_set = set(A.flatten().tolist()) - {bgc}
    acol = min(acol_set) if acol_set else None

    def comps(grid, color):
        seen, res = set(), []
        for r in range(ho):
            for c in range(wo):
                if grid[r, c] != color or (r, c) in seen:
                    continue
                comp = [(r, c)]
                q = deque([(r, c)])
                seen.add((r, c))
                while q:
                    y, x = q.popleft()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < ho and 0 <= nx < wo and (ny, nx) not in seen \
                                and grid[ny, nx] == color:
                            seen.add((ny, nx))
                            q.append((ny, nx))
                            comp.append((ny, nx))
                res.append(comp)
        return res

    # 1. Lay half B's pattern over half A (the cells B's pattern lands on), stamped in
    #    A's pattern colour, so the background still showing is exactly where both
    #    halves are background. A background pocket of A that B blocks completely
    #    needs no marking at all, so it is left alone.
    if acol is not None:
        land = []
        for pocket in comps(A, bgc):
            covered = [(r, c) for r, c in pocket if B[r, c] != bgc]
            if len(covered) < len(pocket):
                land += covered
        if land:
            ops.append(int(acol))
            sels.append(sel_of(sorted(land)))
            for r, c in land:
                G[r, c] = acol

    # 2. B has served its purpose: crop the canvas to half A (also keeps the floods
    #    below from leaking into B's half).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # 3. Every background pocket still showing through the overlay -> flood with 2.
    for comp in comps(G, bgc):
        if all(B[r, c] != bgc for r, c in comp):
            continue  # pocket B blocks entirely: stays background
        ops.append(12)
        sels.append(sel_of([comp[0]]))

    # 4. Each blob of the overlaid (union) pattern -> flood back to background.
    if acol is not None:
        for comp in comps(G, acol):
            ops.append(10 + int(bgc))
            sels.append(sel_of([comp[0]]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
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
                                f"for task 94f9d214"
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
                    f"Failed to build a complete episode for task 94f9d214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"94f9d214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
