"""
ARC Task: ff28f65a (RE-ARC) — LLM-generated grid_maker
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

MPR = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    nred = random.randint(1, 5)
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def place(color_fn, n_target):
        nonlocal inds
        succ, tr, maxtr = 0, 0, 5 * n_target
        while tr < maxtr and succ < n_target:
            tr += 1
            oh = random.randint(1, h // 2 + 1)
            ow = random.randint(1, w // 2 + 1)
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loci, locj = random.choice(cands)
            bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
            if bd.issubset(inds):
                succ += 1
                nb = set()
                for (i, j) in bd:
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nb.add((i + di, j + dj))
                inds = (inds - bd) - nb
                col = color_fn()
                for (i, j) in bd:
                    gi[i][j] = col
        return succ

    nblue = place(lambda: 2, nred)
    if nblue == 0:
        raise ValueError("no red objects placed")
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    remcols = [c for c in cols if c != bgc]
    if namt > 0:
        place(lambda: random.choice(remcols), namt)

    go = [[bgc] * 3 for _ in range(3)]
    for k in range(nblue):
        r, c = MPR[k + 1]
        go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): each red object is one
    # unit; the number of red objects decides how many pips get lit on a 3x3
    # board drawn on the background colour, in the fixed die order
    # (0,0),(0,2),(1,1),(2,0),(2,2). The two readings agree, so the plan is the same.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: in every demo output, cell (0,1) is never a pip, so it is the bg colour
    bgc = None
    if examples:
        try:
            bgc = int(np.asarray(examples[0][1], dtype=int)[0, 1])
        except Exception:
            bgc = None
    if bgc is None:
        cnt = Counter(int(x) for x in I.flatten() if int(x) != 2)
        bgc = cnt.most_common(1)[0][0] if cnt else 0

    # count red objects (4-connected components of colour 2)
    seen = np.zeros((hi, wi), dtype=bool)
    nred = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == 2 and not seen[r, c]:
                nred += 1
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 2:
                            seen[ny, nx] = True
                            q.append((ny, nx))
    nred = min(nred, 5)

    ops, sels = [], []

    # shrink the canvas to the 3x3 board: use an all-background 3x3 window if one exists
    src = None
    for r in range(hi - 2):
        for c in range(wi - 2):
            if bool((I[r:r + 3, c:c + 3] == bgc).all()):
                src = (r, c)
                break
        if src is not None:
            break

    if src is not None:
        # whole 3x3 rectangle, all background: crop to it
        ops.append(33); sels.append([src[0], src[1], 2, 2])
        cur = np.zeros((3, 3), dtype=int) if bgc == 0 else I[src[0]:src[0] + 3, src[1]:src[1] + 3].copy()
    else:
        win = I[0:3, 0:3]
        cropped = np.where(win != 0, win, 0)
        crop_changes = not (hi == 3 and wi == 3 and bool((win != 0).all()))
        if crop_changes:
            ops.append(33); sels.append([0, 0, 2, 2])   # full 3x3 rectangle
        cur = cropped.copy()
        if not bool((cur == bgc).all()):
            ops.append(bgc); sels.append([0, 0, 2, 2])  # blank the whole board to bg
            cur[:, :] = bgc

    # light one pip for each red object, in pip order
    for k in range(1, nred + 1):
        pr, pc = MPR[k]
        ops.append(1); sels.append(sel_of([(pr, pc)]))
        cur[pr, pc] = 1

    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
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
                                f"for task ff28f65a"
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
                    f"Failed to build a complete episode for task ff28f65a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff28f65a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
