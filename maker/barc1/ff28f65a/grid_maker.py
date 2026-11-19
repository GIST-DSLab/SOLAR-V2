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

# The order the pips are lit: row-major on the 3x3 board, every other cell
# (flat indices 0, 2, 4, 6, 8).
MPR = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
NRED_VALUES = [1, 2, 3, 4, 5]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    v = a + int(round((b - a) * d))
    v = max(a, min(b, v))
    return random.randint(min(lo, v), v) if v >= a else a


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(NRED_VALUES):
        ex = list(NRED_VALUES) + [random.choice(NRED_VALUES) for _ in range(n_ex - len(NRED_VALUES))]
        random.shuffle(ex)
    else:
        ex = random.sample(NRED_VALUES, n_ex)
    plan = [{"nred": n} for n in ex]
    plan.append({"nred": random.choice(ex)})
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, nred=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    if nred is None:
        nred = random.randint(1, 5)
    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (3, max_w))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def place(color_fn, target, maxtr):
        succ, tr = 0, 0
        nonlocal inds
        while tr < maxtr and succ < target:
            tr += 1
            oh = random.randint(1, h // 2 + 1)
            ow = random.randint(1, w // 2 + 1)
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            li, lj = random.choice(cands)
            bd = {(i, j) for i in range(li, li + oh) for j in range(lj, lj + ow)}
            if bd.issubset(inds):
                succ += 1
                nb = set()
                for (i, j) in bd:
                    nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
                inds = (inds - bd) - nb
                col = color_fn()
                for (i, j) in bd:
                    gi[i][j] = col
        return succ

    succ = place(lambda: 2, nred, 60 * nred)
    if succ != nred:
        raise ValueError("could not place all red objects")
    namt = _unifint(diff_lb, diff_ub, (0, nred * 2))
    remcols = [c for c in cols if c != bgc]
    place(lambda: random.choice(remcols), namt, 5 * namt)

    go = [[bgc] * 3 for _ in range(3)]
    for k in range(1, nred + 1):
        r, c = MPR[k]
        go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant non-red colour of I
    cnt = Counter(int(x) for x in I.flatten() if int(x) != 2)
    bgc = cnt.most_common(1)[0][0] if cnt else 0

    # red squares = 4-connected components of colour 2
    seen = np.zeros((hi, wi), dtype=bool)
    reds = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == 2 and not seen[r, c]:
                comp = []
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 2:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                reds.append(comp)
    nred = min(len(reds), 5)

    ops, sels = [], []

    # Output canvas: a 3x3 background board. Use an all-background 3x3 window
    # of I if there is one (the full rectangle is the intended selection).
    src = None
    for r in range(hi - 2):
        for c in range(wi - 2):
            if bool((I[r:r + 3, c:c + 3] == bgc).all()):
                src = (r, c)
                break
        if src is not None:
            break
    if src is not None:
        ops.append(33); sels.append([src[0], src[1], 2, 2])  # full 3x3 rectangle
    else:
        ops.append(33); sels.append([0, 0, 2, 2])            # full 3x3 rectangle
        ops.append(bgc); sels.append([0, 0, 2, 2])           # blank whole board to bg

    # One blue cell per red square: fill row-major, skipping every other cell.
    for k in range(1, nred + 1):
        pr, pc = MPR[k]
        ops.append(1); sels.append(sel_of([(pr, pc)]))

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
