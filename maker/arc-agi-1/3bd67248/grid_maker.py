"""
ARC Task: 3bd67248 (RE-ARC) — LLM-generated grid_maker
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


def unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = round(a + (b - a) * diff_lb)
    hi = round(a + (b - a) * diff_ub)
    if lo < a:
        lo = a
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # bgc / linc are ordinary colors; generator excludes 2 and 4 (reserved for the
    # drawn 4-band and 2-staircase). Rule is color-agnostic beyond that, but fix roles.
    cols = [c for c in range(10) if c not in (2, 4)]
    bgc, linc = random.sample(cols, 2)

    # Discrete structural variant = the rotation (which edge the line sits on).
    # Line-edge uniquely determines rotation, so cover all 4 in examples.
    ROT = [0, 1, 2, 3]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT):
        examples = [{"rot": r} for r in ROT]
        examples += [{"rot": random.choice(ROT)} for _ in range(n_ex - len(ROT))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROT, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    maxdim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (3, min(15, maxdim)))
    w = unifint(diff_lb, diff_ub, (3, min(15, maxdim)))
    upper = max(1, maxdim // max(h, w))
    fac = unifint(diff_lb, diff_ub, (1, upper))

    # base frame (identity): line = left column, 4-band = bottom row (cols>=1),
    # 2-staircase shoots up-right from (h-2, 1).
    gi = np.full((h, w), bgc, dtype=int)
    gi[:, 0] = linc
    go = gi.copy()
    go[h - 1, 1:w] = 4
    r, c = h - 2, 1
    while 0 <= r < h and 0 <= c < w:
        go[r, c] = 2
        r -= 1
        c += 1

    gi = np.rot90(gi, k=rot)
    go = np.rot90(go, k=rot)
    gi = np.kron(gi, np.ones((fac, fac), dtype=int))
    go = np.kron(go, np.ones((fac, fac), dtype=int))

    return {"input": gi.astype(int).tolist(), "output": go.astype(int).tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # --- measure rule from I ---
    # line color = minority color (thin edge band vs the rest of the grid)
    cnt = Counter(I.flatten().tolist())
    linc = min(cnt, key=lambda k: cnt[k])

    def row_all(rr):
        return bool(np.all(I[rr, :] == linc))

    def col_all(cc):
        return bool(np.all(I[:, cc] == linc))

    # which full edge holds the line, and its thickness f (= upscale factor)
    if row_all(0):
        edge = "TOP"
        f = 0
        while f < H and row_all(f):
            f += 1
    elif row_all(H - 1):
        edge = "BOTTOM"
        f = 0
        while f < H and row_all(H - 1 - f):
            f += 1
    elif col_all(0):
        edge = "LEFT"
        f = 0
        while f < W and col_all(f):
            f += 1
    else:
        edge = "RIGHT"
        f = 0
        while f < W and col_all(W - 1 - f):
            f += 1

    # number of diagonal steps (base_h/base_w = dim//f)
    N = min(H // f, W // f) - 1

    ops, sels = [], []

    # 4-band: the full edge perpendicular to the line, on the far side, as ONE region.
    # staircase: fac x fac blocks stepping diagonally away from the line corner.
    if edge == "LEFT":
        ops.append(4); sels.append([H - f, f, f - 1, W - 1 - f])        # bottom band
        r0, c0, dr, dc = H - 2 * f, f, -f, f                            # up-right
    elif edge == "BOTTOM":
        ops.append(4); sels.append([0, W - f, H - f - 1, f - 1])        # right band
        r0, c0, dr, dc = 0, f, f, f                                     # down-right
    elif edge == "TOP":
        ops.append(4); sels.append([f, 0, H - 1 - f, f - 1])           # left band
        r0, c0, dr, dc = f, f, f, f                                     # down-right
    else:  # RIGHT
        ops.append(4); sels.append([0, 0, f - 1, W - 1 - f])           # top band
        r0, c0, dr, dc = f, W - 2 * f, f, -f                           # down-left

    r, c = r0, c0
    for _ in range(N):
        ops.append(2); sels.append([r, c, f - 1, f - 1])
        r += dr
        c += dc

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
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
                                f"for task 3bd67248"
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
                    f"Failed to build a complete episode for task 3bd67248 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bd67248-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
