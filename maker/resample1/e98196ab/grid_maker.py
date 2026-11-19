"""
ARC Task: e98196ab (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    topc = random.choice(rem)
    rem = [c for c in rem if c != topc and c != 0]  # botc must be movable (nonzero)
    botc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "topc": topc, "botc": botc,
            "instance_plan": plan}


def _uni(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, lo + int(round(d * (hi - lo)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, topc, botc,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    if transpose:
        rows_lim, cols_lim = max_w, max_h  # pre-transpose dims
    else:
        rows_lim, cols_lim = max_h, max_w
    hmax = min(14, (rows_lim - 1) // 2)
    wmax = min(30, cols_lim)
    h = _uni(diff_lb, diff_ub, 3, hmax)
    w = _uni(diff_lb, diff_ub, 3, wmax)
    inds = [(i, j) for i in range(h) for j in range(w)]
    nocc = _uni(diff_lb, diff_ub, 2, (h * w) // 2)
    subs = random.sample(inds, nocc)
    numa = random.randint(1, nocc - 1)
    A = set(random.sample(subs, numa))
    B = set(subs) - A
    topg = [[bgc] * w for _ in range(h)]
    botg = [[bgc] * w for _ in range(h)]
    for (i, j) in A:
        topg[i][j] = topc
    for (i, j) in B:
        botg[i][j] = botc
    go = [row[:] for row in topg]
    for (i, j) in B:
        go[i][j] = botc
    gi = [row[:] for row in topg] + [[linc] * w] + [row[:] for row in botg]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    horizontal = (H % 2 == 1) and len(set(I[H // 2].tolist())) == 1
    ops, sels = [], []
    if horizontal:
        h = H // 2
        half = I[h + 1:, :]
        bgc = Counter(half.flatten().tolist()).most_common(1)[0][0]
        B = [(r, c) for r in range(h + 1, H) for c in range(W) if I[r, c] != bgc]
        move_op, dr, dc = 20, -1, 0
        crop = [0, 0, h - 1, W - 1]
    else:
        h = W // 2
        half = I[:, h + 1:]
        bgc = Counter(half.flatten().tolist()).most_common(1)[0][0]
        B = [(r, c) for r in range(H) for c in range(h + 1, W) if I[r, c] != bgc]
        move_op, dr, dc = 23, 0, -1
        crop = [0, 0, H - 1, h - 1]
    steps = h + 1
    if B:
        movable = all(I[r, c] != 0 for r, c in B)
        if movable:
            # slide the second half's marks onto the first half (one grab, then empties)
            ops.append(move_op); sels.append(sel_of(B))
            for _ in range(steps - 1):
                ops.append(move_op); sels.append(sel_of([]))
        else:
            # colour 0 cannot be carried by Move; paint them at the overlaid position
            tgt = [(r + dr * steps, c + dc * steps) for r, c in B]
            ops.append(0); sels.append(sel_of(tgt))
    # crop to the first half (full rectangle)
    ops.append(33); sels.append(crop)
    ops.append(34); sels.append([0, 0, crop[2], crop[3]])
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
                        f"num_examples+1 ({num_examples + 1}) for task e98196ab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e98196ab"
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
                                f"for task e98196ab"
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
                    f"Failed to build a complete episode for task e98196ab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e98196ab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
