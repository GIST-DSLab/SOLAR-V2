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


def _unif(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    d = random.uniform(diff_lb, diff_ub)
    a = lo + int(round(d * (hi - lo) * 0.0))
    b = lo + int(round(d * (hi - lo)))
    return random.randint(min(a, b), max(a, b))


ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    topc = random.choice(rem)
    rem = [c for c in rem if c != topc]
    botc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        examples = [dict(v) for v in ORIENTS]
        examples += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "topc": topc, "botc": botc,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, topc, botc,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    lim = min(max_h, max_w)
    h = _unif(diff_lb, diff_ub, 3, min(14, (lim - 1) // 2))
    w = _unif(diff_lb, diff_ub, 3, min(30, lim))
    inds = [(i, j) for i in range(h) for j in range(w)]
    nocc = _unif(diff_lb, diff_ub, 2, max(2, (h * w) // 2))
    subs = random.sample(inds, nocc)
    numa = random.randint(1, nocc - 1)
    A = random.sample(subs, numa)
    Aset = set(A)
    B = [x for x in subs if x not in Aset]
    topg = np.full((h, w), bgc, dtype=int)
    botg = np.full((h, w), bgc, dtype=int)
    for (i, j) in A:
        topg[i, j] = topc
    for (i, j) in B:
        botg[i, j] = botc
    go = topg.copy()
    for (i, j) in B:
        go[i, j] = botc
    br = np.full((1, w), linc, dtype=int)
    gi = np.vstack([topg, br, botg])
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: the second half's figures are laid over the first half at the same
    # relative position, i.e. they TRANSLATE across the separator by (half+1).
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # locate separator line from I: middle row/col that is uniform and not background
    row_split = None
    if h % 2 == 1:
        mr = h // 2
        if len(set(I[mr].tolist())) == 1:
            rest = np.delete(I, mr, axis=0)
            if Counter(rest.flatten().tolist()).most_common(1)[0][0] != I[mr, 0]:
                row_split = True
    if row_split is None:
        row_split = False

    if row_split:
        hh = h // 2
        S_r0, S_c0 = hh + 1, 0
        S = I[hh + 1:, :]
        ho, wo = hh, w
        shift, move_op = hh + 1, 20      # MoveU
        dr, dc = -1, 0
    else:
        ww = w // 2
        S_r0, S_c0 = 0, ww + 1
        S = I[:, ww + 1:]
        ho, wo = h, ww
        shift, move_op = ww + 1, 23      # MoveL
        dr, dc = 0, -1

    # second-half background = its majority colour (foreground < half its area)
    sbg = Counter(S.flatten().tolist()).most_common(1)[0][0]
    fg = [(S_r0 + r, S_c0 + c) for r in range(S.shape[0]) for c in range(S.shape[1])
          if int(S[r, c]) != sbg]

    ops, sels = [], []
    if fg:
        fg_cols = set(int(I[r, c]) for r, c in fg)
        if 0 not in fg_cols:
            # grab the second half's figures once, then slide them across the line
            ops.append(move_op); sels.append(sel_of(fg))
            for _ in range(shift - 1):
                ops.append(move_op); sels.append(sel_of([]))
            # vacated footprint lies in the half that is cropped away: no repair needed
        else:
            # colour 0 cannot be carried by Move/Paste: paint the landing cells directly
            by_col = {}
            for (r, c) in fg:
                by_col.setdefault(int(I[r, c]), []).append((r + dr * shift, c + dc * shift))
            for col, cells in by_col.items():
                cells = [(r, c) for (r, c) in cells if int(I[r, c]) != col]
                if cells:
                    ops.append(col); sels.append(sel_of(cells))

    # keep only the first half: whole rectangle, background included
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
