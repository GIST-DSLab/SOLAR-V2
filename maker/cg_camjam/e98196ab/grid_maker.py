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
    rem = [c for c in rem if c != topc]
    botc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "topc": topc, "botc": botc,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=1, topc=2,
             botc=3, mirror=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if mirror is None:
        mirror = random.choice([True, False])
    # stacked dimension: 2h+1, other dimension: w
    stack_lim = max_w if mirror else max_h
    other_lim = max_h if mirror else max_w
    h = unifint(diff_lb, diff_ub, (3, max(3, min(14, (stack_lim - 1) // 2))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, other_lim))))
    c = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    nocc = unifint(diff_lb, diff_ub, (2, (h * w) // 2))
    subs = random.sample(inds, nocc)
    numa = random.randint(1, nocc - 1)
    A = random.sample(subs, numa)
    Aset = set(A)
    B = [x for x in subs if x not in Aset]
    topg = c.copy()
    for (i, j) in A:
        topg[i, j] = topc
    botg = c.copy()
    for (i, j) in B:
        botg[i, j] = botc
    go = topg.copy()
    for (i, j) in B:
        go[i, j] = botc
    br = np.full((1, w), linc, dtype=int)
    gi = np.vstack([topg, br, botg])
    if mirror:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # separator: the uniform middle row (stacked vertically) or middle column
    row_split = (h % 2 == 1 and len(set(I[h // 2].tolist())) == 1
                 and I[h // 2, 0] != bgc)

    if row_split:
        hh = h // 2
        ho, wo = hh, w
        r0, c0 = hh + 1, 0          # second half origin
        sh, sw = hh, w
        shift = hh + 1
        move_op = 20                # up
    else:
        ww = w // 2
        ho, wo = h, ww
        r0, c0 = 0, ww + 1
        sh, sw = h, ww
        shift = ww + 1
        move_op = 23                # left

    ops, sels = [], []

    if bgc == 0:
        # Overlay reading: the second half is laid over the first half as one
        # layer; its background (0) is transparent to Paste.
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])   # keep first half (full rectangle)
        ops.append(28); sels.append([r0, c0, sh - 1, sw - 1])  # copy the whole second half from input
        ops.append(30); sels.append([0, 0, 0, 0])              # lay it over the first half
    else:
        # second half's foreground layer: every non-background cell of that half
        layer = [(r, c) for r in range(r0, r0 + sh) for c in range(c0, c0 + sw)
                 if I[r, c] != bgc]
        layer_cols = set(int(I[r, c]) for r, c in layer)
        if 0 not in layer_cols:
            # slide the whole layer across the separator onto the first half
            ops.append(move_op); sels.append(sel_of(layer))
            for _ in range(shift - 1):
                ops.append(move_op); sels.append(sel_of([]))
            # vacated footprint lies entirely in the half that gets cropped away
            ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # keep first half (full rectangle)
        else:
            # layer colour 0 cannot be moved/pasted (0 = transparent): paint it
            ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # keep first half (full rectangle)
            by_col = {}
            for r, c in layer:
                by_col.setdefault(int(I[r, c]), []).append((r - r0, c - c0))
            for col, cells in by_col.items():
                ops.append(col); sels.append(sel_of(cells))

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
