"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, mirrored=None, **kw) -> dict:
    from random import choice, randint
    if mirrored is None:
        mirrored = random.choice((True, False))
    # stacked-stripe dimension bounded by R, stripe length by L
    R = max_w if mirrored else max_h
    L = max_h if mirrored else max_w
    R = max(3, R)
    def unif(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))
    h = unif(2, min(15, R - 1))
    w = unif(1, L)
    cols = []
    lastc = -1
    for _ in range(h):
        c = choice([x for x in range(10) if x != lastc])
        cols.append(c)
        lastc = c
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = unif(1, R - h)
    for _ in range(numinserts):
        loc = randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if mirrored:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # rows constant (first row single-valued) -> stripes stacked vertically
    row_mode = len(set(I[0].tolist())) == 1
    G = I.copy() if row_mode else I.T.copy()   # work in "row" frame
    H, W = G.shape
    I_frame = G.copy()
    move_op = 20 if row_mode else 23           # collapse toward top / left
    def cell(r, c):
        return (r, c) if row_mode else (c, r)
    # runs of identical consecutive stripes
    runs = []
    s = 0
    for r in range(1, H + 1):
        if r == H or G[r, 0] != G[s, 0]:
            runs.append((s, r - 1))
            s = r
    n = len(runs)
    ops, sels = [], []
    cur_len = H   # rows of valid content currently at top
    # process from the bottom run upward so upper indices stay valid;
    # slide everything below a repeated run up over its duplicates
    for (a, b) in reversed(runs):
        k = b - a
        below = cur_len - (b + 1)
        if k == 0:
            continue
        if below > 0:
            block = [(r, c) for r in range(b + 1, cur_len) for c in range(W)]
            vals = {p: G[p] for p in block}
            snap = G.copy()
            for p in block:
                snap[p] = 0
            nz = [p for p in block if vals[p] != 0]
            if nz:
                prev = G.copy()
                for step in range(1, k + 1):
                    g = snap.copy()
                    for (r, c) in nz:
                        g[r - step, c] = vals[(r, c)]
                    if np.array_equal(g, prev):
                        break
                    ops.append(move_op)
                    # first step grabs the block, later steps continue the same grab
                    sels.append(sel_of([cell(*p) for p in block]) if step == 1 else sel_of([]))
                    prev = g
                G = prev
        cur_len -= k
    # 0-coloured stripes are transparent to Move, so their final slots still show
    # leftover colour: repaint each zero stripe's slot (stripe order read from I)
    fix = [(j, c) for j, (a, b) in enumerate(runs) if I_frame[a, 0] == 0
           for c in range(W) if G[j, c] != 0]
    if fix:
        ops.append(0); sels.append(sel_of([cell(*p) for p in fix]))
    # crop to one cell per stripe
    if row_mode:
        ops.append(33); sels.append([0, 0, n - 1, 0])
        ops.append(34); sels.append([0, 0, n - 1, 0])
    else:
        ops.append(33); sels.append([0, 0, 0, n - 1])
        ops.append(34); sels.append([0, 0, 0, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
