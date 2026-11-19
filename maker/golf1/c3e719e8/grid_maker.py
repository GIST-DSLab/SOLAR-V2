"""
ARC Task: c3e719e8 (RE-ARC) — LLM-generated grid_maker
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
import random, math
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on which colour is most common in I (no fixed colour role);
    # the output background 0 is hardcoded by the task.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(1, 10))
    hmax = max(2, min(5, math.isqrt(max_h)))
    wmax = max(2, min(5, math.isqrt(max_w)))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = random.randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = [(i, j) for i in range(h) for j in range(w)]
    mc = random.choice(cols)
    remcols = [c for c in cols if c != mc]
    gi = [[-1] * w for _ in range(h)]
    mcc = random.sample(inds, nmc)
    inds = [x for x in inds if x not in mcc]
    for i, j in mcc:
        gi[i][j] = mc
    ocols = random.sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = max(min(nmc - 1, len(inds), k), 1)
        locs = random.sample(inds, unifint(diff_lb, diff_ub, (1, ub)))
        inds = [x for x in inds if x not in locs]
        for i, j in locs:
            gi[i][j] = ocol
    gi = [[mc if v == -1 else v for v in row] for row in gi]
    go = [[0] * (w * w) for _ in range(h * h)]
    for a in range(h):
        for b in range(w):
            if gi[a][b] == mc:
                for i in range(h):
                    for j in range(w):
                        go[a * h + i][b * w + j] = gi[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi
    # Rule: block (r,c) of the (hi*hi, wi*wi) canvas gets a copy of I iff
    # I[r,c] is I's most common colour; other blocks stay 0.
    mc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pats = [frozenset(c for c in range(wi) if I[r, c] == mc) for r in range(hi)]

    ops, sels = [], []
    # Copy the whole input (full rectangle) to the clipboard.
    ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    # Expand canvas; I stays in block (0,0), the rest is 0.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    if 0 not in pats[0]:
        # Block (0,0) must be empty: clear that full block rectangle.
        ops.append(0); sels.append([0, 0, hi - 1, wi - 1])

    clip = 'I'
    built = []
    for r in range(hi):
        need = set(pats[r])
        if r == 0:
            need.discard(0)
        best = ('ind', None, (0 if clip == 'I' else 1) + len(need))
        if r > 0 and need:
            # A finished block-row whose block pattern is contained in this row's
            # pattern can be copied wholesale (larger stroke).
            for p in built:
                if pats[p] and pats[p] <= pats[r]:
                    extra = len(pats[r] - pats[p])
                    cost = (1 if clip == p else 2) + (0 if extra == 0 else 1 + extra)
                    if cost < best[2]:
                        best = ('strip', p, cost)
        if best[0] == 'strip':
            p = best[1]
            if clip != p:
                # CopyO the whole finished block-row p (full rectangle strip).
                ops.append(29); sels.append([p * hi, 0, hi - 1, wo - 1])
                clip = p
            ops.append(30); sels.append([r * hi, 0, 0, 0])
            need = set(pats[r] - pats[p])
        if need:
            if clip != 'I':
                ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
                clip = 'I'
            for c in sorted(need):
                ops.append(30); sels.append([r * hi, c * wi, 0, 0])
        built.append(r)

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
                        f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
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
                                f"for task c3e719e8"
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
                    f"Failed to build a complete episode for task c3e719e8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3e719e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
