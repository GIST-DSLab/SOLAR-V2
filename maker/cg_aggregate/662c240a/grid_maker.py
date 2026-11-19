"""
ARC Task: 662c240a (RE-ARC) — LLM-generated grid_maker
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

ORIENTS = ["h", "v"]


def sample_colors(num_examples=None) -> dict:
    # Colors are sampled freely per block in the original generator; the rule
    # (pick the block that is not diagonal-symmetric) is color-independent.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = list(ORIENTS) + [random.choice(ORIENTS) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ORIENTS, n_ex)
    plan = [{"orient": o} for o in ex]
    plan.append({"orient": random.choice(ex)})
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, orient=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if orient is None:
        orient = random.choice(ORIENTS)
    cols = list(range(10))
    long_max = max_w if orient == "h" else max_h
    short_max = max_h if orient == "h" else max_w
    dmax = min(7, short_max, long_max // 2)
    if dmax < 2:
        raise ValueError("grid too small")
    d = unifint(diff_lb, diff_ub, (2, dmax))
    ng = unifint(diff_lb, diff_ub, (2, long_max // d))
    nc = unifint(diff_lb, diff_ub, (2, min(9, d * d)))

    def sym_block():
        tcolset = random.sample(cols, nc)
        g = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(i, d):
                v = random.choice(tcolset)
                g[i, j] = v
                g[j, i] = v
        return g, tcolset

    while True:
        gik, tcolset = sym_block()
        npairs = d * (d - 1) // 2
        ndistinv = unifint(diff_lb, diff_ub, (0, npairs - 1))
        ndist = npairs - ndistinv
        offd = [(i, j) for i in range(d) for j in range(d) if i != j]
        distinds = random.sample(offd, ndist)
        for (i, j) in distinds:
            if gik[i, j] == gik[j, i]:
                gik[i, j] = random.choice([c for c in tcolset if c != gik[i, j]])
            else:
                gik[i, j] = gik[j, i]
        if not np.array_equal(gik, gik.T):
            break

    go = gik.copy()
    gi = gik.copy()
    ax = 1 if orient == "h" else 0
    for _ in range(ng - 1):
        b, _ = sym_block()
        if random.choice((True, False)):
            gi = np.concatenate([gi, b], axis=ax)
        else:
            gi = np.concatenate([b, gi], axis=ax)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    # Reading (shared by the maker and the golf entry): the input is a strip of
    # d x d blocks along its long axis; every block but one equals its own
    # transpose. The answer is that one asymmetric block -> crop to it.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    d = min(hi, wi)
    if hi >= wi:
        origins = [(k * d, 0) for k in range(hi // d)]
    else:
        origins = [(0, k * d) for k in range(wi // d)]
    r0, c0 = origins[0]
    for (br, bc) in origins:
        blk = I[br:br + d, bc:bc + d]
        if not np.array_equal(blk, blk.T):
            r0, c0 = br, bc
            break
    ops, sels = [], []
    # Crop the canvas to the full d x d block rectangle (whole region intended).
    ops.append(33); sels.append([r0, c0, d - 1, d - 1])
    ops.append(34); sels.append([0, 0, d - 1, d - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 662c240a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 662c240a"
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
                                f"for task 662c240a"
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
                    f"Failed to build a complete episode for task 662c240a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"662c240a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
