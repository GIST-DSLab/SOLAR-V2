"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = int(round(lo + (hi - lo) * random.random()))
    return max(a, min(b, v))


def sample_colors(num_examples=None) -> dict:
    # Stripe colours are sampled per instance in the original generator and the rule
    # (per-band majority colour) does not depend on them. The only discrete structural
    # variant is orientation (horizontal vs. vertical stripes) -> instance plan.
    variants = [{"vertical": False}, {"vertical": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, vertical=None, **kwargs) -> dict:
    if vertical is None:
        vertical = random.choice([True, False])
    # horizontal layout: rows stack stripes (bounded by along_max), width w (bounded by cross_max)
    stack_max = max_w if vertical else max_h
    width_max = max_h if vertical else max_w
    stack_max = min(stack_max, 30)
    width_max = min(width_max, 30)
    cols = list(range(10))
    nr = _unifint(diff_lb, diff_ub, (1, max(1, min(10, stack_max // 2))))
    w = _unifint(diff_lb, diff_ub, (4, max(4, width_max)))
    scols = random.sample(cols, nr)
    sgs = [[[col] * w for _ in range(2)] for col in scols]
    numexp = _unifint(diff_lb, diff_ub, (0, max(0, stack_max - 2 * nr)))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))
    gi_rows, go_rows = [], []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = len(sg), w
        ub = (a * b) // 2 - 1
        nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = [(i, j) for i in range(a) for j in range(b)]
        noise = random.sample(inds, nnoise)
        oc = [c for c in cols if c != col]
        sg2 = [list(r) for r in sg]
        for (i, j) in noise:
            sg2[i][j] = random.choice(oc)
        for ridx in [0, a - 1]:
            while sum(1 for e in sg2[ridx] if e == col) < w // 2:
                locs = [j for j, e in enumerate(sg2[ridx]) if e != col]
                sg2[ridx][random.choice(locs)] = col
        gi_rows += sg2
        go_rows += [list(r) for r in sg]
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if vertical:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _mode(seq):
    return Counter(list(seq)).most_common(1)[0][0]


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    row_modes = [_mode(I[r, :].tolist()) for r in range(hi)]
    col_modes = [_mode(I[:, c].tolist()) for c in range(wi)]
    vertical = len(set(col_modes)) > len(set(row_modes))

    modes = col_modes if vertical else row_modes
    n = len(modes)
    # group consecutive lines with the same majority colour into bands
    bands = []
    start = 0
    for k in range(1, n + 1):
        if k == n or modes[k] != modes[start]:
            bands.append((start, k, modes[start]))
            start = k

    ops, sels = [], []
    for (s, e, col) in bands:
        if vertical:
            cells = [(r, c) for r in range(hi) for c in range(s, e)]
        else:
            cells = [(r, c) for r in range(s, e) for c in range(wi)]
        # skip a band that is already clean (op would have no visible effect)
        if all(I[r, c] == col for r, c in cells):
            continue
        # paint the whole stripe band with its majority colour
        ops.append(int(col))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
