"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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


def _unif(lb, ub, bnd):
    a, b = bnd
    x = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * x))))


def sample_colors(num_examples=None) -> dict:
    bgc = random.randrange(10)
    barc = random.choice([c for c in range(10) if c != bgc])
    n = num_examples if num_examples else 3
    variants = [{"vertical": False}, {"vertical": True}]
    if n >= 2:
        ex = [dict(v) for v in variants]
        ex += [{"vertical": random.choice([True, False])} for _ in range(n - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barc": barc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, barc=1, vertical=None) -> dict:
    if vertical is None:
        vertical = random.choice([True, False])
    long_lim = max_h if vertical else max_w
    short_lim = max_w if vertical else max_h
    for _ in range(100):
        d = _unif(diff_lb, diff_ub, (2, 9))
        k = 4 if d < 7 else 3
        num = _unif(diff_lb, diff_ub, (2, k))
        if num * (d + 1) - 1 <= long_lim and d <= short_lim:
            break
    else:
        d, num = 2, 2
    remcols = [c for c in range(10) if c not in (bgc, barc)]
    ncols = _unif(diff_lb, diff_ub, (1, 8))
    patcols = random.sample(remcols, ncols)
    bgcanv = np.full((d, d), bgc, dtype=int)
    c = np.full((d, d), bgc, dtype=int)
    inds = [(i, j) for i in range(d) for j in range(d)]
    ncells = _unif(diff_lb, diff_ub, (1, d * d - 1))
    for (i, j) in random.sample(inds, ncells):
        c[i, j] = random.choice(patcols)
    barr = np.full((d, 1), barc, dtype=int)
    fill = random.randrange(num)
    gi = c.copy() if fill == 0 else bgcanv.copy()
    go = c.copy()
    for j in range(num - 1):
        c = np.rot90(c, 3)  # clockwise
        gi = np.hstack([gi, barr, c if j + 1 == fill else bgcanv])
        go = np.hstack([go, barr, c])
    if vertical:
        gi = np.rot90(gi, 3)
        go = np.rot90(go, 3)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # panel geometry: d x d square panels separated by 1-cell bars
    vertical = hi > wi
    d = wi if vertical else hi
    num = (max(hi, wi) + 1) // (d + 1)
    origins = [(j * (d + 1), 0) if vertical else (0, j * (d + 1)) for j in range(num)]

    # the single non-uniform panel is the pattern; uniform panels give bgc
    src, bgc = 0, 0
    for j, (r, c) in enumerate(origins):
        p = I[r:r + d, c:c + d]
        if len(np.unique(p)) > 1:
            src = j
        else:
            bgc = int(p[0, 0])
    sr, sc = origins[src]
    has_zero = bool((I[sr:sr + d, sc:sc + d] == 0).any())

    def rect(r, c):
        return [(r + i, c + j) for i in range(d) for j in range(d)]

    ops, sels = [], []

    # Unit of work = one neighbour hand-off: each new panel is copied from the
    # panel just completed beside it (current working grid, CopyO), pasted into
    # place, then turned a single quarter. Moving forward: clockwise (op25);
    # moving backward: counter-clockwise (op24).
    def step(frm, to, rot_op):
        fr, fc = origins[frm]
        tr, tc = origins[to]
        ops.append(29); sels.append(sel_of(rect(fr, fc)))      # full square panel
        if has_zero and bgc != 0:
            # paste is transparent to 0: blank the target so 0-cells come out 0
            ops.append(0); sels.append(sel_of(rect(tr, tc)))
        ops.append(30); sels.append(sel_of([(tr, tc)]))
        ops.append(rot_op); sels.append(sel_of(rect(tr, tc)))  # whole square panel

    for t in range(src + 1, num):
        step(t - 1, t, 25)
    for t in range(src - 1, -1, -1):
        step(t + 1, t, 24)

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
