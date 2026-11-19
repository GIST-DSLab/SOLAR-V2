"""
ARC Task: 0a938d79 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"rot": "180"}, {"rot": "270"}]


def sample_colors(num_examples=None) -> dict:
    bgc, cola, colb = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cola": cola, "colb": colb, "instance_plan": plan}


def _unif(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    d = random.uniform(diff_lb, diff_ub)
    v = lb + int(round(d * (ub - lb)))
    return max(lb, min(ub, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb, rot=None) -> dict:
    if rot is None:
        rot = random.choice(VARIANTS)["rot"]
    # pre-rotation (h, w); rot270 swaps the final dimensions
    if rot == "180":
        h_lim, w_lim = max_h, max_w
    else:
        h_lim, w_lim = max_w, max_h
    h_ub = min(29, h_lim, w_lim - 1)
    h = _unif(diff_lb, diff_ub, 4, h_ub)
    w = _unif(diff_lb, diff_ub, max(h + 1, 5), min(30, w_lim))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    locja = _unif(diff_lb, diff_ub, 3, w - 2)
    locjb = _unif(diff_lb, diff_ub, 1, locja - 2)
    locia = random.choice((0, h - 1))
    locib = random.choice((0, h - 1))
    gi[locia, locja] = cola
    gi[locib, locjb] = colb
    ofs = -2 * (locja - locjb)
    for aa in range(locja, -1, ofs):
        go[:, aa] = cola
    for bb in range(locjb, -1, ofs):
        go[:, bb] = colb
    k = 2 if rot == "180" else 1   # DSL rot270 == np.rot90(k=1)
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: canvas colour (only two marker cells exist)
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[int(np.argmax(counts))])
    marks = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    ops, sels = [], []
    landscape = hi < wi

    if landscape:
        marks.sort(key=lambda t: t[1])
        p1, v1 = marks[0][1], marks[0][2]
        p2, v2 = marks[1][1], marks[1][2]
        L = wi
        line = lambda p: [(r, p) for r in range(hi)]
        block_sel = lambda p, n: [0, p, hi - 1, n - 1]     # full rectangle: one period of stripes
    else:
        marks.sort(key=lambda t: t[0])
        p1, v1 = marks[0][0], marks[0][2]
        p2, v2 = marks[1][0], marks[1][2]
        L = hi
        line = lambda p: [(p, c) for c in range(wi)]
        block_sel = lambda p, n: [p, 0, n - 1, wi - 1]     # full rectangle: one period of stripes
    paste_sel = (lambda p: [0, p, 0, 0]) if landscape else (lambda p: [p, 0, 0, 0])

    stride = 2 * (p2 - p1)

    # 1. Grow each marker into its full line -> this is one period tile
    ops.append(v1); sels.append(sel_of(line(p1)))
    ops.append(v2); sels.append(sel_of(line(p2)))

    # 2. Repeat the period tile (first line, gap, second line, gap) with that stride
    origins = []
    p = p1 + stride
    while p < L:
        origins.append(p)
        p += stride
    if origins:
        ops.append(29); sels.append(block_sel(p1, stride))   # CopyO the tile just built
        for o in origins:
            ops.append(30); sels.append(paste_sel(o))
        # 0-coloured lines are transparent to Copy/Paste: draw those copies directly
        for base, v in ((p1, v1), (p2, v2)):
            if v == 0:
                for o in origins:
                    q = base + (o - p1)
                    if q < L:
                        ops.append(0); sels.append(sel_of(line(q)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 0a938d79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0a938d79"
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
                                f"for task 0a938d79"
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
                    f"Failed to build a complete episode for task 0a938d79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0a938d79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
