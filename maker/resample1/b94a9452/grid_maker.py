"""
ARC Task: b94a9452 (RE-ARC) — LLM-generated grid_maker
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
    bgc, outer, inner = random.sample(range(10), 3)
    return {"bgc": bgc, "outer": outer, "inner": inner}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, outer=None, inner=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None or outer is None or inner is None:
        bgc, outer, inner = random.sample(range(10), 3)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = random.choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = random.choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = random.randint(loci + 1, loci + oh - oh2 - 1)
    locj2 = random.randint(locj + 1, locj + ow - ow2 - 1)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + oh):
        for c in range(locj, locj + ow):
            gi[r][c] = outer
    for r in range(loci2, loci2 + oh2):
        for c in range(locj2, locj2 + ow2):
            gi[r][c] = inner
    go = []
    for r in range(loci, loci + oh):
        row = []
        for c in range(locj, locj + ow):
            v = gi[r][c]
            row.append(inner if v == outer else outer)
        go.append(row)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = sorted(set(I.flatten().tolist()))

    def bbox(col):
        rs, cs = np.where(I == col)
        return rs.min(), rs.max(), cs.min(), cs.max()

    # background: the colour whose bbox spans the whole grid (generator guarantees
    # bgc owns at least one full row and one full column)
    areas = {}
    for col in colors:
        r0, r1, c0, c1 = bbox(col)
        areas[col] = (r1 - r0 + 1) * (c1 - c0 + 1)
    bgc = max(colors, key=lambda k: areas[k])
    rest = [k for k in colors if k != bgc]
    # outer rectangle has the larger bbox; inner sits strictly inside it
    outer = max(rest, key=lambda k: areas[k])
    inner = [k for k in rest if k != outer][0]

    r0, r1, c0, c1 = bbox(outer)
    ir0, ir1, ic0, ic1 = bbox(inner)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # 1. crop canvas to the outer rectangle (full rectangle selection)
    ops.append(33); sels.append([int(r0), int(c0), int(oh - 1), int(ow - 1)])
    # 2. recolour the outer frame (connected ring; corner (0,0) is always frame) to inner colour
    ops.append(10 + inner); sels.append([0, 0, 0, 0])
    # 3. recolour the inner rectangle (full rectangle) to the outer colour
    ops.append(outer)
    sels.append([int(ir0 - r0), int(ic0 - c0), int(ir1 - ir0), int(ic1 - ic0)])
    ops.append(34); sels.append([0, 0, int(oh - 1), int(ow - 1)])
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
                        f"num_examples+1 ({num_examples + 1}) for task b94a9452"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b94a9452"
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
                                f"for task b94a9452"
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
                    f"Failed to build a complete episode for task b94a9452 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b94a9452-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
