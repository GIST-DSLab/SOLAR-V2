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

ROT_VARIANTS = [{"rot": "rot180"}, {"rot": "rot270"}]


def sample_colors(num_examples=None) -> dict:
    bgc, cola, colb = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [dict(v) for v in ROT_VARIANTS]
        examples += [dict(random.choice(ROT_VARIANTS)) for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cola": cola, "colb": colb, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    rng = b - a
    lb = a + round(diff_lb * rng)
    ub = a + round(diff_ub * rng)
    if ub < lb:
        lb, ub = ub, lb
    return random.randint(lb, ub)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROT_VARIANTS)["rot"]
    # pre-rotation canvas bounds (rot270 swaps the dimensions)
    if rot == "rot180":
        HB, WB = min(29, max_h), min(30, max_w)
    else:
        HB, WB = min(29, max_w), min(30, max_h)
    HB = min(HB, WB - 1)
    h = _unifint(diff_lb, diff_ub, (4, max(4, HB)))
    w = _unifint(diff_lb, diff_ub, (max(h + 1, 5), max(h + 1, 5, WB)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    locja = _unifint(diff_lb, diff_ub, (3, w - 2))
    locjb = _unifint(diff_lb, diff_ub, (1, locja - 2))
    locia = random.choice((0, h - 1))
    locib = random.choice((0, h - 1))
    gi[locia, locja] = cola
    gi[locib, locjb] = colb
    ofs = -2 * (locja - locjb)
    for aa in range(locja, -1, ofs):
        go[:, aa] = cola
    for bb in range(locjb, -1, ofs):
        go[:, bb] = colb
    if rot == "rot180":
        gi = np.rot90(gi, 2)
        go = np.rot90(go, 2)
    else:  # rot270 (DSL) = rot90 CCW... DSL rot270 = clockwise 270 = np.rot90 k=1
        gi = np.rot90(gi, 1)
        go = np.rot90(go, 1)
    return {"input": tuple(tuple(int(v) for v in row) for row in gi),
            "output": tuple(tuple(int(v) for v in row) for row in go)}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour (only two marker cells exist)
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[int(np.argmax(counts))])
    marks = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    ops, sels = [], []
    portrait = hi > wi
    # Unit of work: one COLOUR at a time. Each marker colour gets a single Color op
    # whose selection is its whole family of lines (the marker's line and every
    # repeat at stride 2*gap toward the far edge) -- all nameable from I.
    if portrait:
        marks.sort(key=lambda t: t[0])
        (p1, _, v1), (p2, _, v2) = marks[0], marks[1]
        n = hi
    else:
        marks.sort(key=lambda t: t[1])
        (_, p1, v1), (_, p2, v2) = marks[0], marks[1]
        n = wi
    stride = 2 * (p2 - p1)
    for p0, v in ((p1, v1), (p2, v2)):
        cells = []
        for p in range(p0, n, stride):
            if portrait:
                cells += [(p, c) for c in range(wi)]
            else:
                cells += [(r, p) for r in range(hi)]
        ops.append(v)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid for Submit
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
