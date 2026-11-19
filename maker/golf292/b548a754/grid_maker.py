"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the whole scene


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round(diff_lb * (b - a)))
    hi = a + int(round(diff_ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, inc, dotc = random.sample(range(10), 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, inc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    lim = min(max_h, max_w) if rot in (1, 3) else None
    mh = lim if lim is not None else max_h
    mw = lim if lim is not None else max_w
    h = _unifint(diff_lb, diff_ub, (5, max(5, mh)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, mw)))
    hi = _unifint(diff_lb, diff_ub, (4, h - 1))
    wi = _unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)

    def draw(g, r0, c0, r1, c1):
        g[r0:r1 + 1, c0:c1 + 1] = boxc
        g[r0 + 1:r1, c0 + 1:c1] = inc

    go = np.full((h, w), bgc, dtype=int)
    draw(go, loci, locj, loci + hi - 1, locj + wi - 1)
    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = np.full((h, w), bgc, dtype=int)
    draw(gi, loci, locj, cutoff, locj + wi - 1)
    cj = random.randint(locj, locj + wi - 1)
    gi[loci + hi - 1, cj] = dotc
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi_, wi_ = I.shape

    # the box: the colour whose cells form exactly a rectangle outline (>=3x3)
    boxc = None
    for cand in np.unique(I).tolist():
        cells = np.argwhere(I == cand)
        a0, b0 = cells.min(0)
        a1, b1 = cells.max(0)
        bh, bw = a1 - a0 + 1, b1 - b0 + 1
        if bh < 3 or bw < 3 or len(cells) != 2 * bh + 2 * bw - 4:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        if (np.all(sub[0] == cand) and np.all(sub[-1] == cand)
                and np.all(sub[:, 0] == cand) and np.all(sub[:, -1] == cand)):
            inner = sub[1:-1, 1:-1]
            if len(np.unique(inner)) == 1 and int(inner[0, 0]) != cand:
                boxc = cand
                r0, c0, r1, c1 = int(a0), int(b0), int(a1), int(b1)
                break
    inc = int(I[r0 + 1, c0 + 1])

    # the dot: the lone cell of a colour other than box/inside
    rd = cd = None
    for cand in np.unique(I).tolist():
        if cand in (boxc, inc):
            continue
        cells = np.argwhere(I == cand)
        if len(cells) == 1:
            rd, cd = int(cells[0][0]), int(cells[0][1])
            break

    def rect(a0, b0, a1, b1):
        return sel_of([(r, c) for r in range(a0, a1 + 1) for c in range(b0, b1 + 1)])

    ops, sels = [], []
    # the box stretches toward the dot: paint the extension strip (from just past the
    # facing wall to the dot's line) in the wall colour, then hollow it with the fill
    # colour, which also melts the old wall into the stretched body.
    if cd > c1:      # grow right
        ops.append(boxc); sels.append(rect(r0, c1 + 1, r1, cd))
        ops.append(inc);  sels.append(rect(r0 + 1, c1, r1 - 1, cd - 1))
    elif cd < c0:    # grow left
        ops.append(boxc); sels.append(rect(r0, cd, r1, c0 - 1))
        ops.append(inc);  sels.append(rect(r0 + 1, cd + 1, r1 - 1, c0))
    elif rd > r1:    # grow down
        ops.append(boxc); sels.append(rect(r1 + 1, c0, rd, c1))
        ops.append(inc);  sels.append(rect(r1, c0 + 1, rd - 1, c1 - 1))
    else:            # grow up
        ops.append(boxc); sels.append(rect(rd, c0, r0 - 1, c1))
        ops.append(inc);  sels.append(rect(rd + 1, c0 + 1, r0, c1 - 1))

    ops.append(34); sels.append([0, 0, hi_ - 1, wi_ - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
