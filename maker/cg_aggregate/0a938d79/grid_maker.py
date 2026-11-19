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

ROTS = ["rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, cola, colb = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cola": cola, "colb": colb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb, rot=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTS)
    # final grid: rot180 keeps (h, w); rot270 gives (w, h)
    if rot == "rot180":
        hcap, wcap = max_h, max_w
    else:
        hcap, wcap = max_w, max_h
    hcap = min(hcap, 29)
    wcap = min(wcap, 30)
    h = unifint(diff_lb, diff_ub, (4, max(4, min(29, wcap - 1, hcap))))
    w = unifint(diff_lb, diff_ub, (h + 1, max(h + 1, wcap)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    locja = unifint(diff_lb, diff_ub, (3, w - 2))
    locjb = unifint(diff_lb, diff_ub, (1, locja - 2))
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
        gi, go = np.rot90(gi, 2), np.rot90(go, 2)
    else:
        gi, go = np.rot90(gi, 1), np.rot90(go, 1)  # rot270 (CCW)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: canvas colour (only two marker cells exist)
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[int(np.argmax(counts))])
    marks = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    landscape = hi < wi
    if landscape:
        marks.sort(key=lambda t: t[1])
        p1, v1 = marks[0][1], marks[0][2]
        p2, v2 = marks[1][1], marks[1][2]
        length = wi

        def line(p):            # full column p
            return [0, p, hi - 1, 0]

        def block(a, b):        # full columns a..b (whole rectangle, bg included)
            return [0, a, hi - 1, b - a]

        def anchor(p):
            return [0, p, 0, 0]
    else:
        marks.sort(key=lambda t: t[0])
        p1, v1 = marks[0][0], marks[0][2]
        p2, v2 = marks[1][0], marks[1][2]
        length = hi

        def line(p):            # full row p
            return [p, 0, 0, wi - 1]

        def block(a, b):        # full rows a..b (whole rectangle, bg included)
            return [a, 0, b - a, wi - 1]

        def anchor(p):
            return [p, 0, 0, 0]

    g = p2 - p1
    period = 2 * g
    ops, sels = [], []

    # 1. each marker grows into a full line across the grid
    ops.append(v1); sels.append(line(p1))
    ops.append(v2); sels.append(line(p2))

    # 2. the pair of lines (lines p1..p2) is ONE periodic unit; repeat it
    #    every 2*gap further along the axis by copying it off the working grid.
    clip = None  # 'full' or 'first'
    k = 1
    while p1 + k * period < length:
        o = k * period
        q1, q2 = p1 + o, p2 + o
        full = q2 < length
        want = "full" if full else "first"
        # nonzero content in the unit (0 is invisible to Copy/Paste)
        paste_has = (v1 != 0) or (full and v2 != 0)
        if paste_has:
            if clip != want:
                ops.append(29)
                sels.append(block(p1, p2) if full else line(p1))
                clip = want
            ops.append(30); sels.append(anchor(q1))
        # a 0-coloured line cannot travel through the clipboard: draw it directly
        if v1 == 0:
            ops.append(0); sels.append(line(q1))
        if full and v2 == 0:
            ops.append(0); sels.append(line(q2))
        k += 1

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
