"""
ARC Task: 7c008303 (RE-ARC) — LLM-generated grid_maker
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


ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    fgc = random.choice([c for c in cols if c not in (bgc, linc)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]

    cols = interval(0, 10, 1)
    bound = min(max_h, max_w)
    lim = max(2, min(13, (bound - 3) // 2))
    h = unifint(diff_lb, diff_ub, (2, lim))
    w = unifint(diff_lb, diff_ub, (2, lim))
    h = h * 2
    w = w * 2

    remcols = [c for c in cols if c not in (bgc, linc, fgc)]
    fremcols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 4)))
    qc = [random.choice(fremcols) for _ in range(4)]

    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = random.sample(list(inds), nc)
    go = fill(c, fgc, frozenset(cels))

    gi = canvas(bgc, (h + 3, w + 3))
    gi = paint(gi, shift(asobject(go), (3, 3)))
    gi = fill(gi, linc, connect((2, 0), (2, w + 2)))
    gi = fill(gi, linc, connect((0, 2), (h + 2, 2)))
    gi = fill(gi, qc[0], {(0, 0)})
    gi = fill(gi, qc[1], {(0, 1)})
    gi = fill(gi, qc[2], {(1, 0)})
    gi = fill(gi, qc[3], {(1, 1)})

    A = lefthalf(tophalf(go))
    B = righthalf(tophalf(go))
    C = lefthalf(bottomhalf(go))
    D = righthalf(bottomhalf(go))
    A2 = replace(A, fgc, qc[0])
    B2 = replace(B, fgc, qc[1])
    C2 = replace(C, fgc, qc[2])
    D2 = replace(D, fgc, qc[3])
    go = vconcat(hconcat(A2, B2), hconcat(C2, D2))

    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # The two full separator lines: exactly one uniform row and one uniform column.
    R = next(r for r in range(hi) if len(set(I[r, :].tolist())) == 1)
    C = next(c for c in range(wi) if len(set(I[:, c].tolist())) == 1)

    r_ranges = [(0, R), (R + 1, hi)]
    c_ranges = [(0, C), (C + 1, wi)]

    # Pattern block = largest of the four quadrants; key 2x2 block = the opposite one.
    best = None
    for i in range(2):
        for j in range(2):
            area = (r_ranges[i][1] - r_ranges[i][0]) * (c_ranges[j][1] - c_ranges[j][0])
            if best is None or area > best[0]:
                best = (area, i, j)
    _, pi, pj = best
    pr0, pr1 = r_ranges[pi]
    pc0, pc1 = c_ranges[pj]
    kr0, _ = r_ranges[1 - pi]
    kc0, _ = c_ranges[1 - pj]
    ph, pw = pr1 - pr0, pc1 - pc0

    # background = fill of one of the two "empty" quadrants
    bgc = int(I[r_ranges[1 - pi][0], c_ranges[pj][0]])

    key = [[int(I[kr0 + a, kc0 + b]) for b in range(2)] for a in range(2)]
    hh, hw = ph // 2, pw // 2

    # 1. Crop the canvas down to the pattern block.
    ops.append(33); sels.append([pr0, pc0, ph - 1, pw - 1])

    # 2. Recolor the pattern's foreground cells, one quadrant color at a time.
    by_color = {}
    for a in range(2):
        for b in range(2):
            tgt = key[a][b]
            cells = []
            for r in range(a * hh, a * hh + hh):
                for c in range(b * hw, b * hw + hw):
                    if int(I[pr0 + r, pc0 + c]) != bgc:
                        cells.append((r, c))
            if cells:
                by_color.setdefault(tgt, []).extend(cells)
    for tgt, cells in by_color.items():
        ops.append(int(tgt)); sels.append(sel_of(cells))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 7c008303"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7c008303"
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
                                f"for task 7c008303"
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
                    f"Failed to build a complete episode for task 7c008303 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7c008303-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
