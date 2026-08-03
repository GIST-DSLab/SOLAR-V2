"""
ARC Task: caa06a1f (RE-ARC) — LLM-generated grid_maker
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
import numpy as np

VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    tric = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "tric": tric, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, tric, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    # a 90/270 rotation swaps the final dims, so cap pre-rotation dims accordingly
    Hlim, Wlim = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = unifint(diff_lb, diff_ub, (min(10, Hlim), Hlim))
    w = unifint(diff_lb, diff_ub, (min(10, Wlim), Wlim))

    vp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    hp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))

    pool = [c for c in range(10) if c != bgc and c != tric]
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    ccols = random.sample(pool, numc)
    tile = [[random.choice(ccols) for _ in range(hp)] for _ in range(vp)]

    gi = [[tile[r % vp][c % hp] for c in range(w)] for r in range(h)]
    go = [[tile[r % vp][(c - 1) % hp] for c in range(w)] for r in range(h)]

    ioffs = unifint(diff_lb, diff_ub, (1, max(1, h - 2 * vp)))
    joffs = unifint(diff_lb, diff_ub, (1, max(1, w - 2 * hp)))
    for a in range(ioffs):
        for c in range(w):
            gi[a][c] = tric
    for b in range(joffs):
        for r in range(h):
            gi[r][b] = tric

    def _cw(g):
        return [list(row) for row in zip(*g[::-1])]

    for _ in range(rot):
        gi = _cw(gi)
        go = _cw(go)

    return {
        'input': tuple(tuple(row) for row in gi),
        'output': tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # The band color always owns one full edge row + one full edge column,
    # so it is the strict majority of the border ring.
    border = list(I[0, :]) + list(I[hi - 1, :]) + list(I[:, 0]) + list(I[:, wi - 1])
    tric = max(set(border), key=border.count)

    # The remaining cells form one rectangular block of periodic pattern.
    rs = [r for r in range(hi) if any(I[r, c] != tric for c in range(wi))]
    cs = [c for c in range(wi) if any(I[r, c] != tric for r in range(hi))]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    ph, pw = r1 - r0 + 1, c1 - c0 + 1

    # measure the pattern's own periods
    vp = ph
    for p in range(1, ph):
        if all(I[r, c] == I[r + p, c]
               for r in range(r0, r1 - p + 1) for c in range(c0, c1 + 1)):
            vp = p
            break
    hp = pw
    for p in range(1, pw):
        if all(I[r, c] == I[r, c + p]
               for r in range(r0, r1 + 1) for c in range(c0, c1 - p + 1)):
            hp = p
            break

    # the single pattern-bearing corner tells which way the tiling slides by 1
    dr = dc = 0
    if I[0, 0] != tric:
        dc -= 1
    if I[0, wi - 1] != tric:
        dr -= 1
    if I[hi - 1, wi - 1] != tric:
        dc += 1
    if I[hi - 1, 0] != tric:
        dr += 1

    # pick a whole number of periods out of the pattern block whose phase lands
    # exactly on the grid origin once slid by (dr, dc)
    a = r0 + ((-dr - r0) % vp)
    b = c0 + ((-dc - c0) % hp)
    Rh = vp * ((r1 - a + 1) // vp)
    Rw = hp * ((c1 - b + 1) // hp)

    # grab the block from the input before anything touches the canvas
    ops.append(28)
    sels.append([a, b, Rh - 1, Rw - 1])

    # Paste never writes 0 cells. If 0 is one of the pattern's colors, clear the
    # canvas so those cells are already 0 when the tiling lands on them.
    has_zero = bool((I[a:a + Rh, b:b + Rw] == 0).any())
    if has_zero:
        ops.append(0)
        sels.append([0, 0, hi - 1, wi - 1])

    # tile the block across the whole grid
    for r in range(0, hi, Rh):
        for c in range(0, wi, Rw):
            ops.append(30)
            sels.append([r, c, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
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
                                f"for task caa06a1f"
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
                    f"Failed to build a complete episode for task caa06a1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"caa06a1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
