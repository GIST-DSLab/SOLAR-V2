"""
ARC Task: 3f7978a0 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, linec = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS] + [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "noisec": noisec, "linec": linec, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, linec, transpose=None, **kw):
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    if transpose is None:
        transpose = random.choice([True, False])
    HM, WM = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (5, max(5, HM)))
    w = unifint(diff_lb, diff_ub, (5, max(5, WM)))
    g = np.full((h, w), bgc, dtype=int)
    oh = min(unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * h)))), h)
    ow = min(unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * w)))), w)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, nnoise):
        g[i, j] = noisec
    g[loci:loci + oh, locj] = linec
    g[loci:loci + oh, locj + ow - 1] = linec
    for (i, j) in [(loci, locj), (loci + oh - 1, locj),
                   (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)]:
        g[i, j] = noisec
    go = g[loci:loci + oh, locj:locj + ow].copy()
    if transpose:
        g = g.T.copy()
        go = go.T.copy()
    return {"input": g.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(cells):
        cells = set(cells)
        seen = set()
        comps = []
        for cell in sorted(cells):
            if cell in seen:
                continue
            stack = [cell]
            seen.add(cell)
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            comps.append(comp)
        return comps

    # The frame's two sides: the colour forming exactly two parallel straight
    # segments. Detected from I alone.
    best, best_score = None, -1
    for col in np.unique(I):
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
        if len(cells) < 4:
            continue
        comps = components(cells)
        if len(comps) != 2:
            continue
        if all(len({c for _, c in cp}) == 1 and len(cp) >= 2 for cp in comps):
            orient = "v"
        elif all(len({r for r, _ in cp}) == 1 and len(cp) >= 2 for cp in comps):
            orient = "h"
        else:
            continue
        score = min(len(cp) for cp in comps)
        if score > best_score:
            best_score, best = score, (orient, comps)
    orient, comps = best
    rows = [r for cp in comps for r, _ in cp]
    cols = [c for cp in comps for _, c in cp]
    if orient == "v":
        # corners cap each vertical side one row beyond its ends
        r0, r1, c0, c1 = min(rows) - 1, max(rows) + 1, min(cols), max(cols)
    else:
        # corners cap each horizontal side one column beyond its ends
        r0, r1, c0, c1 = min(rows), max(rows), min(cols) - 1, max(cols) + 1

    # Reading: trim the outside away axis by axis.
    # 1) Trim along the sides' own direction to the span of the sides plus
    #    their corner caps (a full-width / full-height band -- a whole rectangle).
    # 2) Trim across, keeping only what lies between the two sides.
    # A trim that would keep the full extent changes nothing and is skipped.
    ops, sels = [], []
    if orient == "v":
        if (r0, r1) != (0, hi - 1):
            ops.append(33); sels.append([r0, 0, r1 - r0, wi - 1])   # row band, full width
        cur_h = r1 - r0 + 1
        if (c0, c1) != (0, wi - 1):
            ops.append(33); sels.append([0, c0, cur_h - 1, c1 - c0])  # columns between sides
    else:
        if (c0, c1) != (0, wi - 1):
            ops.append(33); sels.append([0, c0, hi - 1, c1 - c0])   # column band, full height
        cur_w = c1 - c0 + 1
        if (r0, r1) != (0, hi - 1):
            ops.append(33); sels.append([r0, 0, r1 - r0, cur_w - 1])  # rows between sides
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
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
                        f"num_examples+1 ({num_examples + 1}) for task 3f7978a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3f7978a0"
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
                                f"for task 3f7978a0"
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
                    f"Failed to build a complete episode for task 3f7978a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3f7978a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
