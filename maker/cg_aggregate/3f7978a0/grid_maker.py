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

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec, linec = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        examples = [dict(v) for v in ORIENTS]
        examples += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "noisec": noisec, "linec": linec, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, linec=None,
             transpose=None, instance_plan=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None or noisec is None or linec is None:
        bgc, noisec, linec = random.sample(range(10), 3)
    if transpose is None:
        transpose = random.choice([True, False])

    # if transposed, the h/w roles swap in the final grid
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (5, max(5, lim_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, lim_w)))
    gi = [[bgc] * w for _ in range(h)]
    oh = unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * h))))
    oh = min(oh, h)
    ow = unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * w))))
    ow = min(ow, w)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, nnoise):
        gi[i][j] = noisec
    for i in range(loci, loci + oh):
        gi[i][locj] = linec
        gi[i][locj + ow - 1] = linec
    for (i, j) in [(loci, locj), (loci + oh - 1, locj),
                   (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)]:
        gi[i][j] = noisec
    go = [row[locj:locj + ow] for row in gi[loci:loci + oh]]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry): the answer is the frame
    # cut out of the input -- the two parallel line segments plus the corner cell
    # beyond each end. The golf entry peels the grid one edge at a time until it
    # reaches the frame; peeling every edge in turn comes to the same thing as
    # cropping once to that rectangle, and a second crop would do nothing. So
    # this is a single CropGrid to the frame's bounding rectangle.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(cells):
        cells = set(cells)
        seen = set()
        comps = []
        for cell in cells:
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

    best = None
    best_score = -1
    for col in np.unique(I):
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        if len(cells) < 4:
            continue
        comps = components(cells)
        if len(comps) != 2:
            continue
        a, b = comps
        if len(a) != len(b) or len(a) < 2:
            continue
        ar = sorted({r for r, _ in a}); ac = sorted({c for _, c in a})
        br = sorted({r for r, _ in b}); bc = sorted({c for _, c in b})
        if len(ac) == 1 and len(bc) == 1 and ar == br and ac != bc:
            orient = "v"
        elif len(ar) == 1 and len(br) == 1 and ac == bc and ar != br:
            orient = "h"
        else:
            continue
        score = len(a)
        if score > best_score:
            best_score = score
            best = (orient, comps)

    orient, comps = best
    rows = [r for cp in comps for r, _ in cp]
    cols = [c for cp in comps for _, c in cp]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    if orient == "v":
        r0, r1, c0, c1 = rmin - 1, rmax + 1, cmin, cmax
    else:
        r0, r1, c0, c1 = rmin, rmax, cmin - 1, cmax + 1
    ho2 = r1 - r0 + 1
    wo2 = c1 - c0 + 1

    # CropGrid: the frame rectangle (full rectangle, background included)
    ops = [33, 34]
    sels = [[r0, c0, ho2 - 1, wo2 - 1], [0, 0, ho2 - 1, wo2 - 1]]
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
