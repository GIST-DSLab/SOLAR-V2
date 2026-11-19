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
from maker.sel_helpers import sel_of  # noqa: F401  (selection here is a full rectangle bbox)


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, linec = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "linec": linec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, linec=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or noisec is None or linec is None:
        bgc, noisec, linec = random.sample(range(10), 3)
    flip = random.choice((True, False))
    # if transposed, h becomes width; keep within bounds either way
    lim_h = min(max_h, max_w) if True else max_h
    lim_w = lim_h
    h = unifint(diff_lb, diff_ub, (5, max(5, lim_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, lim_w)))
    gi = np.full((h, w), bgc, dtype=int)
    oh = unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * h))))
    oh = min(oh, h)
    ow = unifint(diff_lb, diff_ub, (4, max(4, int((2 / 3) * w))))
    ow = min(ow, w)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, nnoise):
        gi[i, j] = noisec
    gi[loci:loci + oh, locj] = linec
    gi[loci:loci + oh, locj + ow - 1] = linec
    for (i, j) in [(loci, locj), (loci + oh - 1, locj),
                   (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)]:
        gi[i, j] = noisec
    go = gi[loci:loci + oh, locj:locj + ow].copy()
    if flip:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
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

    # Line colour convention from demonstrations (side colour of the frame).
    hint = None
    if examples:
        try:
            eo = np.asarray(examples[0][1], dtype=int)
            hint = int(eo[1, 0]) if eo[1, 0] == eo[1, -1] else int(eo[0, 1])
        except Exception:
            hint = None

    candidates = []
    for col in np.unique(I):
        col = int(col)
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        if len(cells) < 4:
            continue
        comps = components(cells)
        if len(comps) != 2:
            continue
        a, b = comps
        if len(a) != len(b):
            continue
        if all(len({c for _, c in cp}) == 1 for cp in comps):
            ra = sorted(r for r, _ in a); rb = sorted(r for r, _ in b)
            if ra != rb or ra != list(range(ra[0], ra[-1] + 1)):
                continue
            orient = "v"
        elif all(len({r for r, _ in cp}) == 1 for cp in comps):
            ca = sorted(c for _, c in a); cb = sorted(c for _, c in b)
            if ca != cb or ca != list(range(ca[0], ca[-1] + 1)):
                continue
            orient = "h"
        else:
            continue
        rows = [r for cp in comps for r, _ in cp]
        cols = [c for cp in comps for _, c in cp]
        rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)
        if orient == "v":
            r0, r1, c0, c1 = rmin - 1, rmax + 1, cmin, cmax
        else:
            r0, r1, c0, c1 = rmin, rmax, cmin - 1, cmax + 1
        if r0 < 0 or c0 < 0 or r1 >= hi or c1 >= wi:
            continue
        crn = {I[r0, c0], I[r0, c1], I[r1, c0], I[r1, c1]}
        score = (col == hint, len(crn) == 1, len(a))
        candidates.append((score, (r0, c0, r1, c1)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    r0, c0, r1, c1 = candidates[0][1]
    ho2, wo2 = r1 - r0 + 1, c1 - c0 + 1

    # The frame (two parallel lines + the four corner cells one step past their
    # ends) bounds a full rectangle; crop the canvas to exactly that rectangle.
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
