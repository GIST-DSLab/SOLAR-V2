"""
ARC Task: b7249182 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, ca, cb = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ca=1, cb=2, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice((True, False))

    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))

    # if the grid will be transposed, its height budget is max_w and vice versa
    mh, mw = (max_w, max_h) if mirrored else (max_h, max_w)
    h = unifint(7, max(7, min(30, mh)))
    w = unifint(5, max(5, min(30, mw)))
    ih = unifint(3, (h - 1) // 2)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - 2 * ih)
    locj = random.randint(0, w - 5)
    f = locj + 2
    for r in range(loci, loci + ih - 1):
        go[r][f] = ca
    for c in range(locj, locj + 5):
        go[loci + ih - 2][c] = ca
        go[loci + ih + 1][c] = cb
    go[loci + ih - 1][locj] = ca
    go[loci + ih - 1][locj + 4] = ca
    go[loci + ih][locj] = cb
    go[loci + ih][locj + 4] = cb
    for r in range(loci + ih + 2, loci + 2 * ih):
        go[r][f] = cb
    gi[loci][f] = ca
    gi[loci + 2 * ih - 1][f] = cb
    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Background = the canvas colour; inputs contain only two marker pixels on it.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    (ra, ka), (rb, kb) = marks[0], marks[1]
    vertical = (ka == kb)
    if vertical:
        fixed = ka
        p_lo, p_hi = sorted((ra, rb))
    else:
        fixed = ra
        p_lo, p_hi = sorted((ka, kb))

    def cell(p, s):
        return (p, s) if vertical else (s, p)

    c_lo = int(I[cell(p_lo, fixed)])
    c_hi = int(I[cell(p_hi, fixed)])
    ih = (p_hi - p_lo + 1) // 2
    mid_lo, mid_hi = p_lo + ih - 1, p_lo + ih   # the two meeting-point cells

    G = I.copy()
    ops, sels = [], []

    def paint(cells, color):
        # skip only cells that ALREADY hold this colour in the current grid
        todo = sorted({c for c in cells if G[c] != color})
        if todo:
            ops.append(color)
            sels.append(sel_of(todo))
            for c in todo:
                G[c] = color

    # 1. connect the two markers with a line in the far marker's colour
    paint([cell(p, fixed) for p in range(p_lo + 1, p_hi)], c_hi)
    # 2. the near half of that line (up to its centre) takes the near marker's colour
    paint([cell(p, fixed) for p in range(p_lo + 1, mid_lo + 1)], c_lo)
    # 3. duplicate the two-cell centre block two cells to each side
    centre = [cell(mid_lo, fixed), cell(mid_hi, fixed)]
    if c_lo != 0 and c_hi != 0:
        r0, cc0 = cell(mid_lo, fixed)
        hh, ww = (1, 0) if vertical else (0, 1)
        # bbox is exactly the 2-cell centre block (full rectangle intended)
        ops.append(29); sels.append([r0, cc0, hh, ww])          # CopyO centre block
        for d in (-2, 2):
            tr, tc = cell(mid_lo, fixed + d)
            ops.append(30); sels.append([tr, tc, 0, 0])          # Paste beside it
            G[cell(mid_lo, fixed + d)] = c_lo
            G[cell(mid_hi, fixed + d)] = c_hi
    else:
        # a 0-coloured marker would be dropped by Copy/Paste: paint the copies directly
        paint([cell(mid_lo, fixed - 2), cell(mid_lo, fixed + 2)], c_lo)
        paint([cell(mid_hi, fixed - 2), cell(mid_hi, fixed + 2)], c_hi)
    # 4. bars capping the duplicates: near-side bar in c_lo, far-side bar in c_hi
    paint([cell(mid_lo - 1, s) for s in range(fixed - 2, fixed + 3)], c_lo)
    paint([cell(mid_hi + 1, s) for s in range(fixed - 2, fixed + 3)], c_hi)
    # 5. clear the original centre block back to background
    paint(centre, bgc)

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
                        f"num_examples+1 ({num_examples + 1}) for task b7249182"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b7249182"
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
                                f"for task b7249182"
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
                    f"Failed to build a complete episode for task b7249182 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b7249182-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
