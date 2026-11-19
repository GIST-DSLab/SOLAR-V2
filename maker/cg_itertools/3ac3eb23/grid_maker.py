"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is a fixed colour role. Seed colours are read from I.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    remcols = [c for c in cols if c != bgc]
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for _ in range(nlocs):
        if len(locopts) == 0:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if x not in range(locj - 2, locj + 3)]
        col = random.choice(remcols)
        gi[0][locj] = col
        for p in range(0, h, 2):
            go[p][locj] = col
        for p in range(1, h, 2):
            go[p][locj - 1] = col
            go[p][locj + 1] = col
    gi = np.array(gi)
    go = np.array(go)
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    if gi.shape[0] > max_h or gi.shape[1] > max_w:
        gi = np.rot90(gi, 1)
        go = np.rot90(go, 1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: the border line that holds the seeds plus the line right next to it
    form one 2-line BAND. That band is the repeating unit: first grow every
    seed's two wings into the neighbouring line (which completes the band), then
    copy the whole band and stamp it down the grid every 2 lines. If the depth is
    odd, one seed-line is left at the far border; paint each seed's lane there.
    A seed of colour 0 can't be copied or pasted, so its fan gets Color0 directly.
    O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    if all(r == 0 for r, _ in seeds):
        u, v, D = (1, 0), (0, 1), hi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D = (-1, 0), (0, 1), hi
    elif all(c == 0 for _, c in seeds):
        u, v, D = (0, 1), (1, 0), wi
    else:
        u, v, D = (0, -1), (1, 0), wi

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    ops, sels = [], []

    # 1. Grow each seed's wings into the next line. This completes the 2-line band.
    for s in seeds:
        col = int(I[s])
        ops.append(col)
        sels.append(sel_of([cell(s, 1, -1), cell(s, 1, 1)]))

    # band rectangle = depth 0..1 across the whole border (full rectangle)
    def band_origin(d0):
        # top-left of the 2-line strip that covers depths d0 and d0+1
        cells = []
        for d in (d0, d0 + 1):
            if u[0] != 0:
                r = d if u[0] > 0 else hi - 1 - d
                cells.append((r, 0))
            else:
                c = d if u[1] > 0 else wi - 1 - d
                cells.append((0, c))
        return min(p[0] for p in cells), min(p[1] for p in cells)

    P = D // 2
    nonzero_seed = any(int(I[s]) != 0 for s in seeds)
    if P > 1 and nonzero_seed:
        r0, c0 = band_origin(0)
        if u[0] != 0:
            band = [r0, c0, 1, wi - 1]   # full 2-row band rectangle
        else:
            band = [r0, c0, hi - 1, 1]   # full 2-column band rectangle
        ops.append(29)
        sels.append(band)
        for k in range(1, P):
            pr, pc = band_origin(2 * k)
            ops.append(30)
            sels.append([pr, pc, 0, 0])

    # 2. Odd depth: one seed-line is left at the far border.
    if D % 2 == 1 and D > 1:
        bycol = {}
        for s in seeds:
            col = int(I[s])
            if col == 0:
                continue
            bycol.setdefault(col, []).append(cell(s, D - 1, 0))
        for col, cells in bycol.items():
            ops.append(col)
            sels.append(sel_of(cells))

    # 3. Colour-0 seeds are invisible to Copy/Paste, so paint the rest of their fan directly.
    for s in seeds:
        if int(I[s]) != 0:
            continue
        fan = []
        for d in range(2, D):
            if d % 2 == 0:
                fan.append(cell(s, d, 0))
            else:
                fan.append(cell(s, d, -1))
                fan.append(cell(s, d, 1))
        if fan:
            ops.append(0)
            sels.append(sel_of(fan))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
