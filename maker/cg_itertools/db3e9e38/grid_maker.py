"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # dimensions before rotation; rot 1/3 swap h and w
    hmax, wmax = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (3, min(30, hmax)))
    w = _unifint(diff_lb, diff_ub, (3, min(30, wmax)))
    barth = _unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = _unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc
    go = np.full((h, w), bgc, dtype=int)

    def fillbar(col, dr, dc):
        for r in range(0, loci + 1):
            for c in range(locj, locj + barth):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    go[rr, cc] = col

    for k in range(16):
        fillbar(fgc, -2 * k, 2 * k * barth)
        fillbar(fgc, -2 * k, -2 * k * barth)
        fillbar(8, -(2 * k + 1), (2 * k + 1) * barth)
        fillbar(8, -(2 * k + 1), -(2 * k + 1) * barth)
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: the bar sits on one edge of the grid. On both sides of it, bands
    # as thick as the bar step away from it. Band k is the bar pulled k cells
    # back toward its edge, so each band is one cell shorter than the last.
    # Odd k is colour 8 and even k is the bar's colour.
    # The golf entry reads the same staircase: column length = bar length minus
    # distance, with alternating colour. When barth > 1, its one-column units
    # group into exactly these barth-wide bands. The two readings agree, so
    # nothing changes from the maker: each band is painted as one whole
    # rectangle (its full set of cells), working outward from the bar.
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]  # bar is < half the grid
    mask = I != bgc
    if not mask.any():
        return [34], [[0, 0, hi - 1, wi - 1]]
    rs, cs = np.where(mask)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])
    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    else:
        edge = 'right'
    if edge in ('top', 'bottom'):
        barth = c1 - c0 + 1
        ar, ac = (-1 if edge == 'top' else 1), 0   # retreat toward the edge
        lr, lc = 0, barth                          # lateral step of one band
    else:
        barth = r1 - r0 + 1
        ar, ac = 0, (-1 if edge == 'left' else 1)
        lr, lc = barth, 0
    ops, sels = [], []
    for k in range(1, max(hi, wi) + 1):
        for sign in (1, -1):
            dr = k * ar + sign * k * lr
            dc = k * ac + sign * k * lc
            a, b = max(0, r0 + dr), min(hi - 1, r1 + dr)
            c, d = max(0, c0 + dc), min(wi - 1, c1 + dc)
            if a > b or c > d:
                continue
            band = [(r, cc) for r in range(a, b + 1) for cc in range(c, d + 1)]
            ops.append(8 if k % 2 == 1 else fgc)
            sels.append(sel_of(band))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
