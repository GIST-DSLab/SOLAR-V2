"""
ARC Task: c909285e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "boxcol": boxcol}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round(d * (b - a) * 0.0))
    v = int(round(a + (b - a) * d))
    v = max(a, min(b, v))
    return random.randint(lo, max(lo, v)) if random.random() < 0.0 else v


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxcol=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if boxcol is None or boxcol == bgc:
        boxcol = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, boxcol)]

    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    h = min(h, max_h) if max_h >= 5 else 5
    w = min(w, max_w) if max_w >= 5 else 5
    nfronts = _unifint(diff_lb, diff_ub, (1, max(1, (h + w) // 2)))

    gi = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for _ in range(nfronts):
        if not inds:
            break
        loc = random.choice(inds)
        inds.remove(loc)
        col = random.choice(remcols)
        if random.random() < 0.5:
            gi[loc[0], :] = col   # hfrontier
        else:
            gi[:, loc[1]] = col   # vfrontier

    oh = _unifint(diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
    ow = _unifint(diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
    loci = random.randint(1, h - oh - 1)
    locj = random.randint(1, w - ow - 1)
    r1, c1 = loci + oh - 1, locj + ow - 1
    gi[loci, locj:c1 + 1] = boxcol
    gi[r1, locj:c1 + 1] = boxcol
    gi[loci:r1 + 1, locj] = boxcol
    gi[loci:r1 + 1, c1] = boxcol
    go = gi[loci:r1 + 1, locj:c1 + 1].copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def _frame_colour(I):
    """The colour whose cells, taken as one unit, form exactly a rectangle
    outline (both sides > 1); smallest such outline wins.  Falls back to the
    rarest colour (the code-golf reading) if no colour qualifies."""
    best = None
    for col in np.unique(I).tolist():
        rs, cs = np.nonzero(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        outline = set()
        for c in range(c0, c1 + 1):
            outline.add((r0, c)); outline.add((r1, c))
        for r in range(r0, r1 + 1):
            outline.add((r, c0)); outline.add((r, c1))
        cells = set(zip(rs.tolist(), cs.tolist()))
        if cells != outline:
            continue
        if best is None or hh * ww < best[0]:
            best = (hh * ww, col)
    if best is not None:
        return best[1]
    vals, cnts = np.unique(I, return_counts=True)
    return int(vals[np.argmin(cnts)])


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The frame is "all cells of one colour" (the colour forming a rectangle outline).
    col = _frame_colour(I)
    rows = [r for r in range(hi) if (I[r] == col).any()]      # rows that contain the frame colour
    cols = [c for c in range(wi) if (I[:, c] == col).any()]   # columns that contain the frame colour
    r0, nr = rows[0], len(rows)
    c0, nc = cols[0], len(cols)

    ops, sels = [], []
    # 1. keep only the rows containing the frame colour (full-width band -> exact rectangle)
    if not (r0 == 0 and nr == hi):
        ops.append(33); sels.append([r0, 0, nr - 1, wi - 1])
    # 2. then keep only the columns containing the frame colour (full-height band of the new canvas)
    if not (c0 == 0 and nc == wi):
        ops.append(33); sels.append([0, c0, nr - 1, nc - 1])
    ops.append(34); sels.append([0, 0, nr - 1, nc - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c909285e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c909285e"
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
                                f"for task c909285e"
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
                    f"Failed to build a complete episode for task c909285e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c909285e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
