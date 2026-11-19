"""
ARC Task: e3497940 (RE-ARC) — LLM-generated grid_maker
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
    # The rule only depends on the layout (background, bar, arms), not on the arm colours.
    # So only the background and bar colours are fixed for the episode.
    cols = list(range(10))
    bgc, barc = random.sample(cols, 2)
    return {"bgc": bgc, "barc": barc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    cols = list(range(10))
    if bgc is None or barc is None:
        bgc, barc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (4, min(30, max_h)))
    wmax = max(3, min(14, (max_w - 1) // 2))
    w = unifint(diff_lb, diff_ub, (3, wmax))
    remcols = [c for c in cols if c != bgc and c != barc]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    nlinesocc = unifint(diff_lb, diff_ub, (1, h))
    linesocc = random.sample(range(h), nlinesocc)
    rs = np.full((h, w), bgc, dtype=int)
    ls = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for idx in linesocc:
        j = unifint(diff_lb, diff_ub, (1, w - 1))
        obj = [random.choice(ccols) for _ in range(j)]
        go[idx, :j] = obj
        slen = random.randint(1, j)
        obj2 = obj[:slen]
        obj1 = obj
        if random.choice((True, False)):
            obj1, obj2 = obj2, obj1
        rs[idx, :len(obj1)] = obj1
        ls[idx, :len(obj2)] = obj2
    gi = np.hstack([ls[:, ::-1], np.full((h, 1), barc, dtype=int), rs])
    go = go[:, ::-1]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Same rule as before: each row keeps its LONGER arm, read leftward from the bar,
    and only the left half is kept. This time the work is split differently.

    1. Mirror the WHOLE grid in one stroke (FlipH on the full canvas). Every right arm
       now lands on the left side, laid out outward from the bar. Both arms of a row
       share a colour prefix, so this is already right for every row whose right arm
       was at least as long as its left arm.
    2. For rows where the LEFT arm was the longer one, the mirrored left side only shows
       the shorter (right) arm. The rest of the original left arm sits at exactly the
       same cells it had in I. Restore it ONE COLOUR AT A TIME: for each colour, one
       Color op on every such leftover left-arm cell of that colour in I, across all
       rows. These cells are read straight from I: the cells of the left arm that
       reach further from the bar than the right arm does.
    3. Crop to the left half."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    w = (wi - 1) // 2           # half width; the bar is column w
    bgc = int(I[0, 0])          # column 0 is always background (arms are at most w-1 long)

    ops, sels = [], []

    # 1. whole-grid reflection; the bbox is exactly the full canvas we mean to mirror.
    #    Skip it if the grid is already symmetric (it would do nothing).
    if not np.array_equal(np.fliplr(I), I):
        ops.append(26)
        sels.append([0, 0, hi - 1, wi - 1])

    # 2. left-arm cells that reach past the right arm, grouped by their colour in I
    by_color = {}
    for r in range(hi):
        ll = 0
        while ll < w and I[r, w - 1 - ll] != bgc:
            ll += 1
        lr = 0
        while w + 1 + lr < wi and I[r, w + 1 + lr] != bgc:
            lr += 1
        for k in range(lr, ll):
            c = w - 1 - k
            by_color.setdefault(int(I[r, c]), []).append((r, c))
    for col in sorted(by_color):
        ops.append(col)
        sels.append(sel_of(by_color[col]))

    # 3. keep the left half only (bbox == exactly that full rectangle)
    ops.append(33)
    sels.append([0, 0, hi - 1, w - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task e3497940"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e3497940"
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
                                f"for task e3497940"
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
                    f"Failed to build a complete episode for task e3497940 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e3497940-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
