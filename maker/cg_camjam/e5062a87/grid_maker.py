"""
ARC Task: e5062a87 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator samples two colours: the noise/canvas colour (eligcol) and
    # the marker colour (objc). Colour 0 is hard-coded as the hole colour.
    # The marker colour is the colour the rule paints with, so it stays fixed
    # for the whole episode.
    cols = list(range(1, 10))
    eligcol, objc = random.sample(cols, 2)
    return {"eligcol": eligcol, "objc": objc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def _neighbors(cell):
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)}


def generate(diff_lb, diff_ub, max_h, max_w, eligcol=None, objc=None, **kw) -> dict:
    if eligcol is None or objc is None:
        eligcol, objc = random.sample(list(range(1, 10)), 2)
    h = _unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    gi = [[eligcol] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    indset = set(inds)

    sp = random.choice(inds)
    obj = {sp}
    ncells = _unifint(diff_lb, diff_ub, (3, 9))
    for _ in range(ncells - 1):
        cand = set()
        for c in obj:
            cand |= _neighbors(c)
        cand = sorted((cand & indset) - obj)
        obj.add(random.choice(cand))
    r0 = min(r for r, _ in obj)
    c0 = min(c for _, c in obj)
    obj = {(r - r0, c - c0) for r, c in obj}
    oh = max(r for r, _ in obj) + 1
    ow = max(c for _, c in obj) + 1

    nnoise = _unifint(diff_lb, diff_ub, (int(0.2 * h * w), int(0.5 * h * w)))
    for (i, j) in random.sample(inds, nnoise):
        gi[i][j] = 0

    noccs = _unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    for k in range(noccs):
        li = random.randint(0, h - oh)
        lj = random.randint(0, w - ow)
        colr = objc if k == noccs - 1 else 0
        for (dr, dc) in obj:
            gi[li + dr][lj + dc] = colr

    go = [row[:] for row in gi]
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            if all(gi[i + dr][j + dc] == 0 for dr, dc in obj):
                for dr, dc in obj:
                    go[i + dr][j + dc] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: the single shape drawn in the marker colour is a stamp. Every place
    where that exact shape sits in I as a hole of colour 0 gets stamped in the
    marker colour.

    Compared with the golf reading: both treat the unit the same way, as one
    placement of the marker shape painted in one go, so the selection is the
    same. Where they differ is ORDER. The golf entry tries the stamp at each
    offset by sliding over the grid in raster order. The maker spreads outward
    from the marker. Here we follow the golf order: try the template row by
    row, column by column, and stamp it wherever it fits a hole.
    Where they DON'T line up: the golf entry checks each placement against the
    grid it has already stamped, so it rejects placements that overlap an
    earlier stamp. That fits the original 10x10 task only. In this task's
    verifier, holes are found in I (occurrences(I, ...)), so overlapping holes
    are all filled. Fit is therefore tested against I. A placement whose cells
    have all been stamped already is skipped, because painting it again would
    change nothing.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Marker colour: the rarest colour in I (the one stamped shape).
    cnt = Counter(I.flatten().tolist())
    objc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # Check it against the demonstrations: the one colour every example adds.
    if examples:
        seen = set()
        for pair in examples:
            try:
                ei = np.asarray(pair[0], dtype=int)
                eo = np.asarray(pair[1], dtype=int)
            except Exception:
                continue
            if ei.shape != eo.shape:
                continue
            added = set(eo[ei != eo].tolist())
            if len(added) == 1:
                seen.add(added.pop())
        if len(seen) == 1:
            cand = seen.pop()
            if cand in cnt:
                objc = cand

    ops, sels = [], []
    marker = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == objc]
    if marker:
        r0 = min(r for r, _ in marker)
        c0 = min(c for _, c in marker)
        tmpl = sorted((r - r0, c - c0) for r, c in marker)
        th = max(dr for dr, _ in tmpl) + 1
        tw = max(dc for _, dc in tmpl) + 1

        stamped = set()
        # Slide the template over every offset in raster order (golf order).
        for i in range(hi - th + 1):
            for j in range(wi - tw + 1):
                cells = [(i + dr, j + dc) for dr, dc in tmpl]
                if not all(I[r, c] == 0 for r, c in cells):
                    continue  # the template does not fit a hole of 0 in I here
                if all(cell in stamped for cell in cells):
                    continue  # already fully stamped -> would be a no-op
                ops.append(int(objc))
                sels.append(sel_of(cells))
                stamped.update(cells)

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
                        f"num_examples+1 ({num_examples + 1}) for task e5062a87"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e5062a87"
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
                                f"for task e5062a87"
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
                    f"Failed to build a complete episode for task e5062a87 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e5062a87-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
