"""
ARC Task: 08ed6ac7 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour is sampled independently of the per-bar
    # colours.  The rule ignores bar colours (bars are ranked by where they
    # start), so only bgc is fixed for the episode.
    bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 4)])
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    colopts = list(range(10))
    if bgc is None:
        bgc = random.choice([c for c in colopts if c not in (1, 2, 3, 4)])
    h = max(4, min(max_h, unifint(diff_lb, diff_ub, (4, max_h))))
    w = max(4, min(max_w, unifint(diff_lb, diff_ub, (4, max_w))))
    remcols = [c for c in colopts if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    nbars = max(4, min(w, unifint(diff_lb, diff_ub, (4, w))))
    barlocs = random.sample(list(range(w)), nbars)
    barhs = random.sample(list(range(h)), 4)
    barcols = [random.choice(remcols) for _ in range(nbars)]
    barhsfx = [random.choice(barhs) for _ in range(nbars - 4)] + list(barhs)
    random.shuffle(barhsfx)
    ordered = sorted(barhs)
    for col, loci, locj in zip(barcols, barhsfx, barlocs):
        for r in range(loci, h):
            gi[r][locj] = col
            go[r][locj] = ordered.index(loci) + 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the row-sweep solution): read the grid from the top
    down. Each time new bars start on a row, they get the next colour in the
    sequence 1, 2, 3, 4. Below that row, every bar keeps the colour it was
    given where it started. So a bar's colour depends only on its START ROW,
    and bars that start on the same row all get one colour together.

    Route: sweep the distinct start rows from the top down. For each start
    row, select every bar that begins there (each bar runs from its top row
    to the bottom of its column, so the set can be named from I alone).
    Recolour that whole set with ONE Color op, using the next number in the
    sequence. This differs from the maker, which recoloured one bar per class
    and copied it onto the others with CopyO/Paste.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def bars_for(bg):
        found = []
        for c in range(w):
            col = I[:, c]
            if int(col[h - 1]) == bg:
                if np.any(col != bg):
                    return None
                continue
            x = int(col[h - 1])
            r = h - 1
            while r - 1 >= 0 and int(col[r - 1]) == x:
                r -= 1
            if np.any(col[:r] != bg):
                return None
            found.append((c, r))          # (column, start row)
        return found

    def ok(cand):
        return cand is not None and len(cand) >= 4 and len({t[1] for t in cand}) == 4

    bars = None
    # The background is fixed for the episode: read it from the demonstrations.
    if examples:
        cnt = Counter()
        for ex in examples:
            xi = np.asarray(ex[0], dtype=int)
            cnt.update(xi.flatten().tolist())
        for v, _ in cnt.most_common():
            if v in (1, 2, 3, 4):
                continue
            cand = bars_for(v)
            if ok(cand):
                bars = cand
            break
    if bars is None:
        for v, _ in Counter(I.flatten().tolist()).most_common():
            if v in (1, 2, 3, 4):
                continue
            cand = bars_for(v)
            if ok(cand):
                bars = cand
                break
    if bars is None:
        bars = []

    # Sweep the start rows top-down; each new start row takes the next colour.
    starts = sorted({r for (_, r) in bars})
    for k, r0 in enumerate(starts, start=1):
        cohort = sorted(c for (c, r) in bars if r == r0)
        # leave out bars that already hold colour k (they need no change)
        cohort = [c for c in cohort if int(I[r0, c]) != k]
        if not cohort:
            continue
        cells = [(r, c) for c in cohort for r in range(r0, h)]
        ops.append(k)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])    # full grid = exactly this rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
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
                                f"for task 08ed6ac7"
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
                    f"Failed to build a complete episode for task 08ed6ac7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"08ed6ac7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
