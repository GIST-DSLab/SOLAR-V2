"""
ARC Task: 1f85a75f (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # object colour must be non-zero: ARCLE Move/Crop treat 0 as "nothing"
    objc = random.choice([c for c in cols if c != bgc and c != 0])
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or objc is None:
        bgc, objc = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = randint(3, min(8, h // 2))
    ow = randint(3, min(8, w // 2))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = randint(max(oh, ow), oh * ow)
    sp = choice(totuple(bounds))
    obj = {sp}
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    remcols = remove(bgc, remove(objc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ((h * w) - len(backdrop(obj))) // 4)))
    gi = canvas(bgc, (h, w))
    obj = shift(obj, (loci, locj))
    gi = fill(gi, objc, obj)
    inds = asindices(gi)
    noisecells = sample(totuple(inds - backdrop(obj)), nnoise)
    noiseobj = frozenset({(choice(ccols), ij) for ij in noisecells})
    gi = paint(gi, noiseobj)
    go = fill(canvas(bgc, (oh, ow)), objc, normalize(obj))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour (noise covers at most ~1/4 of the grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # object colour (verifier): a colour whose cells form exactly ONE
    # 4-connected component; among those, the one with the most cells.
    def n_components(cells):
        cells = set(cells)
        n = 0
        while cells:
            n += 1
            stack = [cells.pop()]
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (y + dy, x + dx)
                    if q in cells:
                        cells.remove(q)
                        stack.append(q)
        return n

    best, best_cells = None, None
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(I == col))]
        if n_components(cells) == 1 and (best_cells is None or len(cells) > len(best_cells)):
            best, best_cells = col, cells
    objc, obj = best, best_cells
    r0 = min(r for r, _ in obj); c0 = min(c for _, c in obj)
    oh = max(r for r, _ in obj) - r0 + 1
    ow = max(c for _, c in obj) - c0 + 1

    ops, sels = [], []
    grid = I.copy()

    # 1. Slide the object to the top-left corner: one grab, then empty-selection
    #    steps (up r0 times, then left c0 times).
    steps = [20] * r0 + [23] * c0
    cur = list(obj)
    for k, op in enumerate(steps):
        ops.append(op)
        sels.append(sel_of(cur) if k == 0 else sel_of([]))
        dr, dc = (-1, 0) if op == 20 else (0, -1)
        cur = [(r + dr, c + dc) for r, c in cur]
    if steps:
        # snapshot semantics: background with grabbed cells zeroed, object pasted on top
        for r, c in obj:
            grid[r, c] = 0
        for r, c in cur:
            grid[r, c] = objc

    # 2. Clean the destination rectangle: everything inside it that is neither
    #    background nor the object (noise cells, 0-holes the object left) -> bgc.
    dirty = [(r, c) for r in range(oh) for c in range(ow)
             if grid[r, c] != bgc and grid[r, c] != objc]
    if dirty:
        ops.append(bgc); sels.append(sel_of(dirty))
        for r, c in dirty:
            grid[r, c] = bgc

    # 3. Crop the canvas to the object's rectangle at the origin (full rectangle).
    ops.append(33); sels.append([0, 0, oh - 1, ow - 1])
    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f85a75f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f85a75f"
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
                                f"for task 1f85a75f"
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
                    f"Failed to build a complete episode for task 1f85a75f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f85a75f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
