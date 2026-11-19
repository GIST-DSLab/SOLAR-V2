"""
ARC Task: de1cd16c (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule never depends on which colour is the noise. Fixing it for the whole episode
    # just keeps the noise role the same in every example and in the test.
    noisec = random.choice(range(10))
    return {"noisec": noisec}


def _unifint(diff_lb, diff_ub, lo, hi):
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, lo + int(round((hi - lo) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, **kwargs) -> dict:
    cols = list(range(10))
    if noisec is None:
        noisec = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, 6, max(6, max_h))
    w = _unifint(diff_lb, diff_ub, 6, max(6, max_w))
    remcols = [c for c in cols if c != noisec]
    ncols = _unifint(diff_lb, diff_ub, 2, 9)
    ccols = random.sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = np.full((h, w), starterc, dtype=int)
    # Every region is a rectangle with its own colour: (r0, c0, r1, c1, col), inclusive.
    rects = [(0, 0, h - 1, w - 1, starterc)]
    for k in range(ncols - 1):
        cands = [i for i, (r0, c0, r1, c1, _) in enumerate(rects)
                 if (r1 - r0 + 1) > 5 or (c1 - c0 + 1) > 5]
        if not cands:
            break
        i = random.choice(cands)
        r0, c0, r1, c1, col = rects[i]
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh > 5 and ww > 5:
            ax = random.choice((0, 1))
        elif hh > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = random.randint(r0 + 3, r1 - 2)
            rects[i] = (r0, c0, loci - 1, c1, col)
            rects.append((loci, c0, r1, c1, ccols[k]))
            gi[loci:r1 + 1, c0:c1 + 1] = ccols[k]
        else:
            locj = random.randint(c0 + 3, c1 - 2)
            rects[i] = (r0, c0, r1, locj - 1, col)
            rects.append((r0, locj, r1, c1, ccols[k]))
            gi[r0:r1 + 1, locj:c1 + 1] = ccols[k]
    rects.sort(key=lambda t: (t[2] - t[0] + 1) * (t[3] - t[1] + 1))
    allowances = [max(1, ((t[2] - t[0] - 1) * (t[3] - t[1] - 1)) // 2) for t in rects]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [random.randint(0, min(meann, a)) for a in allowances]
    while max(chosens) == 0:
        chosens = [random.randint(0, min(meann, a)) for a in allowances]
    mx = max(chosens)
    fixinds = [i for i, c in enumerate(chosens) if c == mx]
    gogocol = rects[fixinds[0]][4]
    for i in fixinds[1:]:
        chosens[i] -= 1
    for (r0, c0, r1, c1, _), cnt in zip(rects, chosens):
        inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for (r, c) in random.sample(inner, min(cnt, len(inner))):
            gi[r, c] = noisec
    go = np.array([[gogocol]], dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (read off I only): the grid is split into solid rectangular regions, and one
    colour is scattered as noise. The region whose bounding box holds the most noise
    pixels wins, and the output is a single cell of that region's colour.
    Unit of work here: one piece of NOISE evidence, not a cell of the region.
    Crop down to a noise pixel that sits inside the winning region, then the region
    takes that pixel over: Color it with the region's colour.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Connected same-colour components (4-connectivity, background not excluded).
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((col, cells))

    # Find the noise colour.
    comp_colors = [col for col, _ in comps]
    if len(set(comp_colors)) == len(comp_colors):
        cnt = Counter(I.flatten().tolist())
        noise = min(cnt.items(), key=lambda kv: kv[1])[0]
    else:
        noise = Counter(comp_colors).most_common(1)[0][0]

    # Find the winning region: the most noise pixels inside its bounding box.
    best = None
    best_count = -1
    for col, cells in comps:
        if col == noise:
            continue
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        n = int(np.count_nonzero(I[r0:r1 + 1, c0:c1 + 1] == noise))
        if n > best_count:
            best_count = n
            best = (col, r0, r1, c0, c1)

    win_col, r0, r1, c0, c1 = best
    noise_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if I[r, c] == noise]
    nr, nc = noise_cells[0]  # one noise pixel the winning region holds

    ops, sels = [], []
    # Crop the canvas to that single noise pixel. A 1x1 bbox is exactly the intended cell.
    ops.append(33); sels.append([int(nr), int(nc), 0, 0])
    # The winning region takes over its noise pixel. Noise != region colour, so this is visible.
    ops.append(int(win_col)); sels.append([0, 0, 0, 0])
    ops.append(34); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task de1cd16c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task de1cd16c"
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
                                f"for task de1cd16c"
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
                    f"Failed to build a complete episode for task de1cd16c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"de1cd16c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
