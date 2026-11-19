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
    # The rule depends only on structure: one scattered noise colour, solid rectangular
    # regions, and the region with the most noise holes wins. Only the noise role is fixed
    # per episode. Region colours are sampled for each instance because the number of
    # regions changes with difficulty.
    noisec = random.choice(range(10))
    return {"noisec": noisec}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    lo_i = max(a, min(b, lo_i))
    hi_i = max(lo_i, min(b, hi_i))
    return random.randint(lo_i, hi_i)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, noisec=None, **kwargs) -> dict:
    if noisec is None:
        noisec = random.choice(range(10))
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    remcols = [c for c in range(10) if c != noisec]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = np.full((h, w), starterc, dtype=int)
    # regions: list of [r0, c0, r1, c1, color] (inclusive); every region is a rectangle
    regions = [[0, 0, h - 1, w - 1, starterc]]
    for k in range(ncols - 1):
        cands = [i for i, (r0, c0, r1, c1, _) in enumerate(regions)
                 if (r1 - r0 + 1) > 5 or (c1 - c0 + 1) > 5]
        if not cands:
            break
        i = random.choice(cands)
        r0, c0, r1, c1, col = regions[i]
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh > 5 and ww > 5:
            ax = random.choice((0, 1))
        elif hh > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = random.randint(r0 + 3, r1 - 2)
            regions[i] = [r0, c0, loci - 1, c1, col]
            regions.append([loci, c0, r1, c1, ccols[k]])
            gi[loci:r1 + 1, c0:c1 + 1] = ccols[k]
        else:
            locj = random.randint(c0 + 3, c1 - 2)
            regions[i] = [r0, c0, r1, locj - 1, col]
            regions.append([r0, locj, r1, c1, ccols[k]])
            gi[r0:r1 + 1, locj:c1 + 1] = ccols[k]
    regions.sort(key=lambda t: (t[2] - t[0] + 1) * (t[3] - t[1] + 1))
    allowances = [max(1, ((r1 - r0 - 1) * (c1 - c0 - 1)) // 2) for r0, c0, r1, c1, _ in regions]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [random.randint(0, min(meann, a)) for a in allowances]
    while max(chosens) == 0:
        chosens = [random.randint(0, min(meann, a)) for a in allowances]
    mx = max(chosens)
    fixinds = [i for i, c in enumerate(chosens) if c == mx]
    gogocol = regions[fixinds[0]][4]
    for i in fixinds[1:]:
        chosens[i] -= 1
    for (r0, c0, r1, c1, _), cnt in zip(regions, chosens):
        inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if not inner:
            continue
        for (r, c) in random.sample(inner, min(cnt, len(inner))):
            gi[r, c] = noisec
    go = np.full((1, 1), gogocol, dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading taken from the per-colour formulation: treat each COLOUR (not each connected
    component) as one unit. For each colour k, take its span: the rows that contain k
    crossed with the columns that contain k. The noise colour is scattered, and every other
    colour fills a solid rectangle. A rectangle's 'holes' (span area minus its own cells)
    are exactly the noise pixels it holds. The winner is the colour with the most holes.
    Ops: CropGrid to the winner's span (a full rectangle that can be named from I, holes
    included), then CropGrid to the corner cell of that rectangle. Noise only lies in a
    region's interior, so the corner is always the region's colour. Then Submit.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # connected components (4-conn), used to pick out the noise colour structurally
    seen = np.zeros((hi, wi), dtype=bool)
    comp_colors = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comp_colors.append(col)
    if len(set(comp_colors)) == len(comp_colors):
        cnt = Counter(I.flatten().tolist())
        noise = min(cnt.items(), key=lambda kv: kv[1])[0]
    else:
        noise = Counter(comp_colors).most_common(1)[0][0]

    # per-colour span and hole count
    best = None
    best_holes = -1
    for k in sorted(set(I.flatten().tolist())):
        if k == noise:
            continue
        rows = [r for r in range(hi) if (I[r] == k).any()]
        cols = [c for c in range(wi) if (I[:, c] == k).any()]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        span_area = len(rows) * len(cols)
        holes = span_area - int(np.count_nonzero(I == k))
        if holes > best_holes:
            best_holes = holes
            best = (r0, c0, r1, c1)

    r0, c0, r1, c1 = best
    ops, sels = [], []
    # Crop to the winner's full rectangular span (background holes included on purpose)
    ops.append(33); sels.append([int(r0), int(c0), int(r1 - r0), int(c1 - c0)])
    # Crop to the region's corner cell, which always has the region's own colour
    ops.append(33); sels.append([0, 0, 0, 0])
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
