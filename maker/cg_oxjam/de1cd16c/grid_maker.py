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
from maker.sel_helpers import sel_of
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    # Only the scattered noise colour plays a fixed role across the episode.
    # The rectangle colours are free: the rule reads them from structure, not identity.
    noisec = random.choice(range(10))
    return {"noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    if noisec is None:
        noisec = choice(cols)
    remcols = remove(noisec, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = canvas(starterc, (h, w))
    for k in range(ncols - 1):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: height(o) > 5 or width(o) > 5)
        if len(objs) == 0:
            break
        objs = totuple(objs)
        obj = choice(objs)
        if height(obj) > 5 and width(obj) > 5:
            ax = choice((0, 1))
        elif height(obj) > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = randint(uppermost(obj) + 3, lowermost(obj) - 2)
            newobj = sfilter(toindices(obj), lambda ij: ij[0] >= loci)
        else:
            locj = randint(leftmost(obj) + 3, rightmost(obj) - 2)
            newobj = sfilter(toindices(obj), lambda ij: ij[1] >= locj)
        gi = fill(gi, ccols[k], newobj)
    objs = order(objects(gi, T, F, F), size)
    allowances = [max(1, ((height(o) - 2) * (width(o) - 2)) // 2) for o in objs]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    while max(chosens) == 0:
        chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    mx = max(chosens)
    fixinds = [idx for idx, cnt in enumerate(chosens) if cnt == mx]
    gogoind = fixinds[0]
    gogocol = color(objs[gogoind])
    fixinds = fixinds[1:]
    for idx in fixinds:
        chosens[idx] -= 1
    for obj, cnt in zip(objs, chosens):
        locs = sample(totuple(backdrop(inbox(toindices(obj)))), cnt)
        gi = fill(gi, noisec, locs)
    go = canvas(gogocol, (1, 1))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the code-golf entry): each region colour is one unit, i.e. one
    solid rectangle. Score it by the foreign (noise) pixels that sit inside the span of its
    rows and columns. The best-scoring colour is the answer.
    Where this differs from the maker: the maker reduces straight to one cell. Here the unit
    is the colour's rectangle, so the trajectory first crops to that rectangle, which keeps the
    winning region and its noise. It then crops to one region-coloured cell of it.
    Every selection is computed from I alone.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- noise colour (same structural test as the verifier) ---
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

    # --- per-colour rectangle: count noise inside its row-span x column-span ---
    best = None
    for col in sorted(set(I.flatten().tolist())):
        if col == noise:
            continue
        rs, cs = np.nonzero(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        n = int(np.count_nonzero(I[r0:r1 + 1, c0:c1 + 1] == noise))
        if best is None or n > best[0]:
            best = (n, col, int(r0), int(c0), int(r1), int(c1))
    _, wcol, r0, c0, r1, c1 = best

    ops, sels = [], []
    # 1. Isolate the winning colour's rectangle. This is a whole-rectangle crop, so a bbox
    #    selection is exact here.
    if not (r0 == 0 and c0 == 0 and r1 == hi - 1 and c1 == wi - 1):
        ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    # 2. Reduce to one cell of the region colour inside that rectangle.
    sub = I[r0:r1 + 1, c0:c1 + 1]
    ys, xs = np.nonzero(sub == wcol)
    ry, rx = int(ys[0]), int(xs[0])
    if not (r0 == 0 and c0 == 0 and r1 == hi - 1 and c1 == wi - 1):
        ops.append(33); sels.append([ry, rx, 0, 0])
    else:
        ops.append(33); sels.append([r0 + ry, c0 + rx, 0, 0])

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
