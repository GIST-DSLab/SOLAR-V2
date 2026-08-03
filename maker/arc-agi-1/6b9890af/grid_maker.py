"""
ARC Task: 6b9890af (RE-ARC) — LLM-generated grid_maker
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
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, sqc: int) -> dict:
    cols = interval(0, 10, 1)

    rotf = choice((identity, rot90, rot180, rot270))
    swapped = rotf in (rot90, rot270)
    lim_h = max_w if swapped else max_h
    lim_w = max_h if swapped else max_w

    oh_ub = max(2, min(5, (lim_h - 2) // 2))
    ow_ub = max(2, min(5, (lim_w - 2) // 2))
    oh = unifint(diff_lb, diff_ub, (2, oh_ub))
    ow = unifint(diff_lb, diff_ub, (2, ow_ub))
    h = unifint(diff_lb, diff_ub, (2 * oh + 2, max(2 * oh + 2, lim_h)))
    w = unifint(diff_lb, diff_ub, (2 * ow + 2, max(2 * ow + 2, lim_w)))

    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    while shape(obj) != (oh, ow):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))

    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    maxfac = max(1, maxfac)
    fac = unifint(diff_lb, diff_ub, (1, maxfac))

    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    obj = {(choice(ccols), ij) for ij in obj}

    gi = canvas(bgc, (h, w))
    sq = box(frozenset({(0, 0), (oh * fac + 1, ow * fac + 1)}))
    loci = randint(0, h - (oh * fac + 2) - oh)
    locj = randint(0, w - (ow * fac + 2))
    gi = fill(gi, sqc, shift(sq, (loci, locj)))
    loci = randint(loci + oh * fac + 2, h - oh)
    locj = randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    gi = paint(gi, objp)

    go = canvas(bgc, (oh * fac + 2, ow * fac + 2))
    go = fill(go, sqc, sq)
    go2 = paint(canvas(bgc, (oh, ow)), obj)
    upscobj = asobject(upscale(go2, fac))
    go = paint(go, shift(upscobj, (1, 1)))

    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- find same-color, diagonally-connected, non-background components ---
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append((col, cells))

    # --- the container: component whose cells are exactly the border of its bbox (largest) ---
    best = None
    for col, cells in comps:
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh < 3 or bw < 3:
            continue
        border = {(y, x)
                  for y in range(r0, r1 + 1)
                  for x in range(c0, c1 + 1)
                  if y in (r0, r1) or x in (c0, c1)}
        if set(cells) == border:
            area = bh * bw
            if best is None or area > best[0]:
                best = (area, r0, c0, bh, bw, set(cells))
    _, br, bc, bh, bw, bcells = best

    # --- the small pattern: every non-bg cell outside the container ---
    pts = [(r, c) for r in range(hi) for c in range(wi)
           if I[r, c] != bgc and (r, c) not in bcells]
    prs = [y for y, _ in pts]
    pcs = [x for _, x in pts]
    pr, pc = min(prs), min(pcs)
    oh_p = max(prs) - pr + 1
    ow_p = max(pcs) - pc + 1

    fac = (bh - 2) // oh_p

    ops, sels = [], []

    # each pattern pixel becomes a fac x fac block inside the container's interior
    for i in range(oh_p):
        for j in range(ow_p):
            v = int(I[pr + i, pc + j])
            if v == bgc:
                continue
            ops.append(v)
            sels.append([br + 1 + i * fac, bc + 1 + j * fac, fac - 1, fac - 1])

    # keep only the container
    ops.append(33)
    sels.append([br, bc, bh - 1, bw - 1])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
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
                                f"for task 6b9890af"
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
                    f"Failed to build a complete episode for task 6b9890af "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6b9890af-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
