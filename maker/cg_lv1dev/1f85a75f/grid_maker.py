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
import random
from collections import Counter

from maker.sel_helpers import sel_of  # noqa: F401  (only full-rectangle selections are needed here)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, objc)]
    return {"bgc": bgc, "objc": objc, "remcols": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, remcols=None, **kwargs) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(lo + random.random() * (hi - lo)))) if b >= a else a

    cols = list(range(10))
    if bgc is None or objc is None:
        bgc, objc = random.sample(cols, 2)
    if remcols is None:
        remcols = [c for c in cols if c not in (bgc, objc)]

    H = max(10, max_h)
    W = max(10, max_w)
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = randint(3, min(8, h // 2))
    ow = randint(3, min(8, w // 2))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    ncells = randint(max(oh, ow), oh * ow)
    sp = choice(sorted(bounds))
    obj = {sp}
    for _ in range(ncells - 1):
        cands = set()
        for (i, j) in obj:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cands.add((i + di, j + dj))
        cands = (cands & bounds) - obj
        obj.add(choice(sorted(cands)))
    mi = min(i for i, _ in obj)
    mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, min(numc, len(remcols)))
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, (h * w - oh * ow) // 4)))
    gi = [[bgc] * w for _ in range(h)]
    for i, j in obj:
        gi[i + loci][j + locj] = objc
    bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
    free = [(i, j) for i in range(h) for j in range(w) if (i, j) not in bd]
    for (i, j) in sample(free, min(nnoise, len(free))):
        gi[i][j] = choice(ccols)
    go = [[bgc] * ow for _ in range(oh)]
    for i, j in obj:
        go[i][j] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the majority colour (the canvas the generator paints first).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Split every non-background colour into its 4-connected components.
    visited = np.zeros((hi, wi), bool)
    comps_by_color = {}
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == col:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps_by_color.setdefault(col, []).append(comp)

    # The object's colour forms exactly one connected piece. Noise colours are
    # scattered over many pieces. If several colours have one piece, take the
    # one with the most cells.
    single = [col for col, cs in comps_by_color.items() if len(cs) == 1]
    if single:
        objcol = max(single, key=lambda col: len(comps_by_color[col][0]))
        cells = comps_by_color[objcol][0]
    else:
        cells = max((cp for cs in comps_by_color.values() for cp in cs), key=len)

    rs = [p[0] for p in cells]
    cs_ = [p[1] for p in cells]
    r0, c0 = min(rs), min(cs_)
    oh = max(rs) - r0 + 1
    ow = max(cs_) - c0 + 1

    ops, sels = [], []
    # Crop to the object colour's full bounding rectangle, including the
    # background cells inside it. These are exactly the rows and columns
    # that contain the object colour.
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
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
