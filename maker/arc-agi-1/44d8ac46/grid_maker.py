"""
ARC Task: 44d8ac46 (RE-ARC) — LLM-generated grid_maker
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
    # 2 is the "marker" color the rule paints with -> never a scene color.
    cols = remove(2, interval(0, 10, 1))
    bgc = choice(cols)
    # object colors are irrelevant to the rule (only hole shape matters) -> not fixed
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    par = choice((0, 1))  # alternate square / blob holes so both cases show up
    while succ < num and tr <= maxtrials:
        tr += 1
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(5, 7)
        ow = randint(5, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            ensuresq = (succ % 2 == par)
            if ensuresq:
                dim = randint(1, min(oh, ow) - 2)
                iloci = randint(1, oh - dim - 1)
                ilocj = randint(1, ow - dim - 1)
                inpart = backdrop({(loci + iloci, locj + ilocj),
                                   (loci + iloci + dim - 1, locj + ilocj + dim - 1)})
            else:
                cnds = backdrop(inbox(bd))
                ch = choice(totuple(cnds))
                inpart = {ch}
                kk = unifint(diff_lb, diff_ub, (1, len(cnds)))
                for k in range(kk - 1):
                    inpart.add(choice(totuple((cnds - inpart) & mapply(dneighbors, inpart))))
            inpart = frozenset(inpart)
            hi_, wi_ = shape(inpart)
            if hi_ == wi_ and len(inpart) == hi_ * wi_:
                incol = 2
            else:
                incol = bgc
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            gi = fill(gi, bgc, inpart)
            go = fill(go, incol, inpart)
            succ += 1
            indss = (indss - bd) - outbox(bd)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import deque

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Generator never places a box touching the last row/col -> that corner is background.
    bgc = int(I[hi - 1, wi - 1])

    ops, sels = [], []

    # --- object discovery: same-color 4-connected components of non-background cells ---
    seen = np.zeros((hi, wi), dtype=bool)
    objects = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            objects.append(cells)

    # --- rule: an object's hole (background cells inside its bbox) is painted 2
    #           iff that hole is a solid square ---
    for cells in objects:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        hole = [(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                if I[y, x] == bgc]
        if not hole:
            continue
        hr0 = min(p[0] for p in hole)
        hr1 = max(p[0] for p in hole)
        hc0 = min(p[1] for p in hole)
        hc1 = max(p[1] for p in hole)
        hh = hr1 - hr0 + 1
        hw = hc1 - hc0 + 1
        if hh == hw and len(hole) == hh * hw:
            # solid square hole -> one Color2 over that square region
            ops.append(2)
            sels.append([hr0, hc0, hh - 1, hw - 1])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44d8ac46"
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
                                f"for task 44d8ac46"
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
                    f"Failed to build a complete episode for task 44d8ac46 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44d8ac46-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
