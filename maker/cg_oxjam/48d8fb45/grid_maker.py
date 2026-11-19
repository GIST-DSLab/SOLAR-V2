"""
ARC Task: 48d8fb45 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule depends only on structure: which object has two colours, and which
    # colour is its majority. Foreground colours can vary; only the background is fixed.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    def neighbors8(ij):
        a, b = ij
        return {(a + da, b + db) for da in (-1, 0, 1) for db in (-1, 0, 1)
                if not (da == 0 and db == 0)}

    if bgc is None:
        bgc = random.choice(list(range(10)))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 15)))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    go = None
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        bx = {(i, j) for i in range(oh) for j in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx = bx - {sp}
        obj = {sp}
        for _ in range(nc - 1):
            nb = set()
            for (a, b) in obj:
                for (x, y) in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    nb.add((x, y))
            cand = sorted((bx - obj) & nb)
            obj.add(random.choice(cand))
        tr += 1
        if not done:
            idx = random.choice(sorted(obj))
            coll = random.choice(remcols)
            mainc = random.choice([c for c in remcols if c != coll])
            rest = obj - {idx}
            cobj = {(coll, idx)} | {(mainc, ij) for ij in rest}
            ri = min(a for a, b in rest); rj = min(b for a, b in rest)
            rh = max(a for a, b in rest) - ri + 1
            rw = max(b for a, b in rest) - rj + 1
            g2 = [[bgc] * rw for _ in range(rh)]
            for a, b in rest:
                g2[a - ri][b - rj] = mainc
            cand_go = g2
        else:
            c = random.choice(remcols)
            cobj = {(c, ij) for ij in obj}
        locopts = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not locopts:
            continue
        loc = random.choice(sorted(locopts))
        plcd = {(c, (a + loc[0], b + loc[1])) for c, (a, b) in cobj}
        plcdi = {ij for _, ij in plcd}
        if plcdi.issubset(inds):
            for c, (a, b) in plcd:
                gi[a][b] = c
            succ += 1
            nbs = set()
            for ij in plcdi:
                nbs |= neighbors8(ij)
            inds = (inds - plcdi) - nbs
            if not done:
                done = True
                go = cand_go
    if go is None:
        raise ValueError("two-colour object not placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour the generator paints first (objects are sparse)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected non-background objects
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] \
                                    and I[na, nb] != bgc:
                                seen[na, nb] = True
                                stack.append((na, nb))
                comps.append(cells)

    # The marked object is the only one with two colours. Its odd cell is the marker;
    # its majority colour is the body that becomes the output.
    target = max(comps, key=lambda cs: len(set(int(I[a, b]) for a, b in cs)))
    cnt = Counter(int(I[a, b]) for a, b in target)
    main = cnt.most_common(1)[0][0]
    body = set((a, b) for a, b in target if int(I[a, b]) == main)

    r0 = min(a for a, b in body); r1 = max(a for a, b in body)
    c0 = min(b for a, b in body); c1 = max(b for a, b in body)

    ops, sels = [], []

    # Inside the body's window, clear every non-background cell that is not part of
    # the body: the marker cell, and any other object lying in a concave gap.
    foreign = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)
               if I[a, b] != bgc and (a, b) not in body]
    if foreign:
        ops.append(int(bgc))
        sels.append(sel_of(foreign))

    # Crop to the body's bounding box. The selection is exactly that full rectangle.
    ops.append(33)
    sels.append([r0, c0, r1 - r0, c1 - c0])

    ops.append(34)
    sels.append([0, 0, r1 - r0, c1 - c0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
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
                                f"for task 48d8fb45"
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
                    f"Failed to build a complete episode for task 48d8fb45 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"48d8fb45-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
