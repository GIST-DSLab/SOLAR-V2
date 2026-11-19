"""
ARC Task: be94b721 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background needs to be fixed for the episode. Which object is
    # kept depends on its size, not its colour, so object colours stay random.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def neigh(p):
        r, c = p
        return {(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)}

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    no = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    nc = unifint(diff_lb, diff_ub, (no + 1, max(no + 1, 2 * no)))
    inds = {(r, c) for r in range(h) for c in range(w)}
    ch = random.choice(sorted(inds))
    shp = {ch}
    inds.discard(ch)
    for _ in range(nc - 1):
        cands = set()
        for p in shp:
            cands |= dneigh(p)
        cands = (cands & inds) - shp
        if not cands:
            raise ValueError("cannot grow main shape")
        shp.add(random.choice(sorted(cands)))
    nb = set()
    for p in shp:
        nb |= neigh(p)
    inds = (inds - shp) - nb
    trgc = random.choice(remcols)
    for r, c in shp:
        gi[r][c] = trgc
    r0 = min(r for r, _ in shp); c0 = min(c for _, c in shp)
    r1 = max(r for r, _ in shp); c1 = max(c for _, c in shp)
    go = [[bgc] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c in shp:
        go[r - r0][c - c0] = trgc
    for _ in range(no):
        if not inds:
            break
        ch = random.choice(sorted(inds))
        s = {ch}
        nc2 = unifint(diff_lb, diff_ub, (1, nc - 1))
        for _ in range(nc2 - 1):
            cands = set()
            for p in s:
                cands |= dneigh(p)
            cands = (cands & inds) - s
            if not cands:
                break
            s.add(random.choice(sorted(cands)))
        col = random.choice(remcols)
        for r, c in s:
            gi[r][c] = col
        nb = set()
        for p in s:
            nb |= neigh(p)
        inds = (inds - s) - nb
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading of the task: the background is the largest single-colour
    # 4-connected region. The object to extract is the next largest one; the
    # generator makes it strictly bigger than every distractor.
    # The code-golf entry picks "the most frequent non-background colour" and
    # takes the bbox of all its cells. That only works when no other object
    # shares the target's colour, and this generator allows colours to repeat,
    # so the component reading (the same one the verifier uses) is kept here.
    # The op order differs from the maker: crop to the object's bbox first,
    # then clear the leftover distractor fragments inside the new small canvas.
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    seen = np.zeros((hi, wi), bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                cr, cc = dq.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            comps.append((comp, col))
    comps.sort(key=lambda t: len(t[0]), reverse=True)
    bgc = comps[0][1]
    # main object = largest component that is not background-coloured
    obj = next(cp for cp in comps if cp[1] != bgc)[0]
    objset = set(obj)
    r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
    c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # 1. crop the canvas to the main object's bounding rectangle (whole rectangle intended)
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
    # 2. inside the cropped canvas, clear every cell that is neither the object
    #    nor background (fragments of other objects) to background
    contam = [(r - r0, c - c0) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
              if (r, c) not in objset and int(I[r, c]) != bgc]
    if contam:
        ops.append(int(bgc)); sels.append(sel_of(contam))
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
                        f"num_examples+1 ({num_examples + 1}) for task be94b721"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task be94b721"
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
                                f"for task be94b721"
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
                    f"Failed to build a complete episode for task be94b721 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"be94b721-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
