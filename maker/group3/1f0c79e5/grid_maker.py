"""
ARC Task: 1f0c79e5 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, round(a + (b - a) * d)))
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 24)))
    inds = {(i, j) for i in range(h) for j in range(w)}
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for _ in range(nobjs):
        cands = [ij for ij in inds
                 if all((ij[0] + a, ij[1] + b) in inds for a, b in obj)]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        plcd = [(loc[0] + a, loc[1] + b) for a, b in obj]
        nred = unifint(diff_lb, diff_ub, (1, 3))
        reds = random.sample(plcd, nred)
        for r, c in plcd:
            gi[r, c] = objc
        for r, c in reds:
            gi[r, c] = 2
        for idx in reds:
            dr = 2 * (idx[0] - loc[0]) - 1
            dc = 2 * (idx[1] - loc[1]) - 1
            for r, c in plcd:
                while 0 <= r < h and 0 <= c < w:
                    go[r, c] = objc
                    r += dr
                    c += dc
        rem = set(plcd)
        for r, c in plcd:
            rem |= {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
        inds -= rem
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour (blob cells are at most h*w/6 of the grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    others = [v for v in sorted(set(I.flatten().tolist())) if v not in (bgc, 2)]
    if not others:
        return [34], [[0, 0, hi - 1, wi - 1]]
    objc = others[0]

    # 2x2 blocks = 4-connected components of non-background cells
    seen = set()
    blocks = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            comp, st = [], [(r, c)]
            seen.add((r, c))
            while st:
                a, b = st.pop()
                comp.append((a, b))
                for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in seen \
                            and I[na, nb] != bgc:
                        seen.add((na, nb))
                        st.append((na, nb))
            blocks.append(sorted(comp))

    def ray(r, c, dr, dc):
        out = []
        while 0 <= r < hi and 0 <= c < wi:
            out.append((r, c))
            r += dr
            c += dc
        return out

    # unit of work: ONE diagonal ray shot from one block cell, in the direction the
    # block's red corner points (red at top-left -> up-left, etc.)
    rays = []
    for comp in blocks:
        r0 = min(a for a, _ in comp)
        c0 = min(b for _, b in comp)
        for (rr, rc) in comp:
            if I[rr, rc] != 2:
                continue
            dr, dc = 2 * (rr - r0) - 1, 2 * (rc - c0) - 1
            # the red corner's own ray first, then the other block cells
            starts = [(rr, rc)] + [p for p in comp if p != (rr, rc)]
            for (sr, sc) in starts:
                rays.append(ray(sr, sc, dr, dc))

    # a ray adds something only through cells not already objc in I; drop a ray
    # when every such cell is also covered by another kept ray (rule-only, from I)
    def gain(rr):
        return {p for p in rr if I[p] != objc}
    keep = list(rays)
    i = 0
    while i < len(keep):
        mine = gain(keep[i])
        rest = set()
        for j, rr in enumerate(keep):
            if j != i:
                rest |= set(rr)
        if mine <= rest:
            keep.pop(i)
        else:
            i += 1

    ops, sels = [], []
    for rr in keep:
        ops.append(int(objc))
        sels.append(sel_of(rr))
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
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
                                f"for task 1f0c79e5"
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
                    f"Failed to build a complete episode for task 1f0c79e5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f0c79e5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
