"""
ARC Task: 36fdfd69 (RE-ARC) — LLM-generated grid_maker
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

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # generator: cols = all colors except 4 ; bgc, fgc, objc = sample(cols, 3)
    cols = [c for c in range(10) if c != 4]
    bgc, fgc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "fgc": fgc, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgc=None, objc=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None or fgc is None or objc is None:
        bgc, fgc, objc = sample(cols, 3)

    hu = max(10, min(30, max_h))
    wu = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, hu))
    w = unifint(diff_lb, diff_ub, (10, wu))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 30)))

    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    namt = randint(int(0.35 * h * w), int(0.65 * h * w))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = tuple(e for e in gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            ncells = randint(2, oh * ow - 1)
            obj = {choice(totuple(bd))}
            ok = True
            for k in range(ncells - 1):
                opts = totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))
                if len(opts) == 0:
                    ok = False
                    break
                obj.add(choice(opts))
            if not ok:
                continue
            guard = 0
            while len(obj) == height(obj) * width(obj):
                guard += 1
                if guard > 20:
                    ok = False
                    break
                obj = {choice(totuple(bd))}
                bad = False
                for k in range(ncells - 1):
                    opts = totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))
                    if len(opts) == 0:
                        bad = True
                        break
                    obj.add(choice(opts))
                if bad:
                    ok = False
                    break
            if not ok:
                continue
            obj = normalize(obj)
            oh, ow = shape(obj)
            obj = shift(obj, loc)
            bd = backdrop(obj)
            gi2 = fill(gi, fgc, bd)
            gi2 = fill(gi2, objc, obj)
            if colorcount(gi2, objc) < min(colorcount(gi2, fgc), colorcount(gi2, bgc)):
                succ += 1
                inds = (inds - bd) - (outbox(bd) | outbox(outbox(bd)))
                gi = gi2
                go = fill(go, 4, bd)
                go = fill(go, objc, obj)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (measured from I only):
       take the least-frequent colour's cells; for every such cell, gather the
       least-colour cells within Chebyshev distance <= 2 and take that group's
       bounding box; union all boxes; repeat that closure once more.  Every cell
       of the resulting backdrop region that is not itself a least-colour cell
       becomes 4.  O is never read to decide what to paint."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # --- least colour of I (the object colour; generator guarantees it is rarest)
    cnt = Counter(I.flatten().tolist())
    least = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    obj_cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == least}

    def closure(cells):
        out = set()
        cl = list(cells)
        for (r0, c0) in cl:
            near = [(r, c) for (r, c) in cl
                    if max(abs(r - r0), abs(c - c0)) < 3]
            rs = [p[0] for p in near]
            cs = [p[1] for p in near]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    out.add((r, c))
        return out

    region = set()
    if obj_cells:
        region = closure(closure(obj_cells))

    # --- split the region into connected groups (one group = one object's backdrop)
    seen = set()
    groups = []
    for cell in sorted(region):
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in region and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        groups.append(comp)

    groups.sort(key=lambda g: (min(p[0] for p in g), min(p[1] for p in g)))

    # --- paint each group's backdrop (minus the preserved object cells) with 4
    for comp in groups:
        target = sorted(p for p in comp if p not in obj_cells)
        if not target:
            continue
        ops.append(4)
        sels.append(sel_of(target))

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
                        f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36fdfd69"
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
                                f"for task 36fdfd69"
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
                    f"Failed to build a complete episode for task 36fdfd69 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36fdfd69-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
