"""
ARC Task: dc433765 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
MODES = ["diag", "straight"]


def sample_colors(num_examples=None) -> dict:
    # generator: cols = remove(4, interval(0, 10, 1)); bgc, src = sample(cols, 2)
    cols = [c for c in range(10) if c != 4]
    bgc, src = random.sample(cols, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(MODES):
        examples = [{"mode": m} for m in MODES]
        examples += [{"mode": random.choice(MODES)} for _ in range(n_ex - len(MODES))]
        random.shuffle(examples)
    else:
        examples = [{"mode": m} for m in random.sample(MODES, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "src": src, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, src, mode=None) -> dict:
    def unifint(lo, hi):
        return random.randint(lo + int((hi - lo) * diff_lb), lo + int((hi - lo) * diff_ub))

    def ap(g, name):
        if name == "identity":
            return [row[:] for row in g]
        if name == "vmirror":
            return [row[::-1] for row in g]
        if name == "hmirror":
            return [row[:] for row in g[::-1]]
        if name == "rot180":
            return [row[::-1] for row in g[::-1]]
        if name == "dmirror":
            return [list(r) for r in zip(*g)]
        if name == "cmirror":
            return [list(r) for r in zip(*[row[::-1] for row in g[::-1]])]
        if name == "rot90":
            return [list(r) for r in zip(*g[::-1])]
        if name == "rot270":
            return [list(r) for r in zip(*g)][::-1]
        return [row[:] for row in g]

    mh = max(4, min(30, int(max_h)))
    mw = max(4, min(30, int(max_w)))
    h = unifint(4, mh)
    w = unifint(4, mw)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    if mode is None:
        mode = random.choice(MODES)

    if mode == "diag":
        starts = [(ii, 0) for ii in range(h - 2)] + [(0, jj) for jj in range(1, w - 2)]
        rays = []
        for (i, j) in starts:
            ray = []
            t = 0
            while i + t < h and j + t < w:
                ray.append((i + t, j + t))
                t += 1
            if len(ray) >= 3:
                rays.append(ray)
        rays.sort(key=len)
        opt = unifint(0, len(rays) - 1)
        ln = sorted(rays[opt])
        epi = unifint(2, len(ln) - 1)
        ep = ln[epi]
        pre = ln[:epi - 1][::-1]
        spi = unifint(0, len(pre) - 1)
        sp = pre[spi]
        dsp = (sp[0] + 1, sp[1] + 1)
    else:
        loci = random.randint(0, h - 1)
        objw = unifint(3, w)
        locj1 = random.randint(0, w - objw)
        sp = (loci, locj1)
        ep = (loci, locj1 + objw - 1)
        dsp = (sp[0], sp[1] + 1)

    gi[sp[0]][sp[1]] = src
    gi[ep[0]][ep[1]] = 4
    go[dsp[0]][dsp[1]] = src
    go[ep[0]][ep[1]] = 4

    mfs = ["identity", "dmirror", "cmirror", "vmirror", "hmirror", "rot90", "rot180", "rot270"]
    swapping = {"dmirror", "cmirror", "rot90", "rot270"}
    swap_ok = (h <= mw and w <= mh)
    fns = ["identity"]
    for _ in range(20):
        cand = random.sample(mfs, random.choice((1, 2)))
        par = sum(1 for f in cand if f in swapping) % 2
        if par == 0 or swap_ok:
            fns = cand
            break
    for fn in fns:
        gi = ap(gi, fn)
        go = ap(go, fn)

    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    ops, sels = [], []

    # background = the colour the canvas was painted with (overwhelming majority:
    # the grid holds exactly two non-background pixels)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the two markers: the fixed '4' beacon and the mobile source pixel
    palette = set(I.flatten().tolist()) - {bgc}
    src_col = [c for c in palette if c != 4]
    src_col = src_col[0] if src_col else bgc

    r4, c4 = [int(v) for v in np.argwhere(I == 4)[0]]
    rs, cs = [int(v) for v in np.argwhere(I == src_col)[0]]

    # rule: the source pixel takes ONE step towards the beacon (sign of the offset)
    dr = (r4 > rs) - (r4 < rs)
    dc = (c4 > cs) - (c4 < cs)

    MOVE = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}

    # ARCLE's object ops only grab NON-ZERO cells: if the mobile pixel is colour 0
    # it must first be made visible, then moved, then restored to 0 at its landing spot.
    zero_src = (src_col == 0)
    if zero_src:
        tmp = 1 if bgc != 1 else 2
        ops.append(tmp)
        sels.append(sel_of([(rs, cs)]))

    steps = []
    if dr != 0:
        steps.append((dr, 0))
    if dc != 0:
        steps.append((0, dc))

    for k, (sr, sc) in enumerate(steps):
        ops.append(MOVE[(sr, sc)])
        # first step GRABS the pixel; further steps carry an empty selection so
        # ARCLE keeps the same object grabbed and restores the path behind it
        sels.append(sel_of([(rs, cs)]) if k == 0 else sel_of([]))

    dest = (rs + dr, cs + dc)

    # the grab zeroed the pixel's ORIGINAL footprint; repair just that cell
    if bgc != 0 and (rs, cs) != dest:
        ops.append(bgc)
        sels.append(sel_of([(rs, cs)]))

    if zero_src:
        ops.append(0)
        sels.append(sel_of([dest]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task dc433765"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc433765"
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
                                f"for task dc433765"
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
                    f"Failed to build a complete episode for task dc433765 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc433765-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
