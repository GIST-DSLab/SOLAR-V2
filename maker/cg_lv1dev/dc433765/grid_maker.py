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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, src = random.sample(cols, 2)
    return {"bgc": bgc, "src": src}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, src=None) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    if bgc is None or src is None:
        bgc, src = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    if choice((True, False)):
        opts = {(ii, 0) for ii in range(h - 2)} | {(0, jj) for jj in range(1, w - 2, 1)}
        opts = tuple([inds & shoot(s, (1, 1)) for s in opts])
        opts = order(opts, size)
        k = len(opts)
        opt = unifint(diff_lb, diff_ub, (0, k - 1))
        ln = order(opts[opt], first)
        epi = unifint(diff_lb, diff_ub, (2, len(ln) - 1))
        ep = ln[epi]
        ln = ln[:epi - 1][::-1]
        spi = unifint(diff_lb, diff_ub, (0, len(ln) - 1))
        sp = ln[spi]
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (1, 1))})
        go = fill(go, 4, {ep})
    else:
        loci = randint(0, h - 1)
        objw = unifint(diff_lb, diff_ub, (3, w))
        locj1 = randint(0, w - objw)
        locj2 = locj1 + objw - 1
        sp = (loci, locj1)
        ep = (loci, locj2)
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (0, 1))})
        go = fill(go, 4, {ep})
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # The golf reading agrees with the maker's: the single non-4 pixel takes ONE
    # step toward the fixed 4 (sign of row offset, sign of column offset). The golf
    # writes it as one diagonal jump; ARCLE has no diagonal Move, so the same single
    # step is one grab-Move plus (if diagonal) one empty-selection Move of that pixel.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    # background: the canvas colour (grid holds only two non-background pixels)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    palette = set(I.flatten().tolist()) - {bgc}
    src_cands = [c for c in palette if c != 4]
    src_col = src_cands[0] if src_cands else bgc

    r4, c4 = [int(v) for v in np.argwhere(I == 4)[0]]
    rs, cs = [int(v) for v in np.argwhere(I == src_col)[0]]

    dr = (r4 > rs) - (r4 < rs)
    dc = (c4 > cs) - (c4 < cs)

    MOVE = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}

    # ARCLE object ops grab only non-zero cells: a colour-0 pixel is made visible first
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

    for k, st in enumerate(steps):
        ops.append(MOVE[st])
        # first step grabs the pixel; the next continues the same grab (empty selection)
        sels.append(sel_of([(rs, cs)]) if k == 0 else sel_of([]))

    dest = (rs + dr, cs + dc)

    # the grab zeroed the pixel's original footprint; restore just that cell
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
