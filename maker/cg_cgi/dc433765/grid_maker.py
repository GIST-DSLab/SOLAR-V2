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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, src=None, **kw) -> dict:
    from dsl import (canvas, asindices, shoot, order, size, first, fill, add,
                     identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    from utils import unifint
    cols = [c for c in range(10) if c != 4]
    if bgc is None or src is None:
        bgc, src = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    if random.choice((True, False)):
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
        loci = random.randint(0, h - 1)
        objw = unifint(diff_lb, diff_ub, (3, w))
        locj1 = random.randint(0, w - objw)
        locj2 = locj1 + objw - 1
        sp = (loci, locj1)
        ep = (loci, locj2)
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (0, 1))})
        go = fill(go, 4, {ep})
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf entry): one empty gap line between the source and the
    # beacon '4' disappears. On each axis where they are separated, everything
    # past that gap line on the source's side (a half-plane strip) slides one step
    # toward the beacon. The strip at the grid edge is left empty, so it is
    # refilled with the background colour.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    palette = set(I.flatten().tolist()) - {bgc, 4}
    src_col = palette.pop() if palette else bgc
    r4, c4 = [int(v) for v in np.argwhere(I == 4)[0]]
    rs, cs = [int(v) for v in np.argwhere(I == src_col)[0]]

    # Object ops only pick up non-zero cells, so a colour-0 source is shown in a
    # temporary colour while it moves.
    zero_src = (src_col == 0)
    if zero_src:
        tmp = 1 if bgc != 1 else 2
        ops.append(tmp); sels.append(sel_of([(rs, cs)]))

    cur_r, cur_c = rs, cs

    # vertical: drop the gap row next to the beacon on the source's side
    if rs > r4:
        rows = range(r4 + 2, h)              # strip below the gap row
        strip = [(r, c) for r in rows for c in range(w)]
        ops.append(20); sels.append(sel_of(strip))       # slide strip up one
        vacated = [(h - 1, c) for c in range(w)]
        cur_r -= 1
    elif rs < r4:
        rows = range(0, r4 - 1)              # strip above the gap row
        strip = [(r, c) for r in rows for c in range(w)]
        ops.append(21); sels.append(sel_of(strip))       # slide strip down one
        vacated = [(0, c) for c in range(w)]
        cur_r += 1
    else:
        vacated = None
    if vacated is not None and bgc != 0:
        ops.append(bgc); sels.append(sel_of(vacated))    # refill the emptied edge row

    # horizontal: drop the gap column next to the beacon on the source's side
    if cs > c4:
        strip = [(r, c) for r in range(h) for c in range(c4 + 2, w)]
        ops.append(23); sels.append(sel_of(strip))       # slide strip left one
        vacated = [(r, w - 1) for r in range(h)]
        cur_c -= 1
    elif cs < c4:
        strip = [(r, c) for r in range(h) for c in range(0, c4 - 1)]
        ops.append(22); sels.append(sel_of(strip))       # slide strip right one
        vacated = [(r, 0) for r in range(h)]
        cur_c += 1
    else:
        vacated = None
    if vacated is not None and bgc != 0:
        ops.append(bgc); sels.append(sel_of(vacated))    # refill the emptied edge column

    if zero_src:
        ops.append(0); sels.append(sel_of([(cur_r, cur_c)]))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])    # full-grid rectangle: submit
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
