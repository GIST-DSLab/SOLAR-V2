"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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
from dsl import canvas, rot90, rot270, hconcat, fill, asindices, totuple
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    barc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "barc": barc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, barc=1, **kw) -> dict:
    remcols = [c for c in range(10) if c not in (bgc, barc)]
    lim = max(2, min(9, min(max_h, max_w)))
    d = unifint(diff_lb, diff_ub, (2, lim))
    k = 4 if d < 7 else 3
    kmax = max(2, min(k, (max(max_h, max_w) + 1) // (d + 1)))
    num = unifint(diff_lb, diff_ub, (2, kmax))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    patcols = random.sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncells = unifint(diff_lb, diff_ub, (1, d ** 2 - 1))
    for ij in random.sample(inds, ncells):
        c = fill(c, random.choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = random.choice(range(num))
    gi = rot90(rot270(c if fillinidx == 0 else bgcanv))
    go = rot90(rot270(c))
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if random.choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading shared with the maker: the strip is d x d panels split by 1-cell bars,
    # exactly one panel holds the pattern, and every other panel is that SOURCE panel
    # turned clockwise once for each panel step away from it (each panel is built
    # straight from the source, not chained from its neighbour).
    # Where the golf reading differs: the panel two steps away is the source with its
    # rows reversed and its columns reversed (a half turn = mirror left-right + mirror
    # up-down), not two quarter turns. So that panel is built with FlipH + FlipV.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vertical = hi > wi
    d = wi if vertical else hi
    num = (max(hi, wi) + 1) // (d + 1)
    origins = [(j * (d + 1), 0) if vertical else (0, j * (d + 1)) for j in range(num)]
    src, bgc = 0, None
    for j, (r, c) in enumerate(origins):
        p = I[r:r + d, c:c + d]
        if len(np.unique(p)) > 1:
            src = j
        elif bgc is None:
            bgc = int(p[0, 0])
    sr, sc = origins[src]
    P = I[sr:sr + d, sc:sc + d]
    has_zero = bool((P == 0).any())

    ops, sels = [], []
    ops.append(28); sels.append([sr, sc, d - 1, d - 1])   # copy the whole d x d source panel
    for t in range(num):
        if t == src:
            continue
        tr, tc = origins[t]
        full = [tr, tc, d - 1, d - 1]                     # the whole target panel (full rectangle)
        # Paste skips 0 cells, so if the pattern has 0 cells and the background
        # is not 0, clear the panel to 0 first.
        if has_zero and bgc != 0:
            ops.append(0); sels.append(full)
        ops.append(30); sels.append([tr, tc, 0, 0])
        kk = (t - src) % 4
        cur = P.copy()
        if kk == 1:
            if not np.array_equal(np.rot90(cur, 3), cur):
                ops.append(25); sels.append(full)          # quarter turn clockwise
        elif kk == 3:
            if not np.array_equal(np.rot90(cur, 1), cur):
                ops.append(24); sels.append(full)          # quarter turn counter-clockwise
        elif kk == 2:
            # half turn = reverse the columns, then reverse the rows
            if not np.array_equal(np.rot90(cur, 2), cur):
                if not np.array_equal(np.fliplr(cur), cur):
                    ops.append(26); sels.append(full)
                    cur = np.fliplr(cur)
                if not np.array_equal(np.flipud(cur), cur):
                    ops.append(27); sels.append(full)
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
