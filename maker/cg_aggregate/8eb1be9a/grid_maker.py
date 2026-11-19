"""
ARC Task: 8eb1be9a (RE-ARC) — LLM-generated grid_maker
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
    # Only the background colour is fixed for the episode. The rule (repeat the
    # object's band of rows up and down the grid) does not depend on which
    # foreground colours the object uses.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    H = max(8, max_h)
    W = max(4, max_w)
    h = unifint(diff_lb, diff_ub, (8, H))
    w = unifint(diff_lb, diff_ub, (4, W))
    oh = unifint(diff_lb, diff_ub, (2, h // 3))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    ncells = unifint(diff_lb, diff_ub, (2, (oh * ow) // 3 * 2))
    cells = random.sample(bounds, ncells)
    mi = min(i for i, _ in cells)
    mj = min(j for _, j in cells)
    cells = [(i - mi, j - mj) for i, j in cells]
    oh = max(i for i, _ in cells) + 1
    ow = max(j for _, j in cells) + 1
    remcols = [c for c in cols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, ncols)
    obj = [(random.choice(ccols), (i, j)) for i, j in cells]
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    obj = [(v, (i + loci, j + locj)) for v, (i, j) in obj]

    gi = [[bgc] * w for _ in range(h)]
    for v, (i, j) in obj:
        gi[i][j] = v
    go = [row[:] for row in gi]
    for k in range(h // oh + 1):
        for s in (-oh * k, oh * k):
            for v, (i, j) in obj:
                ii = i + s
                if 0 <= ii < h:
                    go[ii][j] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading of the task (the maker and the golf entry agree on it): the output
    # repeats rows with period oh = the object's height, aligned to where the
    # object sits. Every row is overlaid with the object row in the same phase.
    # Unit = the object's band (its bounding rows x bounding cols). Copy it from
    # the input once and paste it at every whole period above and below. The band
    # cut off by the top edge is pasted from its visible tail.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Background = the canvas fill colour. The object covers at most ~2/9 of
    # the grid, so it is the strict majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1
    zero_cells = [(r, c) for (r, c) in cells if I[r, c] == 0]  # Paste can't carry 0

    # Full band origins, ordered outward from the original object.
    origins = []
    k = 1
    while True:
        up, dn = r0 - k * oh, r0 + k * oh
        if up < 0 and dn >= h:
            break
        if up >= 0:
            origins.append(up)
        if dn < h:
            origins.append(dn)
        k += 1

    def visible_nonzero(o):
        return any(I[r, c] != 0 and r - r0 + o < h for (r, c) in cells)

    copied = False
    for o in origins:
        if visible_nonzero(o):
            if not copied:
                ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])  # CopyI the whole band rectangle
                copied = True
            ops.append(30); sels.append([o, c0, 0, 0])                # Paste band at its origin
        z = [(r - r0 + o, c) for (r, c) in zero_cells if r - r0 + o < h]
        if z:
            ops.append(0); sels.append(sel_of(z))                     # stamp the 0-coloured cells

    # The band cut off by the top edge: only its lower rows are visible.
    rem = r0 % oh
    if rem != 0:
        src_r = r0 + (oh - rem)  # row of I in the same phase as grid row 0
        if any(I[r, c] != 0 for (r, c) in cells if r >= src_r):
            ops.append(28); sels.append([src_r, c0, r1 - src_r, ow - 1])  # CopyI the visible tail rows
            ops.append(30); sels.append([0, c0, 0, 0])                     # Paste flush with the top
        z = [(r - src_r, c) for (r, c) in zero_cells if r >= src_r]
        if z:
            ops.append(0); sels.append(sel_of(z))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
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
                                f"for task 8eb1be9a"
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
                    f"Failed to build a complete episode for task 8eb1be9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8eb1be9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
