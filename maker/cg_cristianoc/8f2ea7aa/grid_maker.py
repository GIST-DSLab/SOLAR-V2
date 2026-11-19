"""
ARC Task: 8f2ea7aa (RE-ARC) — LLM-generated grid_maker
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
import math
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on which cells of the pattern are background,
    # so only the background colour is fixed for the episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    dmax = max(2, min(5, int(math.isqrt(min(max_h, max_w)))))
    d = _unifint(diff_lb, diff_ub, (2, dmax))
    d2 = d * d
    gi = [[bgc] * d2 for _ in range(d2)]
    go = [[bgc] * d2 for _ in range(d2)]
    inds = [(i, j) for i in range(d) for j in range(d)]
    mp = d2 // 2
    dev = _unifint(diff_lb, diff_ub, (0, mp))
    devs = random.choice((+1, -1))
    num = max(min(mp + devs * dev, d2), 0)
    locs = set(random.sample(inds, num))

    def shp(s):
        if not s:
            return (0, 0)
        rs = [p[0] for p in s]
        cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(locs) != (d, d):
        locs.add(random.choice([p for p in inds if p not in locs]))
    ncols = _unifint(diff_lb, diff_ub, (1, 9))
    cols = random.sample(remcols, ncols)
    minig = [[bgc] * d for _ in range(d)]
    for (i, j) in locs:
        minig[i][j] = random.choice(cols)
    plc = (random.randrange(0, d2, d), random.randrange(0, d2, d))
    for i in range(d):
        for j in range(d):
            gi[plc[0] + i][plc[1] + j] = minig[i][j]
    for (bi, bj) in locs:
        for i in range(d):
            for j in range(d):
                go[bi * d + i][bj * d + j] = minig[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared by the maker and the golf entry): the grid is a d x d
    # array of d x d blocks. The pattern sits in one block-aligned block.
    # Output block (i, j) = the pattern if pattern[i][j] is foreground, else
    # empty. The two readings carve the task the same way. The only
    # refinement here: d comes from the grid size (d*d == side), and the
    # source is the aligned block that holds the pattern. Only that one
    # block is erased, not the whole canvas.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    d = int(math.isqrt(hi))

    nz = np.argwhere(I != bgc)
    sbr, sbc = int(nz[0][0]) // d, int(nz[0][1]) // d   # source block index
    r0, c0 = sbr * d, sbc * d
    pattern = I[r0:r0 + d, c0:c0 + d]

    dests = [(i, j) for i in range(d) for j in range(d) if int(pattern[i, j]) != bgc]
    src_is_dest = (sbr, sbc) in dests
    targets = [(i, j) for (i, j) in dests if (i, j) != (sbr, sbc)]

    # Paste only writes nonzero clip cells. On a bgc canvas it changes
    # something only if the pattern has a cell that is neither 0 nor bgc.
    paste_visible = any(int(v) != 0 and int(v) != bgc for v in pattern.flatten())
    zero_cells = [(ii, jj) for ii in range(d) for jj in range(d)
                  if int(pattern[ii, jj]) == 0] if bgc != 0 else []

    ops, sels = [], []
    # 1. Pick up the pattern block from the input (full d x d rectangle).
    if paste_visible and targets:
        ops.append(28); sels.append([r0, c0, d - 1, d - 1])
    # 2. If the source block is not one of the destinations, clear that
    #    block (full rectangle) back to background.
    if not src_is_dest:
        ops.append(bgc); sels.append([r0, c0, d - 1, d - 1])
    # 3. Stamp the pattern into every block whose pattern cell is foreground.
    #    Then restore the 0-valued pattern cells, which Paste cannot carry.
    for (i, j) in targets:
        if paste_visible:
            ops.append(30); sels.append([i * d, j * d, 0, 0])
        if zero_cells:
            ops.append(0)
            sels.append(sel_of([(i * d + ii, j * d + jj) for (ii, jj) in zero_cells]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
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
                                f"for task 8f2ea7aa"
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
                    f"Failed to build a complete episode for task 8f2ea7aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8f2ea7aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
