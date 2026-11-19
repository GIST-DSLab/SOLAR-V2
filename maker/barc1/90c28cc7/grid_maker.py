"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The frame/background is hardcoded 0. The picture's colours are drawn per instance,
    # and the rule (one pixel per chessboard cell) does not depend on them.
    return {}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    v = a + int(round((b - a) * d))
    return max(a, min(b, v))


def _dedupe_rows(g):
    out = [g[0]]
    for r in g[1:]:
        if r != out[-1]:
            out.append(r)
    return out


def _T(g):
    return [list(x) for x in zip(*g)]


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    cols = list(range(1, 10))
    h = _unifint(diff_lb, diff_ub, (2, min(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (2, min(10, max_w)))
    nc = _unifint(diff_lb, diff_ub, (2, 9))
    colss = random.sample(cols, nc)
    gi = [[random.choice(colss) for _ in range(w)] for _ in range(h)]
    gi = _T(_dedupe_rows(_T(_dedupe_rows(gi))))
    go = [list(r) for r in gi]
    h, w = len(gi), len(gi[0])
    fullh = _unifint(diff_lb, diff_ub, (h, max_h))
    fullw = _unifint(diff_lb, diff_ub, (w, max_w))
    inh = _unifint(diff_lb, diff_ub, (h, fullh))
    inw = _unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            gi = gi[:idx + 1] + [list(r) for r in gi[idx:]]
        else:
            t = _T(gi)
            idx = random.randint(0, w - 1)
            t = t[:idx + 1] + [list(r) for r in t[idx:]]
            gi = _T(t)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [[0] * w]
            gi = z + gi if random.random() < 0.5 else gi + z
        else:
            t = _T(gi)
            z = [[0] * h]
            t = z + t if random.random() < 0.5 else t + z
            gi = _T(t)
        h, w = len(gi), len(gi[0])
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading (reference): inside a frame of 0 rows/cols, the picture is a chessboard of
    single-colour rectangular cells. The output has one pixel per chessboard cell.
    Ops: for each cell (i, j), CopyI its top-left pixel from the input and Paste it into
    slot (i, j) of the output block, which is anchored at the picture's top-left corner.
    Then CropGrid to that block.
    Everything here comes from I alone; O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    core = I[r0:r1 + 1, c0:c1 + 1]
    ch, cw = core.shape

    # Chessboard grid lines: a new cell row (or column) starts wherever it differs
    # from the one before it.
    rows = [0] + [i for i in range(1, ch) if not np.array_equal(core[i], core[i - 1])]
    colsv = [0] + [j for j in range(1, cw) if not np.array_equal(core[:, j], core[:, j - 1])]
    ho, wo = len(rows), len(colsv)

    ops, sels = [], []
    grid = I.copy()   # the working grid as it stands
    clip = None       # colour of the 1x1 pixel currently on the clipboard
    for i, rr in enumerate(rows):
        for j, cc in enumerate(colsv):
            sr, sc = r0 + rr, c0 + cc      # top-left pixel of chessboard cell (i, j)
            dr, dc = r0 + i, c0 + j        # that cell's slot in the output block
            val = int(I[sr, sc])
            if grid[dr, dc] == val:
                continue                   # the slot already shows this colour; a paste would do nothing
            if clip != val:
                ops.append(28); sels.append(sel_of([(sr, sc)]))   # CopyI the cell's pixel
                clip = val
            ops.append(30); sels.append(sel_of([(dr, dc)]))       # Paste it into its slot
            grid[dr, dc] = val

    if (r0, c0, ho, wo) != (0, 0, hi, wi):
        # Crop to the full ho x wo output block. A bbox is correct here because the
        # whole rectangle is meant.
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
