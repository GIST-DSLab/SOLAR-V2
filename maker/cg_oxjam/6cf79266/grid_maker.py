"""
ARC Task: 6cf79266 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colour 0 (the blank cells) and colour 1 (the fill) are fixed in the generator.
    # The noise colours do not affect the rule, so nothing here needs to be fixed.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        b = max(a, b)
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    cols = [c for c in range(10) if c not in (0, 1)]
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    nfgcs = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(cols, nfgcs)
    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (int(0.25 * h * w), int(0.6 * h * w)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, num):
        gi[i][j] = 0
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    cands = [(i, j) for i in range(h - 2) for j in range(w - 2)]
    for (i, j) in random.sample(cands, min(noccs, len(cands))):
        for a in range(3):
            for b in range(3):
                gi[i + a][j + b] = 0
    go = [row[:] for row in gi]
    for i in range(h - 2):
        for j in range(w - 2):
            if all(gi[i + a][j + b] == 0 for a in range(3) for b in range(3)):
                for a in range(3):
                    for b in range(3):
                        go[i + a][j + b] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading: each 3x3 window of the INPUT that is entirely 0 is one unit and
    is stamped with colour 1 (the occurrence pattern). The windows are found
    from I only. A window gets its own Color1 stamp, in raster order of its
    top-left corner. A window is dropped only if the other kept windows
    already cover every one of its cells. Then every stamp has at least one
    cell that no other stamp paints, so no op can be deleted.
    """
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    windows = []
    for r in range(hi - 2):
        for c in range(wi - 2):
            if not I[r:r + 3, c:c + 3].any():
                windows.append((r, c))

    cover = np.zeros((hi, wi), dtype=int)
    for r, c in windows:
        cover[r:r + 3, c:c + 3] += 1

    keep = []
    # Check windows from last to first. A window whose every cell is also
    # covered by another kept window would add nothing, so drop it.
    kept_flags = [True] * len(windows)
    for k in range(len(windows) - 1, -1, -1):
        r, c = windows[k]
        if (cover[r:r + 3, c:c + 3] >= 2).all():
            kept_flags[k] = False
            cover[r:r + 3, c:c + 3] -= 1
    keep = [w for w, f in zip(windows, kept_flags) if f]

    ops, sels = [], []
    for r, c in keep:
        # Whole 3x3 window, all of it blank in I: a full rectangle by design.
        ops.append(1)
        sels.append(sel_of([(r + a, c + b) for a in range(3) for b in range(3)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6cf79266"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6cf79266"
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
                                f"for task 6cf79266"
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
                    f"Failed to build a complete episode for task 6cf79266 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6cf79266-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
