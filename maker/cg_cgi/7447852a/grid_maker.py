"""
ARC Task: 7447852a (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    cols = [c for c in range(10) if c != 4]
    # after the rotation, w becomes the height and h becomes the width
    w = _unifint(diff_lb, diff_ub, (2, min(8, max_h)))
    h = _unifint(diff_lb, diff_ub, (w + 1, max_w))
    remcols = [c for c in cols if c not in (bgc, linc)]
    g = np.full((h, w), bgc, dtype=int)
    sp = (h - 1, 0)
    g[sp] = linc
    direc = 1
    while True:
        sp = (sp[0] - 1, sp[1] + direc)
        if sp[0] < 0 or not (0 <= sp[1] < w):
            break
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        g[sp] = linc
    gi = np.rot90(g, 3).copy()  # the line starts at the top-left corner
    H, W = gi.shape

    # the background strips, ordered by their leftmost column
    seen = np.zeros((H, W), bool)
    regs = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or gi[r, c] != bgc:
                continue
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and gi[ny, nx] == bgc:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            regs.append(cells)
    regs.sort(key=lambda cs: min(x for _, x in cs))

    inds = [(r, c) for r in range(H) for c in range(W) if gi[r, c] == bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 7))
    ccols = random.sample(remcols, min(numcols, len(remcols)))
    ncells = _unifint(diff_lb, diff_ub, (0, len(inds)))
    for (r, c) in random.sample(inds, ncells):
        gi[r, c] = random.choice(ccols)

    go = gi.copy()
    for cells in regs[0::3]:
        for (r, c) in cells:
            go[r, c] = 4
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # The zigzag line always starts in the top-left corner, so its colour is I[0,0].
    linc = int(I[0, 0])
    ops, sels = [], []
    # Row-by-row reading (the golf reading): the line cuts each row into horizontal runs.
    # A run's strip index comes from k, the number of line cells to its left in that row:
    #   middle rows cross every strip            -> index k
    #   the top row only meets the upper strips  -> index 2k-1
    #   the bottom row only meets the lower strips -> index 2k
    # Every third strip (index % 3 == 0) is painted 4, one row run at a time.
    for r in range(H):
        k = 0
        c = 0
        while c < W:
            if I[r, c] == linc:
                k += 1
                c += 1
                continue
            start = c
            while c < W and I[r, c] != linc:
                c += 1
            if r == 0:
                idx = 2 * k - 1
            elif r == H - 1:
                idx = 2 * k
            else:
                idx = k
            if idx % 3 == 0:
                ops.append(4)
                sels.append(sel_of([(r, x) for x in range(start, c)]))
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 7447852a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7447852a"
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
                                f"for task 7447852a"
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
                    f"Failed to build a complete episode for task 7447852a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7447852a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
