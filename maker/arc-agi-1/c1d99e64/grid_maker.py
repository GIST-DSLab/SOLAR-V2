"""
ARC Task: c1d99e64 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

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
def sample_colors() -> dict:
    import random
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    import numpy as np
    from random import randint, sample as rsample

    dim_lb = 4
    h = randint(dim_lb, max(dim_lb, max_h))
    w = randint(dim_lb, max(dim_lb, max_w))

    gi = np.full((h, w), bgc, dtype=int)

    nhf = randint(1, max(1, h // 4))
    nvf = randint(1, max(1, w // 4))

    rows = rsample(list(range(h)), min(nhf, h))
    cols = rsample(list(range(w)), min(nvf, w))

    for r in rows:
        gi[r, :] = fgc
    for c in cols:
        gi[:, c] = fgc

    bg_cells = [(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc]
    kk = len(bg_cells)
    midp = (h * w) // 2
    max_noise = max(0, kk - midp - 1)
    num_noise = randint(0, max_noise) if max_noise > 0 else 0
    if num_noise > 0:
        noise = rsample(bg_cells, num_noise)
        for r, c in noise:
            gi[r, c] = fgc

    go = gi.copy()
    for r in range(h):
        if np.all(gi[r, :] == gi[r, 0]):
            go[r, :] = 2
    for c in range(w):
        if np.all(gi[:, c] == gi[0, c]):
            go[:, c] = 2

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    for r in range(hi):
        if np.all(I[r, :] == I[r, 0]):
            ops.append(2)
            sels.append([r, 0, 0, wi - 1])

    for c in range(wi):
        if np.all(I[:, c] == I[0, c]):
            ops.append(2)
            sels.append([0, c, hi - 1, 0])

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
            pr_in:  List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in:  List[NDArray] = []
            ex_out: List[NDArray] = []
            ops:  List[int]       = []
            sels: List[List[int]] = []

            # sample color roles once per episode → consistent across all instances
            colors = sample_colors()

            j = 0
            while j < num_examples + 1:
                ok = False
                for _ in range(10):
                    try:
                        r = generate(
                            random.uniform(0.2, 0.5),
                            random.uniform(0.5, 0.8),
                            max_h, max_w,
                            **colors,
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
                    j += 1
                    continue
                if j == num_examples:
                    pr_in.append(I)
                    pr_out.append(O)
                    ops, sels = derive_operations(I, O)
                else:
                    ex_in.append(I)
                    ex_out.append(O)
                j += 1

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c1d99e64-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
