"""
ARC Task: 3de23699 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    ncol = random.choice([c for c in cols if c not in (bgc, ccol)])
    return {"bgc": bgc, "ccol": ccol, "ncol": ncol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, ccol: int, ncol: int) -> dict:
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    c = canvas(bgc, (h, w))
    hi = unifint(diff_lb, diff_ub, (4, h))
    wi = unifint(diff_lb, diff_ub, (4, w))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    tmpo = frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)})
    cnds = totuple(backdrop(inbox(tmpo)))
    mp = len(cnds) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    ncnds = choice((dev, len(cnds) - dev))
    ncnds = min(max(0, ncnds), len(cnds))
    ss = sample(cnds, ncnds)
    gi = fill(c, ccol, corners(tmpo))
    gi = fill(gi, ncol, ss)
    go = trim(crop(switch(gi, ccol, ncol), (loci, locj), (hi, wi)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # ---- Rule from I vs O --------------------------------------------------
    # I holds a rectangle marked ONLY by its 4 corner cells (colour ccol);
    # its strict interior is sprinkled with cells of a scatter colour ncol.
    # O is that interior, with the scatter repainted to the corner colour.
    # => find the corner colour: exactly 4 cells, exactly the 4 corners of a
    #    (ho+2) x (wo+2) box, and interior-with-scatter-recoloured == O.
    r0 = c0 = ccol = ncol = None
    for col in [int(v) for v in np.unique(I)]:
        pos = np.argwhere(I == col)
        if len(pos) != 4:
            continue
        rs = sorted({int(p[0]) for p in pos})
        cs = sorted({int(p[1]) for p in pos})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if rs[1] - rs[0] + 1 != ho + 2 or cs[1] - cs[0] + 1 != wo + 2:
            continue
        if {(int(p[0]), int(p[1])) for p in pos} != {(rs[0], cs[0]), (rs[0], cs[1]),
                                                     (rs[1], cs[0]), (rs[1], cs[1])}:
            continue
        rr, cc = rs[0] + 1, cs[0] + 1
        reg = I[rr:rr + ho, cc:cc + wo]
        diff = reg != O
        if diff.any():
            src = {int(v) for v in reg[diff]}
            dst = {int(v) for v in O[diff]}
            if len(src) != 1 or dst != {col}:
                continue
            cand_ncol = src.pop()
        else:
            cand_ncol = None
        r0, c0, ccol, ncol = rr, cc, col, cand_ncol
        break

    ops, sels = [], []

    # 1. Crop the canvas down to the box interior (the corner markers fall away).
    ops.append(33)
    sels.append([r0, c0, ho - 1, wo - 1])

    # 2. Repaint each scatter blob (connected region of ncol) to the corner
    #    colour, one FloodFill per whole blob -- largest blob first.
    if ncol is not None:
        reg = I[r0:r0 + ho, c0:c0 + wo]
        seen = [[False] * wo for _ in range(ho)]
        blobs = []
        for r in range(ho):
            for c in range(wo):
                if seen[r][c] or reg[r, c] != ncol:
                    continue
                stack, cells = [(r, c)], []
                seen[r][c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < ho and 0 <= nx < wo and not seen[ny][nx] \
                                and reg[ny, nx] == ncol:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                blobs.append(cells)
        blobs.sort(key=lambda b: (-len(b), b[0][0], b[0][1]))
        for cells in blobs:
            sr, sc = cells[0]
            ops.append(10 + int(ccol))
            sels.append([sr, sc, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 3de23699"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3de23699"
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
                                f"for task 3de23699"
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
                    f"Failed to build a complete episode for task 3de23699 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3de23699-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
