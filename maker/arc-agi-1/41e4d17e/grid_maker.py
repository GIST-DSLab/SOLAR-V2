"""
ARC Task: 41e4d17e (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 6]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    inds = set((i, j) for i in range(h) for j in range(w))
    box_cells = [(i, j) for i in range(5) for j in range(5) if i in (0, 4) or j in (0, 4)]
    bd_cells = [(i, j) for i in range(5) for j in range(5)]

    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = random.choice(sorted(inds))
        bxs = [(loc[0] + i, loc[1] + j) for (i, j) in box_cells]
        if all(p in inds for p in bxs):
            for (i, j) in bxs:
                gi[i][j] = fgc
                go[i][j] = fgc
            cr, cc = loc[0] + 2, loc[1] + 2
            frns = [(cr, j) for j in range(w)] + [(i, cc) for i in range(h)]
            for (i, j) in frns:
                if go[i][j] == bgc:
                    go[i][j] = 6
            for (i, j) in bd_cells:
                inds.discard((loc[0] + i, loc[1] + j))
            succ += 1
        tr += 1

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- rule detection ---
    # I contains 5x5 hollow squares: a uniform ring of fgc around a uniform 3x3 bgc hole.
    # Each such square emits a full row + full column through its center, painted in the
    # fill colour, but only over background cells (square outlines survive).
    boxes = []
    for r in range(hi - 4):
        for c in range(wi - 4):
            win = I[r:r + 5, c:c + 5]
            ring = np.concatenate([win[0, :], win[4, :], win[1:4, 0], win[1:4, 4]])
            inner = win[1:4, 1:4]
            a = int(ring[0])
            if not bool(np.all(ring == a)):
                continue
            b = int(inner[0, 0])
            if not bool(np.all(inner == b)):
                continue
            if a == b:
                continue
            boxes.append((r + 2, c + 2, b))

    if not boxes:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    hole_cols = [b[2] for b in boxes]
    bgc = max(set(hole_cols), key=hole_cols.count)

    # fill colour = the colour the frontiers take (colour present in O, absent from I)
    new_cols = sorted(set(O.flatten().tolist()) - set(I.flatten().tolist()))
    fill = int(new_cols[0]) if new_cols else 6

    painted = set()

    def emit_run(cells):
        # cells: list of (r, c) forming one contiguous straight stretch
        if not cells:
            return
        r0, c0 = cells[0]
        r1, c1 = cells[-1]
        ops.append(fill)
        sels.append([r0, c0, r1 - r0, c1 - c0])
        painted.update(cells)

    # one square at a time: its horizontal beam, then its vertical beam
    for (cr, cc, _b) in sorted(boxes):
        run = []
        for c in range(wi + 1):
            ok = c < wi and I[cr, c] == bgc and (cr, c) not in painted
            if ok:
                run.append((cr, c))
            else:
                emit_run(run)
                run = []
        run = []
        for r in range(hi + 1):
            ok = r < hi and I[r, cc] == bgc and (r, cc) not in painted
            if ok:
                run.append((r, cc))
            else:
                emit_run(run)
                run = []

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
                        f"num_examples+1 ({num_examples + 1}) for task 41e4d17e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 41e4d17e"
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
                                f"for task 41e4d17e"
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
                    f"Failed to build a complete episode for task 41e4d17e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"41e4d17e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
