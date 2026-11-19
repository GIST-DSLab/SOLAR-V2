"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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
    bgc, boxc, inc, dotc = random.sample(range(10), 4)
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxc=None, inc=None, dotc=None, **kw) -> dict:
    def unifint(lb, ub):
        lo, hi_ = lb, ub
        if hi_ <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi_ - lo)))

    if None in (bgc, boxc, inc, dotc):
        bgc, boxc, inc, dotc = random.sample(range(10), 4)
    H = max(5, min(30, max_h))
    W = max(4, min(30, max_w))
    h = unifint(5, H)
    w = unifint(4, W)
    hi = unifint(4, h - 1)
    wi = unifint(3, w - 1)
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)

    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + hi, locj:locj + wi] = boxc
    go[loci + 1:loci + hi - 1, locj + 1:locj + wi - 1] = inc

    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:cutoff + 1, locj:locj + wi] = boxc
    gi[loci + 1:cutoff, locj + 1:locj + wi - 1] = inc
    lc = random.randint(locj, locj + wi - 1)
    gi[loci + hi - 1, lc] = dotc

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- find the box (outline + uniform interior) and the lone dot, from I only ---
    boxc = inc = dotc = None
    r0 = c0 = r1 = c1 = None
    for cand in np.unique(I).tolist():
        cells = np.argwhere(I == cand)
        a0, b0 = int(cells[:, 0].min()), int(cells[:, 1].min())
        a1, b1 = int(cells[:, 0].max()), int(cells[:, 1].max())
        bh, bw = a1 - a0 + 1, b1 - b0 + 1
        if bh < 3 or bw < 3 or len(cells) != 2 * bh + 2 * bw - 4:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        if not (np.all(sub[0, :] == cand) and np.all(sub[-1, :] == cand)
                and np.all(sub[:, 0] == cand) and np.all(sub[:, -1] == cand)):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or len(np.unique(inner)) != 1 or int(inner[0, 0]) == cand:
            continue
        ci = int(inner[0, 0])
        rest = [x for x in np.unique(I).tolist() if x != cand and x != ci]
        if len(rest) != 2:
            continue
        singles = [x for x in rest if int((I == x).sum()) == 1]
        if len(singles) != 1:
            continue
        boxc, inc, dotc = cand, ci, singles[0]
        r0, c0, r1, c1 = a0, b0, a1, b1
        break

    rd, cd = [int(v) for v in np.argwhere(I == dotc)[0]]

    # Direction toward the dot: the wall facing it, the cross-section slice next to it,
    # and the slice positions the stretched box newly needs.
    if cd > c1:      # grow right
        wall = [(r, c1) for r in range(r0, r1 + 1)]
        mop, steps = 22, cd - c1
        slice_sel = [r0, c1 - 1, r1 - r0, 0]               # full column slice (rect)
        targets = [(r0, c) for c in range(c1, cd)]
        vertical_slice = True
    elif cd < c0:    # grow left
        wall = [(r, c0) for r in range(r0, r1 + 1)]
        mop, steps = 23, c0 - cd
        slice_sel = [r0, c0 + 1, r1 - r0, 0]
        targets = [(r0, c) for c in range(c0, cd, -1)]
        vertical_slice = True
    elif rd > r1:    # grow down
        wall = [(r1, c) for c in range(c0, c1 + 1)]
        mop, steps = 21, rd - r1
        slice_sel = [r1 - 1, c0, 0, c1 - c0]               # full row slice (rect)
        targets = [(r, c0) for r in range(r1, rd)]
        vertical_slice = False
    else:            # grow up
        wall = [(r0, c) for c in range(c0, c1 + 1)]
        mop, steps = 20, r0 - rd
        slice_sel = [r0 + 1, c0, 0, c1 - c0]
        targets = [(r, c0) for r in range(r0, rd, -1)]
        vertical_slice = False

    ops, sels = [], []

    if boxc != 0 and inc != 0:
        # 1. The facing wall is the same object, translated onto the dot's line.
        ops.append(mop); sels.append(sel_of(wall))          # grab wall
        for _ in range(steps - 1):
            ops.append(mop); sels.append(sel_of([]))        # keep sliding same wall
        # 2. Stretch the body: replicate the box's cross-section slice into every
        #    gap position (including the wall's vacated footprint).
        ops.append(28); sels.append(slice_sel)              # CopyI the cross-section
        for (tr, tc) in targets:
            ops.append(30); sels.append([tr, tc, 0, 0])     # Paste slice
    else:
        # 0 is one of the box colours: Copy/Paste/Move would drop it, so lay each new
        # cross-section slice explicitly (ends = wall colour, middle = interior colour).
        far = (cd if vertical_slice and cd > c1 else
               cd if vertical_slice else rd)
        for (tr, tc) in targets:
            if vertical_slice:
                ops.append(inc); sels.append(sel_of([(r, tc) for r in range(r0 + 1, r1)]))
                ops.append(boxc); sels.append(sel_of([(r0, tc), (r1, tc)]))
            else:
                ops.append(inc); sels.append(sel_of([(tr, c) for c in range(c0 + 1, c1)]))
                ops.append(boxc); sels.append(sel_of([(tr, c0), (tr, c1)]))
        # wall reforms on the dot's line
        if vertical_slice:
            ops.append(boxc); sels.append(sel_of([(r, cd) for r in range(r0, r1 + 1)]))
        else:
            ops.append(boxc); sels.append(sel_of([(rd, c) for c in range(c0, c1 + 1)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
