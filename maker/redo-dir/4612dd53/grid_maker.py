"""
ARC Task: 4612dd53 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # fallback: a selection is the exact set of cells it names
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# The three visually distinct structural cases this task can show:
#   a horizontal interior bar, a vertical interior bar, or no bar at all.
VARIANTS = [
    {"horiz": True,  "keep_line": True},
    {"horiz": False, "keep_line": True},
    {"horiz": True,  "keep_line": False},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]          # 2 is the marker colour, never sampled
    bgc, col = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "col": col, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, col, horiz=None, keep_line=None) -> dict:
    if horiz is None:
        horiz = random.choice([True, False])
    if keep_line is None:
        keep_line = random.choice([True, False])

    def uf(lo, hi):
        if hi < lo:
            hi = lo
        return random.randint(lo + int((hi - lo) * diff_lb), lo + int((hi - lo) * diff_ub))

    h = uf(8, max(8, max_h))
    w = uf(8, max(8, max_w))
    ih = uf(5, h - 1)
    iw = uf(5, w - 1)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r1, c1 = loci + ih - 1, locj + iw - 1

    bx = set()
    for j in range(locj, c1 + 1):
        bx.add((loci, j))
        bx.add((r1, j))
    for i in range(loci, r1 + 1):
        bx.add((i, locj))
        bx.add((i, c1))
    corner_cells = [(loci, locj), (loci, c1), (r1, locj), (r1, c1)]

    if horiz:
        locc = random.randint(loci + 2, r1 - 2)
        br = [(locc, j) for j in range(locj + 1, c1)]
        # keep enough bar cells that the bar (not its perpendicular frontiers) stays the
        # smaller frontier set -> generator output and verifier output always agree
        kmin = max(2, iw // ih + 1)
    else:
        locc = random.randint(locj + 2, c1 - 2)
        br = [(i, locc) for i in range(loci + 1, r1)]
        kmin = max(2, -(-ih // iw))

    crns = random.sample(corner_cells, 3)
    rembx = [p for p in bx if p not in set(crns)]
    onbr = random.sample(br, 2)
    rembr = [p for p in br if p not in set(onbr)]

    noccbx = uf(0, len(rembx))
    occbx = random.sample(rembx, noccbx)
    max_nocc = max(0, len(br) - kmin)
    noccbr = min(uf(0, len(rembr)), max_nocc)
    occbr = random.sample(rembr, noccbr)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for (i, j) in bx:
        gi[i][j] = col
        go[i][j] = col
    for (i, j) in br:
        gi[i][j] = col
        go[i][j] = col
    for (i, j) in occbx:
        gi[i][j] = bgc
        go[i][j] = 2
    for (i, j) in occbr:
        gi[i][j] = bgc
        go[i][j] = 2
    if not keep_line:
        for (i, j) in br:
            gi[i][j] = bgc
            go[i][j] = bgc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Rule: reframe on the broken rectangle, complete its outline (and its interior bar,
    if a remnant of one is there) with colour 2, then carry the mended frame back into
    the original picture."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]                      # canvas colour: the majority colour
    col = min(cnt.keys(), key=lambda k: cnt[k])         # the frame colour: the rarer one
    if col == bgc:
        col = [k for k in cnt if k != bgc][0] if len(cnt) > 1 else bgc

    pts = np.argwhere(I == col)
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    ih, iw = r1 - r0 + 1, c1 - c0 + 1

    # 1. Reframe: crop the canvas down to the frame's own region.
    #    bbox selection = exactly that full rectangle, background included.
    ops.append(33); sels.append([r0, c0, ih - 1, iw - 1])
    S = I[r0:r1 + 1, c0:c1 + 1]                         # the working grid after the crop

    # 2. Mend the outline, one side at a time (local coordinates of the cropped grid).
    top    = [(0, j) for j in range(iw) if S[0, j] != col]
    right  = [(i, iw - 1) for i in range(1, ih - 1) if S[i, iw - 1] != col]
    bottom = [(ih - 1, j) for j in range(iw) if S[ih - 1, j] != col]
    left   = [(i, 0) for i in range(1, ih - 1) if S[i, 0] != col]
    for side in (top, right, bottom, left):
        if side:
            ops.append(2); sels.append(sel_of(side))

    # 3. Mend the interior bar, if any of it survives inside the frame.
    interior = [(i, j) for i in range(1, ih - 1) for j in range(1, iw - 1) if S[i, j] == col]
    if interior:
        rws = sorted({i for i, _ in interior})
        cls = sorted({j for _, j in interior})
        if len(rws) == 1 and len(cls) > 1:
            horizontal = True
        elif len(cls) == 1 and len(rws) > 1:
            horizontal = False
        else:                                            # single remnant cell: smaller frontier wins
            horizontal = ih > iw
        if horizontal:
            rr = rws[0]
            line = [(rr, j) for j in range(1, iw - 1) if S[rr, j] != col]
        else:
            cc = cls[0]
            line = [(i, cc) for i in range(1, ih - 1) if S[i, cc] != col]
        if line:
            ops.append(2); sels.append(sel_of(line))

    # 4. Carry the mended frame back into the original picture.
    ops.append(29); sels.append([0, 0, ih - 1, iw - 1])  # CopyO: the whole mended subgrid
    ops.append(31); sels.append([0, 0, ih - 1, iw - 1])  # restore the full-size canvas
    ops.append(30); sels.append([r0, c0, 0, 0])          # paste the frame back where it lives

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
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
                                f"for task 4612dd53"
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
                    f"Failed to build a complete episode for task 4612dd53 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4612dd53-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
