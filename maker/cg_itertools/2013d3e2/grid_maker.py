"""
ARC Task: 2013d3e2 (RE-ARC) — LLM-generated grid_maker
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
    # Background is the only colour role the rule depends on (never 0 in this task).
    # Foreground colours only form the pattern, so the generator picks them per instance.
    bgc = random.choice(range(1, 10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    if bgc is None:
        bgc = random.choice(range(1, 10))
    cols = list(range(1, 10))
    hmax = max(3, min(10, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)

    canv = np.full((h, w), bgc, dtype=int)
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = {(r, c) for r in range(h) for c in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        dns.add((r + dr, c + dc))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    for (r, c), v in obj.items():
        canv[r, c] = v

    gi = canv
    rot90 = np.rot90(gi, 3)
    rot180 = np.rot90(gi, 2)
    rot270 = np.rot90(gi, 1)
    top = np.hstack([gi, rot90])
    bot = np.hstack([rot270, rot180])
    block = np.vstack([top, bot])

    fullh = unifint(diff_lb, diff_ub, (2 * h, max_h))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max_w))
    fullh = max(2 * h, min(fullh, max_h))
    fullw = max(2 * w, min(fullw, max_w))
    gic = np.full((fullh, fullw), bgc, dtype=int)
    loci = random.randint(0, fullh - 2 * h)
    locj = random.randint(0, fullw - 2 * w)
    gic[loci:loci + 2 * h, locj:locj + 2 * w] = block

    ys, xs = np.where(gic != bgc)
    sub = gic[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sh, sw = sub.shape
    go = sub[:sh // 2, :sw // 2]
    return {"input": gic.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # How the golf solution reads the task: first throw away the empty
    # background margin rows and columns around the pattern, so you are
    # left with the fourfold-symmetric block alone. Then keep that block's
    # top-left quadrant. The maker does both in one crop. This version
    # makes them two visible steps: isolate the block, then halve it.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The background fills the canvas before the block is pasted on.
    # It comes from colours 1..9, so it is never 0, and the transparent
    # crop keeps every cell. When examples are given, use the colour
    # shared by every example's border.
    bgc = None
    if examples:
        cands = None
        for ex_in, _ in examples:
            E = np.asarray(ex_in, dtype=int)
            border = set(E[0].tolist()) | set(E[-1].tolist()) | set(E[:, 0].tolist()) | set(E[:, -1].tolist())
            cands = border if cands is None else cands & border
        if cands and len(cands) == 1:
            bgc = next(iter(cands))
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ys, xs = np.where(I != bgc)
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    H = rmax - rmin + 1
    W = cmax - cmin + 1
    hh, ww = H // 2, W // 2

    ops, sels = [], []
    # Step 1: crop the canvas to the full bounding rectangle of the
    # non-background block. The whole rectangle is meant, including
    # background cells inside it. Skip this step if the block already
    # fills the grid, because the crop would then change nothing.
    if not (rmin == 0 and cmin == 0 and H == hi and W == wi):
        ops.append(33); sels.append([rmin, cmin, H - 1, W - 1])
    # Step 2: the block is four rotated copies of one tile. Crop to its
    # top-left quadrant, the full rectangle starting at the origin.
    ops.append(33); sels.append([0, 0, hh - 1, ww - 1])
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
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
                                f"for task 2013d3e2"
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
                    f"Failed to build a complete episode for task 2013d3e2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2013d3e2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
