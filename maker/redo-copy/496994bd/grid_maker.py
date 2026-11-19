"""
ARC Task: 496994bd (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# Task 496994bd
#
# Structure produced by the generator:
#   * a ragged, left-packed pattern block `gix` (h x w) sits in ONE half of the
#     grid, the other half (w columns, optionally with one extra blank column
#     in between) is pure background.
#   * the output mirrors that pattern block across the vertical centre line of
#     the whole grid, i.e. O = I overlaid with vmirror(I).
#   * a final optional vmirror puts the pattern on the right instead of the
#     left; a final optional hmirror flips rows (irrelevant to the rule).
#
# Discrete structural variants that must be covered by the examples:
#   flag  -> gap column between the two halves or not
#   vmir  -> pattern lives on the left half or on the right half
# ----------------------------------------------------------------------------

VARIANTS = [
    {"flag": True,  "vmir": False},
    {"flag": True,  "vmir": True},
    {"flag": False, "vmir": False},
    {"flag": False, "vmir": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))          # generator never uses colour 0
    bgc = random.choice(cols)
    pool = [c for c in cols if c != bgc]
    random.shuffle(pool)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test case was shown

    return {"bgc": bgc, "cols_pool": pool, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, cols_pool=None, flag=None, vmir=None) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(1, 10)))
    if cols_pool is None:
        cols_pool = [c for c in range(1, 10) if c != bgc]
    if flag is None or vmir is None:
        v = random.choice(VARIANTS)
        flag = v["flag"] if flag is None else flag
        vmir = v["vmir"] if vmir is None else vmir

    hub = max(3, min(30, int(max_h)))
    wub = max(3, min(14, (int(max_w) - 1) // 2))

    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, wub))

    numcols = unifint(diff_lb, diff_ub, (1, min(8, len(cols_pool))))
    remcols = list(cols_pool[:numcols])

    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)

    top_cells = totuple(sfilter(bx, lambda ij: ij[0] < h // 2))
    bot_cells = totuple(sfilter(bx, lambda ij: ij[0] > h // 2))
    obj = {
        (random.choice(remcols), random.choice(top_cells)),
        (random.choice(remcols), random.choice(bot_cells)),
    }

    for _ in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = random.choice(cands)
        obj.add((random.choice(remcols), ch))
        bx = bx - {ch}

    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)   # left-pack each row

    gap = canvas(bgc, (h, 1))
    if flag:
        gi = hconcat(gix, canv)
        go = hconcat(gix, vmirror(gix))
    else:
        gi = hconcat(gix, hconcat(gap, canv))
        go = hconcat(gix, hconcat(gap, vmirror(gix)))

    if vmir:
        gi = vmirror(gi)
        go = vmirror(go)
    if random.choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)

    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Mirror the pattern block across the grid's vertical centre line.

    Everything is measured from I only:
      * bgc          = the colour filling the empty half (strict majority of I)
      * pattern bbox = bounding box of the non-background cells of I
      * destination  = that bbox reflected through the vertical centre line,
                       column c  ->  column W-1-c
    The empty half is guaranteed to be background, so the copy lands on a clean
    area; a FlipH in place then turns the pasted copy into its mirror image.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = colour of the blank half (always the strict majority here:
    # one full half plus the ragged gaps of the pattern half are background)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    nz = np.argwhere(I != bgc)
    if nz.size == 0:                       # degenerate: nothing to mirror
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    r0 = int(nz[:, 0].min()); r1 = int(nz[:, 0].max())
    c0 = int(nz[:, 1].min()); c1 = int(nz[:, 1].max())
    bh = r1 - r0
    bw = c1 - c0

    # reflection of the block through the vertical centre line of the grid
    dst_c = wi - 1 - c1

    # 1. grab the whole pattern block from the input (full rectangle intended:
    #    the ragged background cells inside the block are part of the mirror)
    ops.append(28); sels.append([r0, c0, bh, bw])

    # 2. drop it into the empty half at its mirrored column position
    ops.append(30); sels.append([r0, dst_c, 0, 0])

    # 3. mirror that freshly pasted rectangle left<->right in place
    #    (skipped when the block is a single column: the flip would change nothing)
    if bw > 0:
        ops.append(26); sels.append([r0, dst_c, bh, bw])

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
                        f"num_examples+1 ({num_examples + 1}) for task 496994bd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 496994bd"
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
                                f"for task 496994bd"
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
                    f"Failed to build a complete episode for task 496994bd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"496994bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
