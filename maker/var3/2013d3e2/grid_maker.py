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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # The task rule (take the top-left quadrant of the 4-fold symmetric block)
    # does not depend on which foreground colors are used, only on background
    # vs non-background. Still, fixing the palette keeps the episode coherent.
    cols = list(range(1, 10))
    bgc = choice(tuple(cols))
    palette = [c for c in cols if c != bgc]
    palette = list(sample(tuple(palette), len(palette)))
    return {"bgc": bgc, "palette": palette}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, palette=None) -> dict:
    cols = interval(1, 10, 1)
    if bgc is None:
        bgc = choice(tuple(cols))
    if palette is None:
        palette = [c for c in cols if c != bgc]
        palette = list(sample(tuple(palette), len(palette)))

    # the block placed on the canvas is 2h x 2w, so h is bounded by the canvas
    hmax = min(10, max_h // 2, max_w // 2)
    if hmax < 3:
        hmax = 3
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h

    numcols = unifint(diff_lb, diff_ub, (1, min(8, len(palette))))
    remcols = list(palette[:numcols])

    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {(choice(tuple(remcols)), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = choice(cands)
        obj.add((choice(tuple(remcols)), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)

    # four-fold symmetric block: [ gi | rot90(gi) ; rot270(gi) | rot180(gi) ]
    gi1 = hconcat(gi, rot90(gi))
    gi2 = hconcat(rot270(gi), rot180(gi))
    gi = vconcat(gi1, gi2)

    fullh = unifint(diff_lb, diff_ub, (2 * h, max(2 * h, max_h)))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max(2 * w, max_w)))
    gio = asobject(gi)
    gic = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - 2 * h)
    locj = randint(0, fullw - 2 * w)
    gi = paint(gic, shift(gio, (loci, locj)))

    reminds = difference(asindices(gi), ofcolor(gi, bgc))
    go = lefthalf(tophalf(subgrid(reminds, gi)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # Rule (read off I alone): the non-background cells occupy a 2h x 2w block
    # that is built from one h x w tile and its three rotations. The answer is
    # that tile, i.e. the top-left quadrant of the block.
    #
    # A solver reaches it by narrowing the canvas down to that quadrant, and
    # there is a real choice in HOW to narrow: go straight to the quadrant,
    # first pull the block out of the big canvas and then halve it, or halve
    # one axis before the other (rows first vs columns first). All of these
    # perform the same rule, every crop visibly shrinks the grid, and none of
    # them retraces a state the grid already held. Which one is taken is read
    # off the input (where the block sits and how much of it is coloured),
    # never off the answer grid.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    # Background is the canvas fill = the dominant colour (never 0 in this task,
    # so a transparent CropGrid copy preserves background cells unchanged and
    # no explicit Color(0) repair is ever needed).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ys, xs = np.where(I != bgc)
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    H = rmax - rmin + 1          # == 2h
    W = cmax - cmin + 1          # == 2w
    hh = H // 2                  # h : top half of the block
    ww = W // 2                  # w : left half of the block

    # Route chosen from the input: how far the block sits from the top/left of
    # the canvas plus how densely it is painted. Same input -> same route,
    # regardless of what the answer grid holds.
    painted = int((I != bgc).sum())
    route = (rmin + cmin + painted) % 4

    ops, sels = [], []
    # Every selection below is exactly a full rectangle of the grid (the region
    # being kept, background cells included), which is what CropGrid acts on.
    if route == 0:
        # straight to the quadrant
        ops.append(33); sels.append([rmin, cmin, hh - 1, ww - 1])
    elif route == 1:
        # pull the whole symmetric block off the canvas first, then halve it
        ops.append(33); sels.append([rmin, cmin, H - 1, W - 1])
        ops.append(33); sels.append([0, 0, hh - 1, ww - 1])
    elif route == 2:
        # sweep rows first: keep the block's top half, then its left half
        ops.append(33); sels.append([rmin, cmin, hh - 1, W - 1])
        ops.append(33); sels.append([0, 0, hh - 1, ww - 1])
    else:
        # sweep columns first: keep the block's left half, then its top half
        ops.append(33); sels.append([rmin, cmin, H - 1, ww - 1])
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
