"""
ARC Task: 9ecd008a (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette[:8]}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, palette=None) -> dict:
    # final grid is 2h x 2w, so cap h by half of the allowed dims
    hub = min(15, max_h // 2, max_w // 2)
    hub = max(3, hub)
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = h
    if palette is None:
        palette = [c for c in range(1, 10) if c != bgc]
    remcols_all = [c for c in palette if c != bgc]
    if len(remcols_all) == 0:
        remcols_all = [c for c in range(1, 10) if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols_all))))
    remcols = remcols_all[:numcols]

    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = choice(cands)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))

    locidev = unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # the hole is the rectangular block of 0s punched out of a doubly-symmetric grid
    zr, zc = np.where(I == 0)
    r0, c0 = int(zr.min()), int(zc.min())
    hh = int(zr.max()) - r0 + 1
    ww = int(zc.max()) - c0 + 1

    # mirroring the whole grid brings the missing content onto the hole's coordinates;
    # pick the mirror that leaves fewest 0s there (hole may overlap its own mirror image)
    vm = np.fliplr(I)
    hm = np.flipud(I)
    reg_v = vm[r0:r0 + hh, c0:c0 + ww]
    reg_h = hm[r0:r0 + hh, c0:c0 + ww]
    if int((reg_v == 0).sum()) <= int((reg_h == 0).sum()):
        flip_op = 26          # FlipH = fliplr
        region = reg_v
    else:
        flip_op = 27          # FlipV = flipud
        region = reg_h

    ops, sels = [], []
    # full-grid bbox: intended cells ARE the whole rectangle
    ops.append(flip_op); sels.append([0, 0, hi - 1, wi - 1])
    # crop down to the hole's window, which now holds the recovered content
    ops.append(33); sels.append([r0, c0, hh - 1, ww - 1])

    # CropGrid is a transparent copy: any 0 inside the window was not carried over
    for i in range(hh):
        for j in range(ww):
            if region[i, j] == 0:
                ops.append(0); sels.append([i, j, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
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
                                f"for task 9ecd008a"
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
                    f"Failed to build a complete episode for task 9ecd008a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9ecd008a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
