"""
ARC Task: d89b689b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # generator samples 6 distinct colors (5 excluded): background, square, and the
    # four satellite pixels. Fix all of them once per episode.
    cols = [c for c in range(10) if c != 5]
    bgc, sqc, a, b, c, d = random.sample(cols, 6)
    return {"bgc": bgc, "sqc": sqc, "a": a, "b": b, "c": c, "d": d}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, a: int, b: int, c: int, d: int) -> dict:
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    loci = randint(1, h - 3)
    locj = randint(1, w - 3)
    canv = canvas(bgc, (h, w))
    go = fill(canv, a, {(loci, locj)})
    go = fill(go, b, {(loci, locj + 1)})
    go = fill(go, c, {(loci + 1, locj)})
    go = fill(go, d, {(loci + 1, locj + 1)})
    inds = totuple(asindices(canv))
    aopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] < locj)
    bopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] > locj + 1)
    copts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] < locj)
    dopts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] > locj + 1)
    aopts = order(aopts, lambda ij: manhattan({ij}, {(loci, locj)}))
    bopts = order(bopts, lambda ij: manhattan({ij}, {(loci, locj + 1)}))
    copts = order(copts, lambda ij: manhattan({ij}, {(loci + 1, locj)}))
    dopts = order(dopts, lambda ij: manhattan({ij}, {(loci + 1, locj + 1)}))
    aidx = unifint(diff_lb, diff_ub, (0, len(aopts) - 1))
    bidx = unifint(diff_lb, diff_ub, (0, len(bopts) - 1))
    cidx = unifint(diff_lb, diff_ub, (0, len(copts) - 1))
    didx = unifint(diff_lb, diff_ub, (0, len(dopts) - 1))
    loca = aopts[aidx]
    locb = bopts[bidx]
    locc = copts[cidx]
    locd = dopts[didx]
    gi = fill(canv, sqc, backdrop({(loci, locj), (loci + 1, locj + 1)}))
    gi = fill(gi, a, {loca})
    gi = fill(gi, b, {locb})
    gi = fill(gi, c, {locc})
    gi = fill(gi, d, {locd})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule read off I: one 2x2 block plus four isolated pixels, each pixel lying
    strictly diagonal to one block corner. Each pixel's colour is transplanted onto
    its nearest (manhattan) block corner; the pixel itself is cleared to background.
    Nothing is read from O."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = dominant colour of I (block=4 cells, pixels=1 cell each)
    bgc = int(np.bincount(I.flatten()).argmax())

    # locate the 2x2 uniform non-background block
    sq_r = sq_c = None
    for r in range(hi - 1):
        for c in range(wi - 1):
            v = int(I[r, c])
            if v != bgc and I[r, c + 1] == v and I[r + 1, c] == v and I[r + 1, c + 1] == v:
                sq_r, sq_c = r, c
                break
        if sq_r is not None:
            break

    corners = [(sq_r, sq_c), (sq_r, sq_c + 1), (sq_r + 1, sq_c), (sq_r + 1, sq_c + 1)]
    block = set(corners)

    # the satellite pixels: every non-background cell outside the block
    sats = [(r, c, int(I[r, c]))
            for r in range(hi) for c in range(wi)
            if int(I[r, c]) != bgc and (r, c) not in block]

    # assign each satellite to its nearest corner, handle them corner by corner
    assigned = []
    for (r, c, col) in sats:
        k = min(range(4), key=lambda i: abs(corners[i][0] - r) + abs(corners[i][1] - c))
        assigned.append((k, r, c, col))
    assigned.sort(key=lambda t: t[0])

    for (k, r, c, col) in assigned:
        tr, tc = corners[k]
        ops.append(col)          # stamp the satellite's colour onto its corner
        sels.append([tr, tc, 0, 0])
        ops.append(bgc)          # remove the satellite it came from
        sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task d89b689b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d89b689b"
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
                                f"for task d89b689b"
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
                    f"Failed to build a complete episode for task d89b689b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d89b689b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
