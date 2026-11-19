"""
ARC Task: 8d5021e8 (RE-ARC) — LLM-generated grid_maker
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
    # The output depends only on the input's geometry, so the background is the
    # only colour role that has to stay the same across the episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    # The output is (3h, 2w), so keep h and w small enough to fit max_h/max_w.
    hub = max(1, min(10, max_h // 3))
    wub = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hub))
    w = unifint(diff_lb, diff_ub, (1, wub))
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r, c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    go1 = np.hstack([np.fliplr(gi), gi])
    go2 = np.vstack([go1, np.flipud(go1)])
    go = np.vstack([np.flipud(go1), go2])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Tile-by-tile route: the output is a 3x2 grid of h x w tiles, and each tile
    # is a flipped copy of I:
    #   band 0: [rot180(I) | flipud(I)]
    #   band 1: [fliplr(I) | I        ]
    #   band 2: [rot180(I) | flipud(I)]
    # So: put a copy of I in every tile, then flip each tile in place as needed.
    # Nothing here reads O; every step depends only on I.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 2 * wi
    ops, sels = [], []

    # Track the working grid so we can skip any op that would change nothing.
    g = np.zeros((ho, wo), dtype=int)
    g[:hi, :wi] = I
    nz = I != 0
    has_nz = bool(nz.any())

    if has_nz:
        # Copy I (its nonzero cells) to the clipboard.
        ops.append(28); sels.append([0, 0, hi - 1, wi - 1])
    # Grow the canvas to 3h x 2w. I stays in tile (0,0) and the rest is zero.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    origins = [(0, wi), (hi, 0), (hi, wi), (2 * hi, 0), (2 * hi, wi)]
    if has_nz:
        for (r, c) in origins:
            # Each target tile is all zeros, so a transparent paste lays down
            # an exact copy of I there.
            ops.append(30); sels.append([r, c, 0, 0])
            t = g[r:r + hi, c:c + wi]
            t[nz] = I[nz]

    # Flip each tile in place. The selection is the tile's full rectangle:
    # the whole tile is meant to be transformed, zeros included.
    plan = [((0, 0), "HV"), ((0, wi), "V"), ((hi, 0), "H"),
            ((2 * hi, 0), "HV"), ((2 * hi, wi), "V")]
    for (r, c), flips in plan:
        for f in flips:
            t = g[r:r + hi, c:c + wi]
            nt = np.fliplr(t) if f == "H" else np.flipud(t)
            if np.array_equal(nt, t):
                continue  # this tile is symmetric, so the flip would do nothing
            ops.append(26 if f == "H" else 27)
            sels.append([r, c, hi - 1, wi - 1])
            g[r:r + hi, c:c + wi] = nt

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
                        f"num_examples+1 ({num_examples + 1}) for task 8d5021e8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d5021e8"
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
                                f"for task 8d5021e8"
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
                    f"Failed to build a complete episode for task 8d5021e8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d5021e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
