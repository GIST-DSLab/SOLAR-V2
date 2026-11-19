"""
ARC Task: 7fe24cdd (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is held fixed for the episode. The rule is purely
    # geometric, so the foreground colours can vary from one instance to the next.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(max(a + round(d * (b - a)), a), b)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    # The output is 2h x 2h, so h is capped at half the allowed size.
    hmax = max(1, min(15, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = h
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gi[i][j] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    gi = np.array(gi, dtype=int)
    top = np.hstack([gi, np.rot90(gi, 3)])                    # I | rot90(I)
    bot = np.hstack([np.rot90(gi, 1), np.rot90(gi, 2)])       # rot270(I) | rot180(I)
    go = np.vstack([top, bot])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf solution): the output has 4-fold rotational symmetry.
    # The top band is [I | rot90(I)], and the bottom band is that whole top band
    # turned 180 degrees: rot180([I | R]) = [rot180(R) | rot180(I)] = [rot270(I) | rot180(I)].
    # So we build the top band, copy it as one unit, paste it below and turn it 180.
    I = np.asarray(I, dtype=int)
    h = I.shape[0]  # the generator forces w == h, so the input is square
    H = 2 * h
    ops, sels = [], []

    def rect(r, c, hh, ww):
        return [(i, j) for i in range(r, r + hh) for j in range(c, c + ww)]

    # Simulate the working grid so we can leave out any op that would change nothing.
    G = np.zeros((H, H), dtype=int)
    G[:h, :h] = I

    # 1. Resize the canvas to 2h x 2h. I stays in the top-left; the new area is 0.
    ops.append(33); sels.append([0, 0, H - 1, H - 1])  # whole new canvas rectangle

    # 2. Build the top band: copy I and paste it into the top-right quadrant.
    if np.any(I != 0):
        ops.append(28); sels.append(sel_of(rect(0, 0, h, h)))       # CopyI of all of I
        ops.append(30); sels.append([0, h, 0, 0])                   # paste at the top-right origin
        tr = G[:h, h:]
        tr[I != 0] = I[I != 0]
        # 3. Turn the top-right quadrant clockwise to get rot90(I).
        rot = np.rot90(G[:h, h:], 3)
        if not np.array_equal(rot, G[:h, h:]):
            ops.append(25); sels.append(sel_of(rect(0, h, h, h)))   # square quadrant, whole region
            G[:h, h:] = rot

    # 4. Copy the finished top band as one unit and paste it into the bottom band.
    top = G[:h, :].copy()
    if np.any(top != 0):
        ops.append(29); sels.append(sel_of(rect(0, 0, h, H)))       # CopyO of the whole top band
        ops.append(30); sels.append([h, 0, 0, 0])                   # paste at the bottom-band origin
        b = G[h:, :]
        b[top != 0] = top[top != 0]
        # 5. Turn the bottom band 180 degrees (FlipH then FlipV on the whole band).
        f1 = np.fliplr(G[h:, :])
        if not np.array_equal(f1, G[h:, :]):
            ops.append(26); sels.append(sel_of(rect(h, 0, h, H)))   # whole bottom band rectangle
            G[h:, :] = f1
        f2 = np.flipud(G[h:, :])
        if not np.array_equal(f2, G[h:, :]):
            ops.append(27); sels.append(sel_of(rect(h, 0, h, H)))   # whole bottom band rectangle
            G[h:, :] = f2

    ops.append(34); sels.append([0, 0, H - 1, H - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7fe24cdd"
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
                                f"for task 7fe24cdd"
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
                    f"Failed to build a complete episode for task 7fe24cdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7fe24cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
