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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour. The other colours are random
    # scatter, and the rule (mirror and tile) does not depend on them.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + int(round(d * (b - a)))))

    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (1, max(1, min(10, max_h // 3))))
    w = unifint(diff_lb, diff_ub, (1, max(1, min(15, max_w // 2))))
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // max(1, numc))))
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
    # Reading: first mirror every row onto its left, giving the band B = [vmirror(I) | I].
    # Then flip B upside down once to get A = flipud(B). The output is A on top,
    # B in the middle and A at the bottom. A is built once as a unit and copied
    # to the bottom; nothing is built twice and flipped twice.
    # Only I is used here. O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 2 * wi

    ops, sels = [], []
    rect = lambda r, c, h, w: sel_of([(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)])

    # Track the grid state so no op that would leave the grid unchanged gets emitted.
    g = np.zeros((ho, wo), dtype=int)

    # 1. Grow the canvas to the 3h x 2w output frame. I stays at the top left, and its
    #    zeros stay zero.
    ops.append(33); sels.append(rect(0, 0, ho, wo))
    g[:hi, :wi] = I

    # 2. Copy I to the clipboard and paste it to the right of itself in the top band.
    if I.any():
        ops.append(28); sels.append(rect(0, 0, hi, wi))
        ops.append(30); sels.append(sel_of([(0, wi)]))
        g[:hi, wi:] = I
        # 3. Mirror the left copy left to right. Each row now reads reversed then forward.
        if not np.array_equal(np.fliplr(I), I):
            ops.append(26); sels.append(rect(0, 0, hi, wi))
            g[:hi, :wi] = np.fliplr(I)
    B = g[:hi, :].copy()

    if B.any():
        # 4. Copy band B and paste it into the middle band.
        ops.append(29); sels.append(rect(0, 0, hi, wo))
        ops.append(30); sels.append(sel_of([(hi, 0)]))
        g[hi:2 * hi, :] = B
        # 5. Flip the top band upside down so it becomes A = flipud(B).
        A = np.flipud(B)
        if not np.array_equal(A, B):
            ops.append(27); sels.append(rect(0, 0, hi, wo))
            g[:hi, :] = A
            # 6. Copy the finished band A and paste it at the bottom.
            ops.append(29); sels.append(rect(0, 0, hi, wo))
        # If A equals B, the clipboard already holds it.
        ops.append(30); sels.append(sel_of([(2 * hi, 0)]))
        g[2 * hi:, :] = A

    ops.append(34); sels.append(rect(0, 0, ho, wo))
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
