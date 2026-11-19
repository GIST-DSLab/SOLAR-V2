"""
ARC Task: ac0a08a4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is a colour role; foreground colours are just "distinct non-bg cells".
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        return a + round((b - a) * random.uniform(lb, ub))
    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_h))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_w))))
    hi_num = max(1, min(9, h * w - 2, max_h // h, max_w // w))
    num = unifint(diff_lb, diff_ub, (1, hi_num))
    gi = [[bgc] * w for _ in range(h)]
    locs = random.sample([(i, j) for i in range(h) for j in range(w)], num)
    cols = random.sample([c for c in range(10) if c != bgc], num)
    for col, (i, j) in zip(cols, locs):
        gi[i][j] = col
    go = [[gi[i // num][j // num] for j in range(w * num)] for i in range(h * num)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (separable upscale): factor k = number of non-background cells.
    # First every input COLUMN is stretched into k copies (right to left),
    # then every widened ROW is stretched into k copies (bottom to top).
    # O is never consulted: every selection is a column, a row, a band, or a paste point.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    # every foreground colour is unique, so the background is the colour that repeats
    rep = [c for c, n in cnt.items() if n >= 2]
    bgc = rep[0] if len(rep) == 1 else cnt.most_common(1)[0][0]
    k = int(np.sum(I != bgc))
    ho, wo = hi * k, wi * k

    ops, sels = [], []
    g = np.zeros((ho, wo), dtype=int)   # simulated working grid
    g[:hi, :wi] = I

    # 1. expand canvas to the upscaled size (full rectangle)
    if (ho, wo) != (hi, wi):
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    def stretch(get_clip, copy_op, copy_sel, band_of, paste_at, n_units):
        for u in range(n_units - 1, -1, -1):
            clip = get_clip(u)
            ops.append(copy_op); sels.append(copy_sel(u))          # copy this whole line
            r0, c0, bh, bw = band_of(u)                            # its target band (full rectangle)
            band = g[r0:r0 + bh, c0:c0 + bw]
            tiled = np.tile(clip, (bh // clip.shape[0], bw // clip.shape[1]))
            if np.any((band != 0) & (tiled == 0)):
                # band still holds stale content where the line has 0 (paste is
                # transparent for 0) -> clear the whole band before pasting
                band[:] = 0
                ops.append(0); sels.append([r0, c0, bh - 1, bw - 1])
            for j in range(k):
                pr, pc = paste_at(u, j)
                hh, ww = clip.shape
                tgt = g[pr:pr + hh, pc:pc + ww]
                new = np.where(clip != 0, clip, tgt)
                if np.array_equal(new, tgt):
                    continue                                       # paste would change nothing
                tgt[:] = new
                ops.append(30); sels.append([pr, pc, 0, 0])
            if ops[-1] in (28, 29):                                # copy never used -> drop it
                ops.pop(); sels.pop()

    # 2. stretch each input column into k columns (copy from the input, full column)
    stretch(lambda c: I[:, c:c + 1].copy(), 28,
            lambda c: [0, c, hi - 1, 0],
            lambda c: (0, c * k, hi, k),
            lambda c, j: (0, c * k + j), wi)

    # 3. stretch each widened row into k rows (copy the row just produced, full row)
    stretch(lambda r: g[r:r + 1, :].copy(), 29,
            lambda r: [r, 0, 0, wo - 1],
            lambda r: (r * k, 0, k, wo),
            lambda r, j: (r * k + j, 0), hi)

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
                        f"num_examples+1 ({num_examples + 1}) for task ac0a08a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ac0a08a4"
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
                                f"for task ac0a08a4"
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
                    f"Failed to build a complete episode for task ac0a08a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ac0a08a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
