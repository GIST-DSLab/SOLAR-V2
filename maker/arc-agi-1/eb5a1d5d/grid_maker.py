"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # Layer colors are episode-wide: shuffled palette, generate() takes the first d.
    # Rule (drop duplicated rows/cols) is structural, but a stable palette keeps the
    # whole episode on one color scheme.
    pool = list(range(10))
    random.shuffle(pool)
    return {"colss_pool": pool}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, colss_pool=None) -> dict:
    if colss_pool is None:
        colss_pool = list(range(10))
        random.shuffle(colss_pool)
    d_ub = min(10, (min(max_h, max_w) + 1) // 2)
    d = unifint(diff_lb, diff_ub, (2, max(2, d_ub)))
    go = canvas(-1, (d * 2 - 1, d * 2 - 1))
    colss = list(colss_pool)[:d]
    for j, cc in enumerate(colss):
        go = fill(go, cc, box(frozenset({(j, j), (2 * d - 2 - j, 2 * d - 2 - j)})))
    nvenl = unifint(diff_lb, diff_ub, (0, max(0, max_h - d)))
    nhenl = unifint(diff_lb, diff_ub, (0, max(0, max_w - d)))
    enl = [nvenl, nhenl]
    gi = tuple(e for e in go)
    while enl[0] > 0 or enl[1] > 0:
        h, w = len(gi), len(gi[0])
        opts = []
        if enl[0] > 0 and h < max_h:
            opts.append((identity, 0))
        if enl[1] > 0 and w < max_w:
            opts.append((dmirror, 1))
        if len(opts) == 0:
            break
        mirrf, ch = choice(opts)
        gi = mirrf(gi)
        idx = randint(0, len(gi) - 1)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirrf(gi)
        enl[ch] -= 1
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # I is the nested-boxes figure with some rows/cols duplicated (stretched).
    # Rule: squeeze out every duplicated row and column -> each ring becomes 1 thick.
    # Realization: 1) crop away the duplicated outer border lines in one go,
    #              2) for each remaining duplicated row/col, slide the block below/right
    #                 of it one cell up/left (the content reappears shifted -> the dup
    #                 line is consumed),
    #              3) crop the canvas down to the squeezed figure.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def blocks(lines):
        bs = []
        for i, ln in enumerate(lines):
            if bs and lines[bs[-1][0]] == ln:
                bs[-1].append(i)
            else:
                bs.append([i])
        return bs

    rb = blocks([tuple(r) for r in I.tolist()])
    cb = blocks([tuple(c) for c in I.T.tolist()])

    # keep the innermost copy of the outermost duplicated band on every side:
    # one crop removes all leading/trailing duplicates at once
    r0, r1 = rb[0][-1], rb[-1][0]
    c0, c1 = cb[0][-1], cb[-1][0]

    G = I.copy()
    if (r0, c0, r1, c1) != (0, 0, hi - 1, wi - 1):
        ops.append(33)
        sels.append([r0, c0, r1 - r0, c1 - c0])
        G = I[r0:r1 + 1, c0:c1 + 1].copy()
    h1, w1 = G.shape

    live_h, live_w = h1, w1

    # interior duplicated rows: pull everything under them up by one
    r = 0
    while r < live_h - 1:
        if np.array_equal(G[r, :live_w], G[r + 1, :live_w]):
            ops.append(20)
            sels.append([r + 1, 0, live_h - 1 - (r + 1), w1 - 1])
            G[r:live_h - 1, :] = G[r + 1:live_h, :]
            G[live_h - 1, :] = 0
            live_h -= 1
        else:
            r += 1

    # interior duplicated columns: pull everything right of them left by one
    c = 0
    while c < live_w - 1:
        if np.array_equal(G[:live_h, c], G[:live_h, c + 1]):
            ops.append(23)
            sels.append([0, c + 1, live_h - 1, live_w - 1 - (c + 1)])
            G[:, c:live_w - 1] = G[:, c + 1:live_w]
            G[:, live_w - 1] = 0
            live_w -= 1
        else:
            c += 1

    # shrink the canvas onto the squeezed figure
    if (live_h, live_w) != G.shape:
        ops.append(33)
        sels.append([0, 0, live_h - 1, live_w - 1])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
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
                                f"for task eb5a1d5d"
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
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
