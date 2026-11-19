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
import random
import numpy as np


def sample_colors(num_examples=None) -> dict:
    # The generator picks new ring colours for each instance (sample(cols, d)).
    # There is no separate background colour. The rule removes every repeated
    # row or column, and it works the same whatever the colours are, so
    # nothing needs to be fixed for the episode.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    lim = min(max_h, max_w)
    dmax = max(2, min(10, (lim + 1) // 2))
    d = unifint(diff_lb, diff_ub, (2, dmax))
    n = 2 * d - 1
    go = [[0] * n for _ in range(n)]
    colss = random.sample(range(10), d)
    for j, cc in enumerate(colss):
        for r in range(j, n - j):
            for c in range(j, n - j):
                if r in (j, n - 1 - j) or c in (j, n - 1 - j):
                    go[r][c] = cc

    nv = unifint(diff_lb, diff_ub, (0, max(0, max_h - n)))
    nh = unifint(diff_lb, diff_ub, (0, max(0, max_w - n)))
    gi = [row[:] for row in go]
    while nv > 0 or nh > 0:
        opts = []
        if nv > 0 and len(gi) < max_h:
            opts.append(0)
        if nh > 0 and len(gi[0]) < max_w:
            opts.append(1)
        if not opts:
            break
        ch = random.choice(opts)
        if ch == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [gi[idx][:]] + gi[idx + 1:]
            nv -= 1
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [row[:idx + 1] + [row[idx]] + row[idx + 1:] for row in gi]
            nh -= 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # How this differs from the maker:
    # The golf entry treats "remove repeats" as one act that is the same for
    # every line. It does all rows first, then turns the grid and does the
    # columns, and it treats a border line no differently from an inner one.
    # The maker instead crops the outer repeated lines away first, as a
    # special case, and only then pulls the inner ones.
    # Here every repeat, on the border or inside, is removed the same way.
    # If a row equals the row just below it, the block under it is slid up
    # one cell, so the repeat is used up. Columns get the same treatment,
    # sliding left. At the end the canvas is cropped to what is left.
    #
    # Only I is used. Every selection is a full rectangle (a block of rows or
    # columns, background included) named from the grid as it stands.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    G = I.copy()
    live_h, live_w = hi, wi

    # rows: pull the block below each repeated row up by one cell
    r = 0
    while r < live_h - 1:
        if np.array_equal(G[r, :live_w], G[r + 1, :live_w]):
            N = G.copy()
            N[r:live_h - 1, :live_w] = G[r + 1:live_h, :live_w]
            N[live_h - 1, :live_w] = 0
            if not np.array_equal(N, G):  # an all-zero block would change nothing
                ops.append(20)  # MoveU on the whole rectangle of rows r+1..live_h-1
                sels.append([r + 1, 0, live_h - 1 - (r + 1), live_w - 1])
            G = N
            live_h -= 1
        else:
            r += 1

    # columns: pull the block right of each repeated column left by one cell
    c = 0
    while c < live_w - 1:
        if np.array_equal(G[:live_h, c], G[:live_h, c + 1]):
            N = G.copy()
            N[:live_h, c:live_w - 1] = G[:live_h, c + 1:live_w]
            N[:live_h, live_w - 1] = 0
            if not np.array_equal(N, G):
                ops.append(23)  # MoveL on the whole rectangle of cols c+1..live_w-1
                sels.append([0, c + 1, live_h - 1, live_w - 1 - (c + 1)])
            G = N
            live_w -= 1
        else:
            c += 1

    # crop the canvas to the figure that is left (a full rectangle)
    if (live_h, live_w) != (hi, wi):
        ops.append(33)
        sels.append([0, 0, live_h - 1, live_w - 1])

    ops.append(34)
    sels.append([0, 0, live_h - 1, live_w - 1])
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
