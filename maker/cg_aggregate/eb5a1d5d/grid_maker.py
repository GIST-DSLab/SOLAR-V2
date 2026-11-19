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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The ring colours are drawn fresh per instance in the original generator and
    # the rule (squeeze out repeated rows/cols) does not depend on any colour role.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    lim = min(max_h, max_w)
    dmax = max(2, min(10, (lim + 1) // 2))
    d = unifint(diff_lb, diff_ub, (2, dmax))
    n = 2 * d - 1
    colss = random.sample(range(10), d)
    go = [[-1] * n for _ in range(n)]
    for j, cc in enumerate(colss):
        a, b = j, 2 * d - 2 - j
        for k in range(a, b + 1):
            go[a][k] = cc
            go[b][k] = cc
            go[k][a] = cc
            go[k][b] = cc
    go = tuple(tuple(r) for r in go)

    nvenl = unifint(diff_lb, diff_ub, (0, max(0, max_h - n)))
    nhenl = unifint(diff_lb, diff_ub, (0, max(0, max_w - n)))
    gi = [list(r) for r in go]
    while nvenl > 0 or nhenl > 0:
        opts = []
        if nvenl > 0 and len(gi) < max_h:
            opts.append(0)
        if nhenl > 0 and len(gi[0]) < max_w:
            opts.append(1)
        if not opts:
            break
        ch = random.choice(opts)
        if ch == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [list(gi[idx])] + gi[idx + 1:]
            nvenl -= 1
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [row[:idx + 1] + [row[idx]] + row[idx + 1:] for row in gi]
            nhenl -= 1
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (shared by the code-golf entry): the figure is the nested-box picture
    # where some rows/cols were repeated. Every line equal to its predecessor is a
    # copy to be squeezed out -- rows first, then columns -- with NO special case
    # for the outer border: a repeated border line is consumed exactly like an
    # interior one, by sliding everything after it back one cell. Copies trailing at
    # the bottom/right edge have nothing after them to slide, so the closing crop
    # drops them.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    G = I.copy()
    live_h, live_w = hi, wi

    # ---- rows: pull the block under each repeated row up by one ----
    r = 0
    while r < live_h - 1:
        if np.array_equal(G[r, :live_w], G[r + 1, :live_w]):
            rest = G[r + 1:live_h, :live_w]
            if all(np.array_equal(rest[k], G[r, :live_w]) for k in range(rest.shape[0])):
                live_h = r + 1          # only trailing copies remain -> crop drops them
                break
            # whole rectangle (rows r+1..live_h-1, all live cols) slides up one
            ops.append(20)
            sels.append([r + 1, 0, live_h - 1 - (r + 1), live_w - 1])
            G[r:live_h - 1, :live_w] = G[r + 1:live_h, :live_w]
            G[live_h - 1, :live_w] = 0
            live_h -= 1
        else:
            r += 1

    # ---- columns: pull the block right of each repeated column left by one ----
    c = 0
    while c < live_w - 1:
        if np.array_equal(G[:live_h, c], G[:live_h, c + 1]):
            rest = G[:live_h, c + 1:live_w]
            if all(np.array_equal(rest[:, k], G[:live_h, c]) for k in range(rest.shape[1])):
                live_w = c + 1
                break
            # whole rectangle (live rows, cols c+1..live_w-1) slides left one
            ops.append(23)
            sels.append([0, c + 1, live_h - 1, live_w - 1 - (c + 1)])
            G[:live_h, c:live_w - 1] = G[:live_h, c + 1:live_w]
            G[:live_h, live_w - 1] = 0
            live_w -= 1
        else:
            c += 1

    # ---- shrink the canvas onto the squeezed figure (drops trailing copies) ----
    if (live_h, live_w) != (hi, wi):
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
