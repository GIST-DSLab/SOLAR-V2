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
    # Every ring colour is sampled fresh per instance and the rule ignores colour.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kwargs) -> dict:
    lim = min(max_h, max_w, 30)
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, min(10, (lim + 1) // 2)))
    go = canvas(-1, (d * 2 - 1, d * 2 - 1))
    colss = sample(cols, d)
    for j, cc in enumerate(colss):
        go = fill(go, cc, box(frozenset({(j, j), (2 * d - 2 - j, 2 * d - 2 - j)})))
    nvenl = unifint(diff_lb, diff_ub, (0, max(0, max_h - (2 * d - 1))))
    nhenl = unifint(diff_lb, diff_ub, (0, max(0, max_w - (2 * d - 1))))
    enl = [nvenl, nhenl]
    gi = tuple(e for e in go)
    while (enl[0] > 0 or enl[1] > 0):
        opts = []
        if enl[0] > 0 and len(gi) < max_h:
            opts.append((identity, 0))
        if enl[1] > 0 and len(gi[0]) < max_w:
            opts.append((dmirror, 1))
        if not opts:
            break
        mirrf, ch = choice(opts)
        gi = mirrf(gi)
        idx = randint(0, len(gi) - 1)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirrf(gi)
        enl[ch] -= 1
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading: the figure is the 1-thick nested boxes with some lines repeated.
    # Dedupe lines: walk the columns left->right; every run of k identical
    # columns keeps its FIRST copy, and the whole block to the right of the run
    # slides left k-1 cells over the extra copies (one grab, then empty moves).
    # Then the same for rows, top->bottom (block below slides up).
    # Finally the canvas is cropped to the deduplicated figure.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    G = I.copy()
    live_h, live_w = hi, wi

    A = I.copy()   # what the canvas really shows (Move pastes non-zero cells only)

    def slide(cells, op, steps):
        if all(G[p] == 0 for p in cells):
            # only colour-0 content beyond the run: nothing visible to slide;
            # the 0 ring is repainted at the end
            return
        dr, dc = {20: (-1, 0), 23: (0, -1)}[op]
        snap = A.copy()
        for (r0, c0) in cells:
            snap[r0, c0] = 0
        vals = [(r0, c0, A[r0, c0]) for (r0, c0) in cells]
        A[:, :] = snap
        for (r0, c0, v) in vals:
            if v != 0:
                A[r0 + dr * steps, c0 + dc * steps] = v
        ops.append(op); sels.append(sel_of(cells))     # grab the block
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))    # keep sliding same block

    # columns
    c = 0
    while c < live_w:
        k = 1
        while c + k < live_w and np.array_equal(G[:live_h, c], G[:live_h, c + k]):
            k += 1
        if k > 1 and c + k < live_w:
            block = [(r, cc) for r in range(live_h) for cc in range(c + k, live_w)]
            slide(block, 23, k - 1)
            G[:live_h, c + 1:live_w - (k - 1)] = G[:live_h, c + k:live_w].copy()
            G[:live_h, live_w - (k - 1):live_w] = 0
            live_w -= (k - 1)
        elif k > 1:
            live_w = c + 1   # trailing copies: dropped by the final crop
        c += 1

    # rows
    r = 0
    while r < live_h:
        k = 1
        while r + k < live_h and np.array_equal(G[r, :live_w], G[r + k, :live_w]):
            k += 1
        if k > 1 and r + k < live_h:
            block = [(rr, cc) for rr in range(r + k, live_h) for cc in range(live_w)]
            slide(block, 20, k - 1)
            G[r + 1:live_h - (k - 1), :live_w] = G[r + k:live_h, :live_w].copy()
            G[live_h - (k - 1):live_h, :live_w] = 0
            live_h -= (k - 1)
        elif k > 1:
            live_h = r + 1
        r += 1

    if (live_h, live_w) != (hi, wi):
        # crop canvas to the deduplicated figure (full rectangle intended)
        ops.append(33); sels.append([0, 0, live_h - 1, live_w - 1])
    # Colour 0 is transparent to Move (zeros are never pasted), so wherever
    # the 0-coloured ring slid over duplicate lines those still show through.
    # Repaint the 0 ring: its cells that do not currently read 0.
    leak = [(rr, cc) for rr in range(live_h) for cc in range(live_w)
            if G[rr, cc] == 0 and A[rr, cc] != 0]
    if leak:
        ops.append(0); sels.append(sel_of(leak))
    ops.append(34); sels.append([0, 0, live_h - 1, live_w - 1])
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
