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
    # The rule (rotate copies of I) never depends on colours; only fix the background.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    lo_i = max(a, min(b, lo_i))
    hi_i = max(lo_i, min(b, hi_i))
    return random.randint(lo_i, hi_i)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(1, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (1, hmax))
    w = h
    gi = np.full((h, w), bgc, dtype=int)
    remcols = [c for c in cols if c != bgc]
    numc = _unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(r, c) for r in range(h) for c in range(w)]
    for col in colsch:
        if not inds:
            break
        num = _unifint(diff_lb, diff_ub, (1, max(1, len(inds) // max(1, numc))))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (r, c) in chos:
            gi[r, c] = col
        chs = set(chos)
        inds = [x for x in inds if x not in chs]
    top = np.hstack([gi, np.rot90(gi, 3)])
    bot = np.hstack([np.rot90(gi, 1), np.rot90(gi, 2)])
    go = np.vstack([top, bot])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the TOP STRIP [I | rot90(I)] is one unit; the bottom strip is that
    # whole strip turned 180 degrees (rot180(hconcat(I, rot90 I)) =
    # hconcat(rot270 I, rot180 I)). So: build the top strip, copy it as one block,
    # paste it below, and rotate that block 180 degrees (FlipH + FlipV).
    I = np.asarray(I, dtype=int)
    h = I.shape[0]  # generator forces w = h -> square
    H = 2 * h
    ops, sels = [], []

    # Python-side simulation used only to skip ops that would not change the grid
    G = np.zeros((H, H), dtype=int)
    G[:h, :h] = I

    def emit(op, sel, newG):
        nonlocal G
        if not np.array_equal(newG, G):
            ops.append(op); sels.append(sel)
            G = newG
            return True
        return False

    # 1. CopyI: the clipboard holds the whole input square (whole rectangle)
    ops.append(28); sels.append([0, 0, h - 1, h - 1])

    # 2. Resize the canvas to 2h x 2h (whole new canvas). I stays top-left.
    ops.append(33); sels.append([0, 0, H - 1, H - 1])

    # 3. Paste I into the top-right square (the region is empty, all 0)
    g = G.copy()
    sub = g[0:h, h:H]
    sub[I != 0] = I[I != 0]
    emit(30, [0, h, 0, 0], g)

    # 4. Rotate that square CW (op25 = rot90), the whole h x h square
    g = G.copy()
    g[0:h, h:H] = np.rot90(G[0:h, h:H], 3)
    emit(25, [0, h, h - 1, h - 1], g)

    # 5. CopyO the finished top strip (rows 0..h-1, all 2h columns) as one unit
    ops.append(29); sels.append([0, 0, h - 1, H - 1])
    strip = G[0:h, :].copy()

    # 6. Paste the strip into the empty bottom half
    g = G.copy()
    sub = g[h:H, :]
    sub[strip != 0] = strip[strip != 0]
    emit(30, [h, 0, 0, 0], g)

    # 7-8. Turn the bottom strip 180 degrees: FlipH then FlipV on the whole strip
    g = G.copy()
    g[h:H, :] = np.fliplr(G[h:H, :])
    emit(26, [h, 0, h - 1, H - 1], g)
    g = G.copy()
    g[h:H, :] = np.flipud(G[h:H, :])
    emit(27, [h, 0, h - 1, H - 1], g)

    # If nothing was pasted (all-zero input), the CopyI/CopyO had no purpose
    if not any(op in (30,) for op in ops):
        ops = [op for op in ops if op not in (28, 29)]
        sels = [s for op, s in zip([o for o in ops], sels)] if False else sels
        # rebuild cleanly
        ops, sels = [33], [[0, 0, H - 1, H - 1]]
    else:
        # drop CopyO if its paste was skipped
        if 29 in ops and ops.index(29) == len(ops) - 1:
            idx = ops.index(29)
            ops.pop(idx); sels.pop(idx)

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
