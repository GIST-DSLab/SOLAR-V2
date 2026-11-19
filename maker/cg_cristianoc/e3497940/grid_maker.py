"""
ARC Task: e3497940 (RE-ARC) — LLM-generated grid_maker
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
    bgc, barc = random.sample(range(10), 2)
    return {"bgc": bgc, "barc": barc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if a >= b:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, int(round(a + (b - a) * d))), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None) -> dict:
    if bgc is None or barc is None:
        bgc, barc = random.sample(range(10), 2)
    h = _unifint(diff_lb, diff_ub, (4, max(4, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (3, max(3, min(14, (max_w - 1) // 2))))
    remcols = [c for c in range(10) if c not in (bgc, barc)]
    ncols = _unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    nlinesocc = _unifint(diff_lb, diff_ub, (1, h))
    linesocc = random.sample(range(h), nlinesocc)
    rs = [[bgc] * w for _ in range(h)]
    ls = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for idx in linesocc:
        j = _unifint(diff_lb, diff_ub, (1, w - 1))
        obj = [(random.choice(ccols), (idx, jj)) for jj in range(j)]
        for c, (a, b) in obj:
            go[a][b] = c
        slen = random.randint(1, j)
        obj2 = obj[:slen]
        if random.choice((True, False)):
            obj, obj2 = obj2, obj
        for c, (a, b) in obj:
            rs[a][b] = c
        for c, (a, b) in obj2:
            ls[a][b] = c
    gi = [ls[r][::-1] + [barc] + rs[r] for r in range(h)]
    go = [row[::-1] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading: the whole right half, mirrored, is laid over the left half -- every
    left cell that is background takes the mirrored right cell (both arms share the
    same colours from the bar outward, so where both are coloured they agree).
    The right half is treated as ONE unit rather than row by row:
      1. FlipH the whole right-half rectangle in place (bbox == that full rectangle).
      2. Grab its coloured cells as one object and slide it left by w+1 columns onto
         the left half (one grab, then empty-selection Moves).  ARCLE composites the
         object transparently, so left-arm cells under background stay visible.
      3. ARCLE never writes 0-valued object cells; arm cells of colour 0 are repainted
         at their landing spots with one Color0.
      4. Crop to the left half."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    w = (wi - 1) // 2
    bgc = int(I[0, 0])              # column 0 is never reached by an arm (length <= w-1)
    ops, sels = [], []

    # 1. mirror the right half in place (full rectangle, background included)
    ops.append(26); sels.append([0, w + 1, hi - 1, w - 1])
    G = I.copy()
    G[:, w + 1:] = I[:, w + 1:][:, ::-1]

    # 2. slide the mirrored right arms onto the left half as one object
    arm = [(r, c) for r in range(hi) for c in range(w + 1, wi) if G[r, c] != bgc]
    obj = [(r, c) for r, c in arm if G[r, c] != 0]
    shift = w + 1
    if obj:
        ops.append(23); sels.append(sel_of(obj))          # grab
        for _ in range(shift - 1):
            ops.append(23); sels.append(sel_of([]))       # continue same object
    cur = G.copy()
    for r, c in obj:
        cur[r, c] = 0
    for r, c in obj:
        cur[r, c - shift] = G[r, c]

    # 3. colour-0 arm cells land as whatever was underneath; set them to 0
    zfix = [(r, c - shift) for r, c in arm if G[r, c] == 0 and cur[r, c - shift] != 0]
    if zfix:
        ops.append(0); sels.append(sel_of(zfix))

    # 4. keep the left half (full rectangle)
    ops.append(33); sels.append([0, 0, hi - 1, w - 1])
    ops.append(34); sels.append([0, 0, hi - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e3497940"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e3497940"
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
                                f"for task e3497940"
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
                    f"Failed to build a complete episode for task e3497940 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e3497940-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
