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
    cols = list(range(10))
    bgc, barc = random.sample(cols, 2)
    rem = [c for c in cols if c not in (bgc, barc)]
    ncols = random.randint(1, 8)
    return {"bgc": bgc, "barc": barc, "ccols": random.sample(rem, ncols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, ccols=None) -> dict:
    cols = list(range(10))
    if bgc is None or barc is None:
        bgc, barc = random.sample(cols, 2)
    if not ccols:
        rem = [c for c in cols if c not in (bgc, barc)]
        ccols = random.sample(rem, random.randint(1, 8))

    def unif(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))

    h = unif(4, min(30, max_h))
    w = unif(3, max(3, min(14, (max_w - 1) // 2)))
    nl = unif(1, h)
    lines = random.sample(range(h), nl)
    rs = [[bgc] * w for _ in range(h)]
    ls = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for idx in lines:
        j = unif(1, w - 1)
        obj = [random.choice(ccols) for _ in range(j)]
        for jj, c in enumerate(obj):
            go[idx][jj] = c
        slen = random.randint(1, j)
        o2 = obj[:slen]
        if random.choice((True, False)):
            obj, o2 = o2, obj
        for jj, c in enumerate(obj):
            rs[idx][jj] = c
        for jj, c in enumerate(o2):
            ls[idx][jj] = c
    gi = [row[::-1] + [barc] + rr for row, rr in zip(ls, rs)]
    go = [row[::-1] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (from the golf entry): the right half, taken as ONE block, is mirrored
    and laid over the left half -- wherever the mirrored right half has an arm cell it
    shows through, elsewhere the left half stays.  Then only the left half is kept.

    Ops: FlipH the whole right-half rectangle in place; make its background transparent
    (Color0 on its bgc cells, when bgc != 0); CopyO the right half and Paste it onto the
    left half; restore arm cells that are literally colour 0 (Paste cannot write 0);
    crop to the left half."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    w = (wi - 1) // 2          # width of one half; bar at column w
    bgc = int(I[0, 0])         # column 0 is always background (arms are at most w-1 long)
    g = I.copy()
    ops, sels = [], []

    R = I[:, w + 1:]
    Rm = R[:, ::-1]            # right half as it looks after FlipH

    # 1. mirror the whole right half in place (bbox == the full right-half rectangle)
    if not np.array_equal(R, Rm):
        ops.append(26); sels.append([0, w + 1, hi - 1, w - 1])
        g[:, w + 1:] = Rm

    # 2. make the right half's background transparent for the overlay
    if bgc != 0:
        cells = [(r, w + 1 + c) for r in range(hi) for c in range(w) if Rm[r, c] == bgc]
        if cells:
            ops.append(0); sels.append(sel_of(cells))
            for r, c in cells:
                g[r, c] = 0

    # 3. overlay the mirrored right half onto the left half
    clip = g[:, w + 1:].copy()
    after = g.copy()
    nz = clip != 0
    after[:, :w][nz] = clip[nz]
    if not np.array_equal(after, g):
        # bbox == exactly the full right-half rectangle
        ops.append(29); sels.append([0, w + 1, hi - 1, w - 1])
        ops.append(30); sels.append([0, 0, 0, 0])
        g = after

    # 4. arm cells of colour 0 (only possible when bgc != 0) are invisible to Paste
    if bgc != 0:
        cells = [(r, c) for r in range(hi) for c in range(w)
                 if Rm[r, c] == 0 and g[r, c] != 0]
        if cells:
            ops.append(0); sels.append(sel_of(cells))
            for r, c in cells:
                g[r, c] = 0

    # 5. keep the left half (bbox == that full rectangle)
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
