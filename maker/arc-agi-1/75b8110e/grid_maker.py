"""
ARC Task: 75b8110e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import deque


def sample_colors(num_examples=None) -> dict:
    # Only the background is rule-relevant (it is the colour common to all four
    # quadrants, i.e. the "transparent" one).  The four quadrant colours only
    # ride along, so they stay free inside generate().
    return {"bgc": 0}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int = 0) -> dict:
    cols = interval(0, 10, 1)
    hub = max(2, min(15, max_h // 2))
    wub = max(2, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    canv = canvas(bgc, (h, w))
    cels = totuple(asindices(canv))
    mp = (h * w) // 2
    nums = []
    for k in range(4):
        dev = unifint(diff_lb, diff_ub, (0, mp))
        if choice((True, False)):
            num = h * w - dev
        else:
            num = dev
        num = min(max(0, num), h * w - 1)
        nums.append(num)
    s1, s2, s3, s4 = [sample(cels, num) for num in nums]
    gi1 = fill(canv, c1, s1)
    gi2 = fill(canv, c2, s2)
    gi3 = fill(canv, c3, s3)
    gi4 = fill(canv, c4, s4)
    gi = vconcat(hconcat(gi1, gi2), hconcat(gi3, gi4))
    go = fill(gi1, c4, s4)
    go = fill(go, c3, s3)
    go = fill(go, c2, s2)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    h, w = H // 2, W // 2
    ho, wo = O.shape

    # Quadrant origins.  TL is the base layer; the other three are overlays,
    # listed in top-down stacking order (TR covers BL covers BR covers TL).
    tl = (0, 0)
    stack = [(0, w), (h, 0), (h, w)]          # TR, BL, BR

    # Background = the single colour present in ALL four quadrants.
    pals = [set(np.unique(I[r:r + h, c:c + w]).tolist()) for r, c in [tl] + stack]
    bgc = sorted(set.intersection(*pals))[0]

    ops, sels = [], []

    # Canvas becomes the TL quadrant (the base layer of the stack).
    ops.append(33); sels.append([0, 0, h - 1, w - 1])

    if bgc == 0:
        # Background is transparent for Copy/Paste, so each overlay quadrant can
        # be stamped straight onto the base, bottom layer first.
        for (qr, qc) in reversed(stack):      # BR, then BL, then TR on top
            Q = I[qr:qr + h, qc:qc + w]
            if not (Q != bgc).any():
                continue                       # empty layer, nothing to stamp
            ops.append(28); sels.append([qr, qc, h - 1, w - 1])
            ops.append(30); sels.append([0, 0, 0, 0])
    else:
        # Background is opaque: stamp the overlays by painting each layer's
        # visible marks, top layer first so nothing is ever painted twice.
        base = I[:h, :w]
        owner = -np.ones((h, w), dtype=int)
        qcol = {}
        for qi, (qr, qc) in enumerate(stack):
            Q = I[qr:qr + h, qc:qc + w]
            mk = Q != bgc
            if not mk.any():
                continue
            qcol[qi] = int(Q[mk][0])
            owner[mk & (owner < 0)] = qi

        for qi, (qr, qc) in enumerate(stack):
            if qi not in qcol:
                continue
            col = qcol[qi]
            todo = set()
            for r in range(h):
                for c in range(w):
                    if owner[r, c] == qi and int(base[r, c]) != col:
                        todo.add((r, c))
            # emit one connected mark-region at a time
            while todo:
                seed = min(todo)
                todo.discard(seed)
                comp = [seed]
                dq = deque([seed])
                while dq:
                    r, c = dq.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nb = (r + dr, c + dc)
                        if nb in todo:
                            todo.discard(nb)
                            comp.append(nb)
                            dq.append(nb)
                for (r, c) in comp:
                    ops.append(col); sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 75b8110e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 75b8110e"
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
                                f"for task 75b8110e"
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
                    f"Failed to build a complete episode for task 75b8110e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"75b8110e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
