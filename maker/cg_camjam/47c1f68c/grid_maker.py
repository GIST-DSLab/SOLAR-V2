"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    return {"bgc": bgc, "linc": linc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))
    hmax = max(2, min(14, (max_h - 1) // 2))
    wmax = max(2, min(14, (max_w - 1) // 2))
    h = unifint(2, hmax)
    w = unifint(2, wmax)
    canv = np.full((h, w), bgc, dtype=int)
    nc = unifint(1, h * w - 1)
    bx = {(i, j) for i in range(h) for j in range(w)}
    first = random.choice(sorted(bx))
    obj = {first}
    bx.discard(first)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)
    gi = canv.copy()
    for (i, j) in obj:
        gi[i, j] = objc
    col = np.full((h, 1), linc, dtype=int)
    gi1 = np.hstack([gi, col, canv])
    gi2 = np.hstack([canv, col, canv])
    gi = np.vstack([gi1, np.full((1, 2 * w + 1), linc, dtype=int), gi2])
    go = canv.copy()
    for (i, j) in obj:
        go[i, j] = objc
    go = np.hstack([go, np.fliplr(go)])
    go = np.vstack([go, np.flipud(go)])
    go[go == objc] = linc
    k = random.randrange(4)
    if k & 1:
        gi, go = np.flipud(gi), np.flipud(go)
    if k & 2:
        gi, go = np.fliplr(gi), np.fliplr(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # the divider row / column sit in the middle and split I into four h x w panels
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, 0])                     # divider colour = colour the object takes on

    bgc, src = None, None
    for (r0, c0) in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()                # an empty panel shows the background
        else:
            src = (r0, c0)                  # the one panel carrying the object
    r0, c0 = src
    panel = I[r0:r0 + h, c0:c0 + w]
    objc = [v for v in set(panel.flatten().tolist()) if v != bgc][0]

    ops, sels = [], []

    # 1. every object cell takes the divider colour -- the whole colour class at once
    obj_cells = [(r0 + r, c0 + c) for r in range(h) for c in range(w) if panel[r, c] == objc]
    ops.append(linc); sels.append(sel_of(obj_cells))

    # 2. keep only the object's panel (full rectangle). When the panel already sits
    #    at the top-left corner the grow below drops everything else by itself.
    if (r0, c0) != (0, 0):
        ops.append(33); sels.append([r0, c0, h - 1, w - 1])
    # 3. grow / crop the canvas to the 2h x 2w mosaic (full rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])

    # 4. double the rows: copy the panel, paste it below, mirror the half that is
    #    NOT in the corner the object occupied in I
    ops.append(29); sels.append([0, 0, h - 1, w - 1])              # whole panel
    ops.append(30); sels.append([h, 0, 0, 0])
    if not np.array_equal(panel, panel[::-1, :]):
        fr = h if r0 == 0 else 0
        ops.append(27); sels.append([fr, 0, h - 1, w - 1])         # full half rectangle
    strip = np.vstack([panel, panel[::-1, :]])

    # 5. double the columns: copy the left strip, paste it right, mirror one side
    ops.append(29); sels.append([0, 0, 2 * h - 1, w - 1])          # whole left strip
    ops.append(30); sels.append([0, w, 0, 0])
    if not np.array_equal(strip, strip[:, ::-1]):
        fc = w if c0 == 0 else 0
        ops.append(26); sels.append([0, fc, 2 * h - 1, w - 1])     # full half rectangle

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
