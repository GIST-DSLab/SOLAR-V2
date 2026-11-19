"""
ARC Task: 9172f3a0 (RE-ARC) — LLM-generated grid_maker
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
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    hmax = max(1, min(10, max_h // 3))
    wmax = max(1, min(10, max_w // 3))
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    gi = canvas(bgc, (h, w))
    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = upscale(gi, 3)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading taken from the golf entry: the upscale is separable, not a set of
    # 3x3 blocks. First every input column becomes 3 equal columns, then every
    # widened row becomes 3 equal rows. Only I is used; O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    ops, sels = [], []

    # Make the canvas 3x larger. I stays at the top-left and the rest is 0.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I
    clip = None  # (h, w, content) of the current clipboard

    def paste_effect(content, r0, c0):
        h, w = content.shape
        region = cur[r0:r0 + h, c0:c0 + w]
        return np.where(content != 0, content, region)

    def stretch(src, copy_op, copy_sel, targets):
        # src: the strip to copy (a column or a row); targets: where it lands
        nonlocal clip
        h, w = src.shape
        planned = []
        for (r0, c0) in targets:
            new = paste_effect(src, r0, c0)
            if not np.array_equal(new, cur[r0:r0 + h, c0:c0 + w]):
                planned.append((r0, c0))
        if planned:
            key = (h, w, src.tobytes())
            if clip != key:
                ops.append(copy_op); sels.append(copy_sel)
                clip = key
        for (r0, c0) in targets:
            if (r0, c0) in planned:
                ops.append(30); sels.append([r0, c0, 0, 0])
                cur[r0:r0 + h, c0:c0 + w] = paste_effect(src, r0, c0)
            # Paste does not write 0s from the strip, so clear the landing
            # cells that should be 0 but still hold old content.
            zero_fix = [(r0 + i, c0 + j) for i in range(h) for j in range(w)
                        if src[i, j] == 0 and cur[r0 + i, c0 + j] != 0]
            if zero_fix:
                ops.append(0); sels.append(sel_of(zero_fix))
                for (r, c) in zero_fix:
                    cur[r, c] = 0

    # Phase 1: turn each input column into three columns. Go right to left so
    # no column is overwritten before it has been copied.
    for c in range(wi - 1, -1, -1):
        src = I[:, c:c + 1].copy()
        stretch(src, 28, [0, c, hi - 1, 0],                # CopyI one input column
                [(0, 3 * c + k) for k in range(3)])

    # Phase 2: turn each widened row into three rows. Go bottom to top.
    for r in range(hi - 1, -1, -1):
        src = cur[r:r + 1, :].copy()
        stretch(src, 29, [r, 0, 0, wo - 1],                 # CopyO one widened row
                [(3 * r + k, 0) for k in range(3)])

    # Remove any op the rest of the trajectory does not need (for example a
    # copy whose strip the clipboard already holds). The target is the rule's
    # own result, upscale(I, 3), computed from I alone.
    target = np.repeat(np.repeat(I, 3, axis=0), 3, axis=1)
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            res = _sim(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:], ho, wo)
            if res is not None and np.array_equal(res, target):
                del ops[k]; del sels[k]
                changed = True
                break

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
    return ops, sels


def _sim(I, ops, sels, ho, wo):
    # Small replay of the ops used above (Color, CopyI, CopyO, Paste, Resize).
    hi, wi = I.shape
    g = np.zeros((30, 30), dtype=int)
    g[:hi, :wi] = I
    dim = (hi, wi)
    clip = None
    for op, sel in zip(ops, sels):
        if isinstance(sel, dict):
            cells = [tuple(x) for x in sel["cells"]]
            r0, c0, h, w = None, None, None, None
        else:
            r0, c0, h, w = sel
            cells = [(r, c) for r in range(r0, r0 + h + 1) for c in range(c0, c0 + w + 1)]
        if op <= 9:
            for r, c in cells:
                g[r, c] = op
        elif op == 33:
            sub = g[r0:r0 + h + 1, c0:c0 + w + 1].copy()
            g[:] = 0
            g[:h + 1, :w + 1] = sub
            dim = (h + 1, w + 1)
        elif op in (28, 29):
            src = np.zeros((30, 30), dtype=int)
            src[:hi, :wi] = I
            base = src if op == 28 else g
            clip = base[r0:r0 + h + 1, c0:c0 + w + 1].copy()
        elif op == 30:
            if clip is None:
                continue
            ch, cw = clip.shape
            reg = g[r0:r0 + ch, c0:c0 + cw]
            cl = clip[:reg.shape[0], :reg.shape[1]]
            g[r0:r0 + ch, c0:c0 + cw] = np.where(cl != 0, cl, reg)
    if dim != (ho, wo):
        return None
    return g[:ho, :wo].copy()


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
                        f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
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
                                f"for task 9172f3a0"
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
                    f"Failed to build a complete episode for task 9172f3a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9172f3a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
