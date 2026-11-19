"""
ARC Task: b91ae062 (RE-ARC) — LLM-generated grid_maker
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
    # Rule (upscale by numcolors(I)-1) does not depend on any colour role; there is no
    # background and colours are drawn per instance by the generator itself.
    return {}


def generate(diff_lb, diff_ub, max_h, max_w, **kw) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(max(a, int(round(a + (b - a) * d + random.uniform(-0.5, 0.5)))), b)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_h // 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_w // 2))))
    hi_c = min(h * w, 10, max_h // h + 1, max_w // w + 1)
    numc = unifint(diff_lb, diff_ub, (3, max(3, hi_c)))
    ccols = random.sample(cols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    fix = random.sample(inds, numc)
    gi = [[None] * w for _ in range(h)]
    for cc, (i, j) in zip(ccols, fix):
        gi[i][j] = cc
    for i, j in inds:
        if gi[i][j] is None:
            gi[i][j] = random.choice(ccols)
    k = numc - 1
    go = [[gi[r // k][c // k] for c in range(w * k)] for r in range(h * k)]
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O=None, examples=None):
    # Separable reading (vs. the maker's per-cell k x k blocks): first stretch every input
    # COLUMN k-fold sideways (CopyI the column, paste it k times), then stretch every
    # resulting ROW k-fold downward (CopyO the row, paste it k times).
    # k = numcolors(I) - 1 is measured from I alone; O is never consulted.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    k = len(np.unique(I)) - 1
    ho, wo = hi * k, wi * k
    ops, sels = [], []

    # Expand canvas to the output size (I transparent-copied to the top-left, rest 0).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    def paste_effect(clip, r, c):
        h, w = clip.shape
        reg = cur[r:r + h, c:c + w]
        return np.where(clip != 0, clip, reg)

    def lay_strip(clip, r0, c0, dr, dc, skip_self):
        # Destination strip = k copies of the line side by side. Paste cannot write 0, so if
        # stale leftover input content sits under a 0 of the line, clear the strip first
        # (whole rectangle, excluding the source line itself when it lies inside).
        h, w = clip.shape
        if dr:
            rr, cc, hh, ww = r0, c0, k * h, w
        else:
            rr, cc, hh, ww = r0, c0, h, k * w
        region = cur[rr:rr + hh, cc:cc + ww]
        mask_clip = np.tile(clip == 0, (k, 1) if dr else (1, k))
        if skip_self is not None:
            sr, sc, sh, sw = skip_self
            mask_clip[sr - rr:sr - rr + sh, sc - cc:sc - cc + sw] = False
        if np.any(mask_clip & (region != 0)):
            if skip_self is not None:
                if dr:
                    rr2, hh2, cc2, ww2 = rr + h, hh - h, cc, ww
                else:
                    cc2, ww2, rr2, hh2 = cc + w, ww - w, rr, hh
            else:
                rr2, cc2, hh2, ww2 = rr, cc, hh, ww
            if np.any(cur[rr2:rr2 + hh2, cc2:cc2 + ww2] != 0):
                ops.append(0); sels.append([rr2, cc2, hh2 - 1, ww2 - 1])  # full strip rectangle
                cur[rr2:rr2 + hh2, cc2:cc2 + ww2] = 0
        for t in range(k):
            r, c = (r0 + t * h, c0) if dr else (r0, c0 + t * w)
            new = paste_effect(clip, r, c)
            if np.array_equal(new, cur[r:r + h, c:c + w]):
                continue  # this copy is already in place (e.g. the source line itself)
            ops.append(30); sels.append([r, c, 0, 0])
            cur[r:r + h, c:c + w] = new

    # Phase 1: input column j -> columns j*k .. j*k+k-1 (right to left so sources are intact).
    for j in range(wi - 1, -1, -1):
        clip = I[:, j:j + 1].copy()
        ops.append(28); sels.append([0, j, hi - 1, 0])   # CopyI: whole input column (rectangle)
        lay_strip(clip, 0, j * k, 0, 1, (0, 0, hi, 1) if j == 0 else None)

    # Phase 2: stretched row i -> rows i*k .. i*k+k-1 (bottom to top so sources are intact).
    for i in range(hi - 1, -1, -1):
        clip = cur[i:i + 1, :].copy()
        ops.append(29); sels.append([i, 0, 0, wo - 1])   # CopyO: whole stretched row (rectangle)
        lay_strip(clip, i * k, 0, 1, 0, (0, 0, 1, wo) if i == 0 else None)

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
                        f"num_examples+1 ({num_examples + 1}) for task b91ae062"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b91ae062"
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
                                f"for task b91ae062"
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
                    f"Failed to build a complete episode for task b91ae062 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b91ae062-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
