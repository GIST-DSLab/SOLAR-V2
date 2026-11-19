"""
ARC Task: ac0a08a4 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (2, min(5, max_h)))
    w = unifint(diff_lb, diff_ub, (2, min(5, max_w)))
    num = unifint(diff_lb, diff_ub, (1, min(min(9, h * w - 2), min(max_h // h, max_w // w))))
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = sample(totuple(inds), num)
    remcols = [x for x in cols if x != bgc]
    obj = {(col, loc) for col, loc in zip(sample(remcols, num), locs)}
    gi = paint(c, obj)
    go = upscale(gi, num)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: upscale = stretch every ROW k times, then every COLUMN k times,
    # with k = number of non-background cells in I (bg is the only repeated colour).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    k = int(np.sum(I != bgc))
    ho, wo = hi * k, wi * k
    ops, sels = [], []
    clip = None

    if k > 1:
        # 1. expand canvas to the stretched size (I stays top-left, rest is 0)
        G = np.zeros((ho, wo), dtype=int)
        G[:hi, :wi] = I
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

        # 2. vertical stretch: input row r fills output rows r*k .. r*k+k-1
        for r in range(hi):
            band = range(r * k, r * k + k)
            # cells in this band still holding stale input pixels where row r has 0
            stale = [(t, c) for t in band for c in range(wi)
                     if I[r, c] == 0 and G[t, c] != 0]
            if stale:
                ops.append(0); sels.append(sel_of(stale))
                for t, c in stale:
                    G[t, c] = 0
            m = I[r] != 0
            for t in band:
                new = G[t, :wi].copy()
                new[m] = I[r][m]
                if (new != G[t, :wi]).any():
                    if clip is None or clip.shape != (1, wi) or (clip[0] != I[r]).any():
                        ops.append(28); sels.append([r, 0, 0, wi - 1])   # CopyI input row r
                        clip = I[r:r + 1].copy()
                    ops.append(30); sels.append([t, 0, 0, 0])            # paste row
                    G[t, :wi] = new

        # 3. horizontal stretch, right to left so sources survive until copied:
        #    stretched column c fills output columns c*k .. c*k+k-1
        for c in range(wi - 1, -1, -1):
            src = G[:, c].copy()
            band = range(c * k, c * k + k)
            stale = [(r, t) for t in band for r in range(ho)
                     if src[r] == 0 and G[r, t] != 0]
            if stale:
                ops.append(0); sels.append(sel_of(stale))
                for r, t in stale:
                    G[r, t] = 0
            m = src != 0
            for t in band:
                new = G[:, t].copy()
                new[m] = src[m]
                if (new != G[:, t]).any():
                    if clip is None or clip.shape != (ho, 1) or (clip[:, 0] != src).any():
                        ops.append(29); sels.append([0, c, ho - 1, 0])   # CopyO stretched column
                        clip = src.reshape(ho, 1).copy()
                    ops.append(30); sels.append([0, t, 0, 0])            # paste column
                    G[:, t] = new

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
                        f"num_examples+1 ({num_examples + 1}) for task ac0a08a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ac0a08a4"
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
                                f"for task ac0a08a4"
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
                    f"Failed to build a complete episode for task ac0a08a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ac0a08a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
