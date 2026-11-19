"""
ARC Task: c1d99e64 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d + random.uniform(-0.5, 0.5)))))


def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 2]
    nofrontcol = random.choice(colopts)
    noisefrontcol = random.choice([c for c in colopts if c != nofrontcol])
    return {"nofrontcol": nofrontcol, "noisefrontcol": noisefrontcol}


def generate(diff_lb, diff_ub, max_h, max_w, nofrontcol=None, noisefrontcol=None, **kw) -> dict:
    colopts = [c for c in range(10) if c != 2]
    if nofrontcol is None:
        nofrontcol = random.choice(colopts)
    if noisefrontcol is None or noisefrontcol == nofrontcol:
        noisefrontcol = random.choice([c for c in colopts if c != nofrontcol])
    h = _unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    gi = [[nofrontcol] * w for _ in range(h)]
    nhf = _unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    nvf = _unifint(diff_lb, diff_ub, (1, max(1, w // 4)))
    for j in random.sample(range(w), nvf):
        for i in range(h):
            gi[i][j] = noisefrontcol
    for i in random.sample(range(h), nhf):
        for j in range(w):
            gi[i][j] = noisefrontcol
    cands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == nofrontcol]
    kk = len(cands)
    midp = (h * w) // 2
    num_noise = _unifint(diff_lb, diff_ub, (0, max(0, kk - midp - 1)))
    for (i, j) in random.sample(cands, num_noise):
        gi[i][j] = noisefrontcol
    go = [row[:] for row in gi]
    for i in range(h):
        if len(set(gi[i])) == 1 and gi[i][0] == noisefrontcol:
            for j in range(w):
                go[i][j] = 2
    for j in range(w):
        col = set(gi[i][j] for i in range(h))
        if len(col) == 1 and gi[0][j] == noisefrontcol:
            for i in range(h):
                go[i][j] = 2
    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # A frontier is a full row or column of one single colour (visible in I).
    # Each frontier line is recoloured to 2, one line per op.
    for r in range(hi):
        if len(set(I[r, :].tolist())) == 1 and I[r, 0] != 2:
            ops.append(2)
            sels.append(sel_of([(r, c) for c in range(wi)]))
    for c in range(wi):
        if len(set(I[:, c].tolist())) == 1 and I[0, c] != 2:
            ops.append(2)
            sels.append(sel_of([(r, c) for r in range(hi)]))
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c1d99e64"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c1d99e64"
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
                                f"for task c1d99e64"
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
                    f"Failed to build a complete episode for task c1d99e64 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c1d99e64-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
