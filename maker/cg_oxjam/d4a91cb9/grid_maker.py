"""
ARC Task: d4a91cb9 (RE-ARC) — LLM-generated grid_maker
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

# Episode plan: which of the four orientations each instance uses (0..3 quarter-turns CW).
ROTATIONS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 4, 8)]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        ex = [{"rot": r} for r in ROTATIONS]
        ex += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTATIONS)
    h = _unifint(diff_lb, diff_ub, (4, max_h))
    w = _unifint(diff_lb, diff_ub, (4, max_w))
    gi = _unifint(diff_lb, diff_ub, (1, h - 1))
    gj = _unifint(diff_lb, diff_ub, (1, w - 1))
    g = np.full((h, w), bgc, dtype=int)
    g[gi, gj] = 8
    g = np.rot90(g, 2)
    pi, pj = [int(v) for v in np.argwhere(g == 8)[0]]
    ri = _unifint(diff_lb, diff_ub, (pi + 1, h - 1))
    rj = _unifint(diff_lb, diff_ub, (pj + 1, w - 1))
    g[ri, rj] = 2
    gi_ = np.rot90(g, -rot).copy()  # rot quarter-turns clockwise
    a, b = [int(v) for v in np.argwhere(gi_ == 2)[0]]
    c, d = [int(v) for v in np.argwhere(gi_ == 8)[0]]
    go = gi_.copy()
    for j in range(min(b, d), max(b, d) + 1):
        go[a, j] = 4
    for i in range(min(a, c), max(a, c) + 1):
        go[i, d] = 4
    go[a, b] = 2
    go[c, d] = 8
    return {"input": gi_.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    # Rule (seen from I only): the 8 and the 2 each send a ray of 4s.
    # The 8 shoots along its column toward the 2's row, ending at the corner
    # (r2, c8). The 2 shoots along its row toward the 8's column and stops
    # when it hits the 8's ray. That makes one L-shaped path joining the two markers.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    r8, c8 = [int(v) for v in np.argwhere(I == 8)[0]]
    r2, c2 = [int(v) for v in np.argwhere(I == 2)[0]]

    ops, sels = [], []

    # Ray 1: from the 8 along column c8, running to the 2's row (corner included).
    step = 1 if r2 > r8 else -1
    ray8 = [(r, c8) for r in range(r8 + step, r2 + step, step)]
    if ray8:
        ops.append(4); sels.append(sel_of(ray8))

    # Ray 2: from the 2 along row r2, running until it meets ray 1 (corner is already painted).
    step = 1 if c8 > c2 else -1
    ray2 = [(r2, c) for c in range(c2 + step, c8, step)]
    if ray2:
        ops.append(4); sels.append(sel_of(ray2))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d4a91cb9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d4a91cb9"
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
                                f"for task d4a91cb9"
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
                    f"Failed to build a complete episode for task d4a91cb9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d4a91cb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
