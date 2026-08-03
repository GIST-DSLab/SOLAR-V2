"""
ARC Task: a2fd1cf0 (RE-ARC) — LLM-generated grid_maker
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


ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3, 8)]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rotname": r} for r in ROTS]
        examples += [{"rotname": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rotname": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rotname=None) -> dict:
    if rotname is None:
        rotname = random.choice(ROTS)
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rotname]

    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    g = canvas(bgc, (h, w))
    g = fill(g, 3, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 3))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi + 1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj + 1, w - 1))
    g = fill(g, 2, {(rloci, rlocj)})
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 3))
    go = fill(gi, 8, connect((a, b), (a, d)))
    go = fill(go, 8, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 3, {(c, d)})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    (a,), (b,) = np.where(I == 2)[0], np.where(I == 2)[1]
    (c,), (d,) = np.where(I == 3)[0], np.where(I == 3)[1]

    ops, sels = [], []

    # Horizontal arm: row of the 2, from just past the 2 up to the 3's column
    if d > b:
        c0, c1 = b + 1, d
    else:
        c0, c1 = d, b - 1
    ops.append(8); sels.append([a, c0, 0, c1 - c0])  # exact full rectangle (1 row)

    # Vertical arm: column of the 3, from the corner up to just before the 3
    if c > a:
        r0, r1 = a, c - 1
    else:
        r0, r1 = c + 1, a
    ops.append(8); sels.append([r0, d, r1 - r0, 0])  # exact full rectangle (1 col)

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
                        f"num_examples+1 ({num_examples + 1}) for task a2fd1cf0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a2fd1cf0"
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
                                f"for task a2fd1cf0"
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
                    f"Failed to build a complete episode for task a2fd1cf0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a2fd1cf0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
