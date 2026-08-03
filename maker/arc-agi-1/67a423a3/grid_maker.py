"""
ARC Task: 67a423a3 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]          # 4 is reserved marker color
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])

    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    examples = [dict(v) for v in variants]
    examples += [dict(random.choice(variants)) for _ in range(max(0, n_ex - len(variants)))]
    examples = examples[:n_ex]
    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])

    max_h = max(3, min(max_h, 30))
    max_w = max(3, min(max_w, 30))

    h = random.randint(3, max_h)
    w = random.randint(3, max_w)
    lineh = random.randint(1, max(1, h // 3))
    linew = random.randint(1, max(1, w // 3))
    # need loci in [1, h-lineh-1], locj in [1, w-linew-1]
    loci = random.randint(1, h - lineh - 1)
    locj = random.randint(1, w - linew - 1)

    gi = np.full((h, w), bgc, dtype=int)
    # horizontal band (full width)
    gi[loci:loci + lineh, :] = acol
    # vertical band (full height) overwrites at crossing
    gi[:, locj:locj + linew] = bcol

    go = gi.copy()
    # outbox ring around crossing rectangle (one cell outside), filled with 4
    r0, r1 = loci - 1, loci + lineh
    c0, c1 = locj - 1, locj + linew
    go[r0, c0:c1 + 1] = 4
    go[r1, c0:c1 + 1] = 4
    go[r0:r1 + 1, c0] = 4
    go[r0:r1 + 1, c1] = 4

    if mirror:
        gi = gi.T.copy()
        go = go.T.copy()

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # The two crossing lines each fully span their direction: a horizontal band
    # occupies whole rows (no bgc), a vertical band occupies whole cols (no bgc).
    band_rows = [r for r in range(hi) if np.all(I[r, :] != bgc)]
    band_cols = [c for c in range(wi) if np.all(I[:, c] != bgc)]

    r0, r1 = min(band_rows), max(band_rows)   # horizontal band extent
    c0, c1 = min(band_cols), max(band_cols)   # vertical band extent
    # outbox = one cell outside the crossing rectangle
    R0, R1, C0, C1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1

    ops, sels = [], []
    ops.append(4); sels.append([R0, C0, 0, C1 - C0])      # top edge
    ops.append(4); sels.append([R1, C0, 0, C1 - C0])      # bottom edge
    ops.append(4); sels.append([R0, C0, R1 - R0, 0])      # left edge
    ops.append(4); sels.append([R0, C1, R1 - R0, 0])      # right edge

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
                        f"num_examples+1 ({num_examples + 1}) for task 67a423a3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 67a423a3"
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
                                f"for task 67a423a3"
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
                    f"Failed to build a complete episode for task 67a423a3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"67a423a3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
