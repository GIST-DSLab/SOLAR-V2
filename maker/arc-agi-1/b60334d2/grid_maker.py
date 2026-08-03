"""
ARC Task: b60334d2 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        v = a + (b - a) * random.uniform(lb, ub)
        return max(a, min(b, int(round(v))))

    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 9))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]

    inds = {(r, c) for r in range(h) for c in range(w)}
    # box of 3x3 = ring cells; backdrop = full 3x3 block
    bx = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    bd = {(r, c) for r in range(3) for c in range(3)}

    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = random.choice(sorted(inds))
        bxs = {(r + loc[0], c + loc[1]) for r, c in bx}
        if bxs.issubset(inds):
            cr, cc = loc[0] + 1, loc[1] + 1
            gi[cr][cc] = fgc
            for r, c in [(cr - 1, cc - 1), (cr - 1, cc + 1), (cr + 1, cc - 1), (cr + 1, cc + 1)]:
                go[r][c] = fgc
            for r, c in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
                go[r][c] = 1
            inds -= {(r + loc[0], c + loc[1]) for r, c in bd}
            succ += 1
        tr += 1

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = dominant color of the sparse-dot input
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # foreground dot color: the only other color present in I
    others = [c for c in set(I.flatten().tolist()) if c != bgc]
    fgc = others[0] if others else None

    if fgc is not None:
        # each dot in I becomes a 3x3 ring: corners fgc, edge-midpoints 1, centre back to bgc
        centers = sorted((r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc)
        for (cr, cc) in centers:
            edges = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
            corners = [(cr - 1, cc - 1), (cr - 1, cc + 1), (cr + 1, cc - 1), (cr + 1, cc + 1)]
            edges = [(r, c) for r, c in edges if 0 <= r < hi and 0 <= c < wi]
            corners = [(r, c) for r, c in corners if 0 <= r < hi and 0 <= c < wi]

            if corners:
                ops.append(int(fgc)); sels.append(sel_of(corners))
            if edges:
                ops.append(1); sels.append(sel_of(edges))
            # the dot itself is consumed by the ring it spawns
            ops.append(int(bgc)); sels.append(sel_of([(cr, cc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task b60334d2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b60334d2"
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
                                f"for task b60334d2"
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
                    f"Failed to build a complete episode for task b60334d2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b60334d2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
