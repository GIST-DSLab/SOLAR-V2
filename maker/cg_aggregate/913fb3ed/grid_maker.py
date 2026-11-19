"""
ARC Task: 913fb3ed (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# In this generator, each source colour always gets the same ring colour.
SRC_TO_RING = {2: 1, 3: 6, 8: 4}


def sample_colors(num_examples=None) -> dict:
    # Only the background is sampled at random. The (centre, ring) colour pairs are hardcoded.
    bgc = random.choice([0, 5, 7, 9])
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    if bgc is None:
        bgc = random.choice([0, 5, 7, 9])
    prs = [(2, 1), (3, 6), (8, 4)]
    h = unifint(diff_lb, diff_ub, (1, max_h))
    w = unifint(diff_lb, diff_ub, (1, max_w))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    numc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    inds = {(r, c) for r in range(h) for c in range(w)}
    for _ in range(numc):
        if not inds:
            break
        loc = random.choice(sorted(inds))
        a, b = random.choice(prs)
        r0, c0 = loc
        # Remove the 5x5 block around loc. This keeps rings from overlapping.
        inds = {(r, c) for (r, c) in inds
                if not (abs(r - r0) <= 2 and abs(c - c0) <= 2)}
        gi[r0][c0] = a
        go[r0][c0] = a
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    go[rr][cc] = b
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Read the rule as two growth steps:
    #   1) each centre grows its ring colour one cell left and one cell right,
    #      making a 1x3 strip;
    #   2) that strip grows one cell up and one cell down. This adds the top
    #      and bottom rows of the ring, corners included.
    # Each step is done for all centres of one colour at once. Every selection
    # comes from the input centres or the strip painted in the step before.
    # The generator places centres at least 3 cells apart (Chebyshev), so the
    # rings never overlap. Every cell painted here is still background.
    for src in (2, 3, 8):
        ring = SRC_TO_RING[src]
        centers = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == src]
        if not centers:
            continue

        # Step 1: grow sideways. Select the cells left and right of each centre.
        arms = []
        for r, c in centers:
            for cc in (c - 1, c + 1):
                if 0 <= cc < wi:
                    arms.append((r, cc))
        if arms:
            ops.append(ring)
            sels.append(sel_of(sorted(set(arms))))

        # Step 2: grow the strip (centre plus arms) up and down.
        ext = []
        for r, c in centers:
            strip = [(r, cc) for cc in (c - 1, c, c + 1) if 0 <= cc < wi]
            for (sr, sc) in strip:
                for rr in (sr - 1, sr + 1):
                    if 0 <= rr < hi:
                        ext.append((rr, sc))
        if ext:
            ops.append(ring)
            sels.append(sel_of(sorted(set(ext))))

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
                        f"num_examples+1 ({num_examples + 1}) for task 913fb3ed"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 913fb3ed"
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
                                f"for task 913fb3ed"
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
                    f"Failed to build a complete episode for task 913fb3ed "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"913fb3ed-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
