"""
ARC Task: d9fac9be (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec, ringc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "ringc": ringc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, noisec: int, ringc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    gi = canvas(bgc, (h, w))
    nnoise1 = unifint(diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = asindices(gi)
    noise1 = sample(totuple(inds), nnoise1)
    noise2 = sample(difference(totuple(inds), noise1), nnoise2)
    gi = fill(gi, noisec, noise1)
    gi = fill(gi, ringc, noise2)
    rng = neighbors((1, 1))
    fp1 = recolor(noisec, rng)
    fp2 = recolor(ringc, rng)
    fp1occ = occurrences(gi, fp1)
    fp2occ = occurrences(gi, fp2)
    for occ1 in fp1occ:
        loc = choice(totuple(shift(rng, occ1)))
        gi = fill(gi, choice((bgc, ringc)), {loc})
    for occ2 in fp2occ:
        loc = choice(totuple(shift(rng, occ2)))
        gi = fill(gi, choice((bgc, noisec)), {loc})
    loci = randint(0, h - 3)
    locj = randint(0, w - 3)
    ringp = shift(rng, (loci, locj))
    gi = fill(gi, ringc, ringp)
    gi = fill(gi, noisec, {(loci + 1, locj + 1)})
    go = canvas(noisec, (1, 1))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # Rule (read off I alone): exactly one 3x3 window has a uniform non-background
    # 8-cell ring surrounding a differently-colored non-background center.
    # The answer is that center pixel itself -> crop the grid down to it.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cr, cc = 0, 0
    found = False
    for r in range(1, hi - 1):
        for c in range(1, wi - 1):
            ctr = int(I[r, c])
            if ctr == bgc:
                continue
            ring = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    ring.add(int(I[r + dr, c + dc]))
            if len(ring) != 1:
                continue
            rc = ring.pop()
            if rc == bgc or rc == ctr:
                continue
            cr, cc, found = r, c, True
            break
        if found:
            break

    ops, sels = [], []
    # CropGrid onto the single ring-center cell: that pixel becomes the whole grid.
    ops.append(33)
    sels.append([cr, cc, 0, 0])
    ops.append(34)
    sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
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
                                f"for task d9fac9be"
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
                    f"Failed to build a complete episode for task d9fac9be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d9fac9be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
