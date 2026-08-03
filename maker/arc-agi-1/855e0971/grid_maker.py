"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    # dot color is the rule-carrying role -> fix it for the whole episode
    dotc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, dotc, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = choice((True, False))
    cols = interval(0, 10, 1)
    # build in bar-rows orientation; dmirror at the end swaps h/w
    hb, wb = (max_w, max_h) if mirrored else (max_h, max_w)
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = max(3, min(nbars, hb // 2))
    lo = 2 * nbars
    h = unifint(diff_lb, diff_ub, (lo, max(lo, hb)))
    w = unifint(diff_lb, diff_ub, (3, max(3, wb)))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = randint(0, nbars - 1)
        barsizes[j] += 1
    gi = tuple()
    go = tuple()
    locs = interval(0, w, 1)
    remcols = remove(dotc, cols)
    lastcol = -1
    nloclbs = [choice((0, 1)) for k in range(len(barsizes))]
    if sum(nloclbs) < 2:
        loc1, loc2 = sample(interval(0, len(nloclbs), 1), 2)
        nloclbs[loc1] = 1
        nloclbs[loc2] = 1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = choice(remove(lastcol, remcols))
        gim = canvas(col, (bs, w))
        gom = canvas(col, (bs, w))
        nl = unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        chlocs = sample(locs, nl)
        for jj in chlocs:
            idx = (randint(0, bs - 1), jj)
            gim = fill(gim, dotc, {idx})
            gom = fill(gom, dotc, vfrontier(idx))
        lastcol = col
        gi = gi + gim
        go = go + gom
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    changed = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != O[r, c]]
    if not changed:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    dotc = int(O[changed[0][0], changed[0][1]])

    # bars run across rows iff every row holds at most 2 distinct colors
    horizontal = all(len(set(I[r].tolist())) < 3 for r in range(hi))
    A = I if horizontal else I.T
    n, m = A.shape

    # bar color of each line (the single non-dot color on it)
    base = []
    for i in range(n):
        vals = [v for v in set(A[i].tolist()) if v != dotc]
        base.append(vals[0] if vals else dotc)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and base[j + 1] == base[i]:
            j += 1
        # one bar spans lines i..j; each dot in it grows to fill the bar
        for b in range(m):
            if any(A[k, b] == dotc for k in range(i, j + 1)):
                # bbox == exactly the intended cells (a full 1-wide strip of the bar)
                if horizontal:
                    sels.append([i, b, j - i, 0])
                else:
                    sels.append([b, i, 0, j - i])
                ops.append(dotc)
        i = j + 1

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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
