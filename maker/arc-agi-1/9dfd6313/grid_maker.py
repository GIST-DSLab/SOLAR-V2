"""
ARC Task: 9dfd6313 (RE-ARC) — LLM-generated grid_maker
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

# Discrete structural variants = mirror-line orientation (generator's lni 1..4).
# 1: horizontal line -> hmirror(flipud)   2: vertical line -> vmirror(fliplr)
# 3: main diagonal   -> dmirror(transpose) 4: anti-diagonal -> cmirror(anti-transpose)
VARIANTS = [1, 2, 3, 4]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(VARIANTS):
        examples = list(VARIANTS)
        examples += [random.choice(VARIANTS) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = random.sample(VARIANTS, n_ex)
    plan = [{"lni": v} for v in examples]
    plan.append({"lni": random.choice(examples)})  # test case
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, lni=None) -> dict:
    cols = interval(0, 10, 1)
    lim = (min(max_h, max_w) - 1) // 2
    dh_ub = min(14, max(1, lim))
    dh = unifint(diff_lb, diff_ub, (1, dh_ub))
    d = 2 * dh + 1
    remcols = remove(bgc, cols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (d, d))
    inds = asindices(gi)
    if lni is None:
        lni = choice((1, 2, 3, 4))
    if lni == 1:
        ln = connect((dh, 0), (dh, d - 1))
        mirrf = hmirror
        cands = sfilter(inds, lambda ij: ij[0] > dh)
    elif lni == 2:
        ln = connect((0, dh), (d - 1, dh))
        mirrf = vmirror
        cands = sfilter(inds, lambda ij: ij[1] > dh)
    elif lni == 3:
        ln = connect((0, 0), (d - 1, d - 1))
        mirrf = dmirror
        cands = sfilter(inds, lambda ij: ij[0] > ij[1])
    else:
        ln = connect((d - 1, 0), (0, d - 1))
        mirrf = cmirror
        cands = sfilter(inds, lambda ij: (ij[0] + ij[1]) > d)
    gi = fill(gi, linc, ln)
    mp = (d * (d - 1)) // 2
    numcols = unifint(diff_lb, diff_ub, (1, min(7, mp)))
    colsch = sample(remcols, numcols)
    numpix = unifint(diff_lb, diff_ub, (1, len(cands)))
    pixs = sample(totuple(cands), numpix)
    for pix in pixs:
        gi = fill(gi, choice(colsch), {pix})
    go = mirrf(gi)
    if choice((True, False)):
        gi, go = go, gi
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    sel = [0, 0, hi - 1, wi - 1]   # whole square grid (rectangular full-region: bbox ok)
    ops, sels = [], []

    if np.array_equal(O, np.flipud(I)):
        # hmirror  -> FlipV
        ops = [27]; sels = [sel]
    elif np.array_equal(O, np.fliplr(I)):
        # vmirror  -> FlipH
        ops = [26]; sels = [sel]
    elif np.array_equal(O, np.transpose(I)):
        # dmirror (transpose) = flipud(rot90(I,1)) -> Rotate90(CCW) then FlipV
        ops = [24, 27]; sels = [sel, sel]
    elif np.array_equal(O, I.T[::-1, ::-1]):
        # cmirror (anti-transpose) = flipud(rot90(I,3)) -> Rotate270(CW) then FlipV
        ops = [25, 27]; sels = [sel, sel]
    else:
        # fallback (should not occur for valid instances)
        ops = [27]; sels = [sel]

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9dfd6313"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9dfd6313"
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
                                f"for task 9dfd6313"
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
                    f"Failed to build a complete episode for task 9dfd6313 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9dfd6313-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
