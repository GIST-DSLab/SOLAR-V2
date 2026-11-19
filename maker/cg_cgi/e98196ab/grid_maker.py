"""
ARC Task: e98196ab (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem); rem.remove(linc)
    topc = random.choice(rem); rem.remove(topc)
    botc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "topc": topc, "botc": botc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, topc, botc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if transpose:
        hcap = (max_w - 1) // 2; wcap = max_h
    else:
        hcap = (max_h - 1) // 2; wcap = max_w
    h = unifint(diff_lb, diff_ub, (3, min(14, hcap)))
    w = unifint(diff_lb, diff_ub, (3, min(30, wcap)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    nocc = unifint(diff_lb, diff_ub, (2, (h * w) // 2))
    subs = random.sample(inds, nocc)
    numa = random.randint(1, nocc - 1)
    A = random.sample(subs, numa)
    Aset = set(A)
    B = [x for x in subs if x not in Aset]
    top = np.full((h, w), bgc, dtype=int)
    bot = np.full((h, w), bgc, dtype=int)
    for r, c in A:
        top[r, c] = topc
    for r, c in B:
        bot[r, c] = botc
    go = top.copy()
    for r, c in B:
        go[r, c] = botc
    gi = np.vstack([top, np.full((1, w), linc, dtype=int), bot])
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # Reading: the input is two halves around a uniform separator line; the
    # second half is laid over (superimposed onto) the first half as one layer.
    if h % 2 == 1 and len(set(I[h // 2].tolist())) == 1:
        row_split = True
        half = h // 2
        A = I[:half, :]
        B = I[half + 1:, :]
    else:
        row_split = False
        half = w // 2
        A = I[:, :half]
        B = I[:, half + 1:]

    # background = the only colour shared by both halves
    common = set(A.flatten().tolist()) & set(B.flatten().tolist())
    if len(common) == 1:
        bgc = common.pop()
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    shift = half + 1
    # the second half's foreground layer, in I coordinates
    if row_split:
        layer = [(r + shift, c) for r in range(B.shape[0]) for c in range(B.shape[1])
                 if B[r, c] != bgc]
    else:
        layer = [(r, c + shift) for r in range(B.shape[0]) for c in range(B.shape[1])
                 if B[r, c] != bgc]
    ho, wo = A.shape
    ops, sels = [], []

    layer_cols = {int(I[r, c]) for r, c in layer}
    if layer and 0 not in layer_cols:
        # slide the whole second-half layer across the separator onto the first half
        mop = 20 if row_split else 23
        ops.append(mop); sels.append(sel_of(layer))      # grab the layer once
        for _ in range(shift - 1):
            ops.append(mop); sels.append(sel_of([]))     # keep sliding the same object
        # vacated footprint lies in the second half, which the crop discards
    elif layer:
        # colour 0 cannot be carried by Move (zeros are transparent): paint landing cells
        if row_split:
            dst = [(r - shift, c) for r, c in layer]
        else:
            dst = [(r, c - shift) for r, c in layer]
        for col in sorted(layer_cols):
            cells = [d for d, s in zip(dst, layer) if I[s] == col and I[d] != col]
            if cells:
                ops.append(col); sels.append(sel_of(cells))

    # keep only the first half: crop to that full rectangle (background included)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e98196ab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e98196ab"
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
                                f"for task e98196ab"
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
                    f"Failed to build a complete episode for task e98196ab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e98196ab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
