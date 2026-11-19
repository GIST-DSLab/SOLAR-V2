"""
ARC Task: fafffa47 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round(d * (b - a) * 0.0))
    v = int(round(a + (b - a) * random.uniform(max(0.0, diff_lb), min(1.0, max(diff_lb, diff_ub)))))
    return max(a, min(b, v if v >= lo else lo))


def _split_vertical(I):
    hi, wi = I.shape

    def nc(g):
        return len(set(g.flatten().tolist()))
    return (nc(I[:hi // 2]) == 2 and nc(I[hi // 2:]) == 2) and not (
        nc(I[:, :wi // 2]) == 2 and nc(I[:, wi // 2:]) == 2)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None,
             transpose=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(cols)
    if acol is None:
        acol = random.choice([c for c in cols if c != bgc])
    if bcol is None:
        bcol = random.choice([c for c in cols if c not in (bgc, acol)])
    if transpose is None:
        transpose = random.choice([False, True])
    lim = min(max_h, max_w)
    hmax = max(2, min(30, lim))
    wmax = max(2, min(14, lim // 2))
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (2, hmax))
        w = _unifint(diff_lb, diff_ub, (2, wmax))
        n = h * w
        inds = [(i, j) for i in range(h) for j in range(w)]
        numadev = _unifint(diff_lb, diff_ub, (0, n // 2))
        numbdev = _unifint(diff_lb, diff_ub, (0, n // 2))
        numa = random.choice((numadev, n - numadev))
        numb = random.choice((numadev, n - numbdev))
        numa = min(max(1, numa), n - 1)
        numb = min(max(1, numb), n - 1)
        aset = random.sample(inds, numa)
        bset = random.sample(inds, numb)
        A = np.full((h, w), bgc, dtype=int)
        B = np.full((h, w), bgc, dtype=int)
        for p in aset:
            A[p] = acol
        for p in bset:
            B[p] = bcol
        gi = np.hstack([A, B])
        go = np.full((h, w), bgc, dtype=int)
        for p in set(inds) - (set(aset) | set(bset)):
            go[p] = 2
        if transpose:
            gi = gi.T.copy()
            go = go.T.copy()
        # keep only instances where the split rule reads the orientation unambiguously
        if _split_vertical(gi) != bool(transpose):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("could not generate unambiguous instance")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def nc(g):
        return len(set(g.flatten().tolist()))

    # which way the grid splits into two halves, read from I alone
    vertical = (nc(I[:hi // 2]) == 2 and nc(I[hi // 2:]) == 2) and not (
        nc(I[:, :wi // 2]) == 2 and nc(I[:, wi // 2:]) == 2)
    if vertical:
        H1, W1 = hi // 2, wi
        F = I[:H1, :W1]
        S = I[H1:2 * H1, :W1]
    else:
        H1, W1 = hi, wi // 2
        F = I[:H1, :W1]
        S = I[:H1, W1:2 * W1]

    # the background is the one colour both halves' palettes share
    bgc = sorted(set(F.flatten().tolist()) & set(S.flatten().tolist()))[0]

    # 1. the answer canvas has the shape of one half: crop to the first half (full rectangle)
    ops.append(33)
    sels.append([0, 0, H1 - 1, W1 - 1])

    # 2. lay a fresh background over the whole canvas (full rectangle; the half
    #    always holds at least one mark, so this visibly changes the grid)
    ops.append(bgc)
    sels.append([0, 0, H1 - 1, W1 - 1])

    # 3. mark cell-wise overlay: every cell blank in BOTH halves becomes 2
    both_blank = [(r, c) for r in range(H1) for c in range(W1)
                  if F[r, c] == bgc and S[r, c] == bgc]
    if both_blank:
        ops.append(2)
        sels.append(sel_of(both_blank))

    ops.append(34)
    sels.append([0, 0, H1 - 1, W1 - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task fafffa47"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fafffa47"
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
                                f"for task fafffa47"
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
                    f"Failed to build a complete episode for task fafffa47 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fafffa47-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
