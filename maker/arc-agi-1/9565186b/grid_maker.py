"""
ARC Task: 9565186b (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # generator samples: mostcol (dominant color) + othcols (the rest).
    # 5 is reserved as the blanking color and never appears in the input.
    cols = [c for c in range(10) if c != 5]
    mostcol = random.choice(cols)
    rem = [c for c in cols if c != mostcol]
    random.shuffle(rem)
    return {"mostcol": mostcol, "othcols_pool": rem}


def generate(diff_lb, diff_ub, max_h, max_w, mostcol, othcols_pool) -> dict:
    h = _unifint(diff_lb, diff_ub, (2, max_h))
    w = _unifint(diff_lb, diff_ub, (2, max_w))

    numcols = _unifint(diff_lb, diff_ub, (2, min(h * w - 1, 8)))
    numcols = min(numcols, len(othcols_pool) + 1)

    nummostcol_lb = (h * w) // numcols + 1
    nummostcol_ub = h * w - numcols + 1
    ubmlb = max(0, nummostcol_ub - nummostcol_lb)
    nmcdev = _unifint(diff_lb, diff_ub, (0, ubmlb))
    nummostcol = nummostcol_ub - nmcdev
    nummostcol = min(max(nummostcol, nummostcol_lb), nummostcol_ub)

    inds = [(i, j) for i in range(h) for j in range(w)]
    mostcollocs = random.sample(inds, nummostcol)

    gi = [[5] * w for _ in range(h)]
    go = [[5] * w for _ in range(h)]
    for (i, j) in mostcollocs:
        gi[i][j] = mostcol
        go[i][j] = mostcol

    othcols = list(othcols_pool[:numcols - 1])

    mostset = set(mostcollocs)
    reminds = [ij for ij in inds if ij not in mostset]

    bufferlocs = random.sample(reminds, numcols - 1)
    for c, l in zip(othcols, bufferlocs):
        gi[l[0]][l[1]] = c

    bufset = set(bufferlocs)
    reminds = [ij for ij in reminds if ij not in bufset]

    colcounts = {c: 1 for c in othcols}
    live = list(othcols)
    for (i, j) in reminds:
        if len(live) == 0:
            gi[i][j] = mostcol
            go[i][j] = mostcol
        else:
            chc = random.choice(live)
            gi[i][j] = chc
            colcounts[chc] += 1
            if colcounts[chc] == nummostcol - 1:
                live = [c for c in live if c != chc]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # Rule measured from I only: partition I by color, the largest color group
    # survives; every other color group is blanked to 5.
    counts = Counter(I.flatten().tolist())
    mostcol = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # one Color5 op per losing color group, largest group first
    others = sorted([c for c in counts if c != mostcol],
                    key=lambda c: (-counts[c], c))
    for c in others:
        cells = [(r, k) for r in range(hi) for k in range(wi) if I[r, k] == c]
        if not cells:
            continue
        ops.append(5)
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9565186b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9565186b"
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
                                f"for task 9565186b"
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
                    f"Failed to build a complete episode for task 9565186b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9565186b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
