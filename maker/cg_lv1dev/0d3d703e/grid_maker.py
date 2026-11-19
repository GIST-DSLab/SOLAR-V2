"""
ARC Task: 0d3d703e (RE-ARC) — LLM-generated grid_maker
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

INCOLS = (1, 2, 3, 4, 5, 6, 8, 9)
OUTCOLS = (5, 6, 4, 3, 1, 2, 9, 8)
MAPPING = dict(zip(INCOLS, OUTCOLS))


def sample_colors(num_examples=None) -> dict:
    # All colours in this task are hardcoded (fixed swap table), so no colour role is sampled.
    # What varies per instance is which subset of the swap pairs is used. Plan those subsets
    # so that the test's subset is one the examples already showed; that keeps every colour the
    # test uses demonstrated in the examples.
    k = len(INCOLS)
    n_ex = num_examples if num_examples else 3
    examples = []
    for _ in range(n_ex):
        numc = random.randint(1, k)
        idxes = sorted(random.sample(range(k), numc))
        examples.append({"idxes": idxes})
    test = dict(random.choice(examples))
    test["idxes"] = list(test["idxes"])
    plan = examples + [test]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, idxes=None, **kwargs) -> dict:
    k = len(INCOLS)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    if idxes is None:
        numc = unifint(diff_lb, diff_ub, (1, k))
        idxes = sample(interval(0, k, 1), numc)
    idxes = list(idxes)
    gi = [[0] * w for _ in range(h)]
    go = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            idx = choice(idxes)
            gi[i][j] = INCOLS[idx]
            go[i][j] = OUTCOLS[idx]
    gi = tuple(tuple(r) for r in gi)
    go = tuple(tuple(r) for r in go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Rule: a fixed per-colour substitution (swap pairs 1<->5, 2<->6, 3<->4, 8<->9).
    # The code-golf entry reads the task the same way: one lookup table that maps each
    # colour to its partner. It indexes only the first row because the original ARC grids
    # have constant columns, but the unit it works with is still "a colour", and that is
    # the same unit the maker uses. The two readings agree, so the trajectory stays the same:
    # one Color op for each input colour class, selecting every cell of that colour in I.
    #
    # Read the table from the demonstrations wherever they show it, and use the fixed task
    # table only for a colour the demonstrations never show.
    mapping = dict(MAPPING)
    if examples:
        for ex in examples:
            if isinstance(ex, dict):
                ei, eo = ex["input"], ex["output"]
            else:
                ei, eo = ex
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            for a, b in zip(ei.flatten().tolist(), eo.flatten().tolist()):
                mapping[a] = b

    ops, sels = [], []
    # Each mask is "all cells of colour src in the INPUT". It can be named before the answer
    # is known. Because every mask comes from I and not from the working grid, swapping the
    # two members of a pair cannot clobber the other one, whatever the order.
    for src in sorted(set(I.flatten().tolist())):
        dst = mapping.get(src, src)
        if dst == src:
            continue
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == src]
        if not cells:
            continue
        ops.append(int(dst))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid, for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 0d3d703e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0d3d703e"
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
                                f"for task 0d3d703e"
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
                    f"Failed to build a complete episode for task 0d3d703e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0d3d703e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
