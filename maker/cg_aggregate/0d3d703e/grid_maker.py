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
    # No colours are sampled at random here: the colour map is fixed by the task.
    # What changes from one instance to the next is WHICH input colours appear.
    # Plan the colour subsets so the examples together show every colour pair.
    # Then the test copies one example's subset, so every colour it uses has been demonstrated.
    k = len(INCOLS)
    n_ex = num_examples if num_examples else 3
    n_ex = max(1, n_ex)
    order = list(range(k))
    random.shuffle(order)
    buckets = [[] for _ in range(n_ex)]
    for i, idx in enumerate(order):
        buckets[i % n_ex].append(idx)
    examples = []
    for b in buckets:
        extra_pool = [i for i in range(k) if i not in b]
        n_extra = random.randint(0, len(extra_pool))
        sub = sorted(set(b) | set(random.sample(extra_pool, n_extra)))
        examples.append({"idxes": sub})
    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, idxes=None, **kwargs) -> dict:
    k = len(INCOLS)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    if idxes is None:
        numc = unifint(diff_lb, diff_ub, (1, k))
        idxes = random.sample(range(k), numc)
    idxes = list(idxes)
    gi = [[0] * w for _ in range(h)]
    go = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            idx = random.choice(idxes)
            gi[r][c] = INCOLS[idx]
            go[r][c] = OUTCOLS[idx]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Rule: each cell's colour depends only on the colour it starts with.
    # (The code-golf entry reads the task the same way: x -> f(x) for every cell.)
    # The colour map comes from the demonstrations: in every example a colour
    # always turns into the same colour at the same cell. The fixed task map is
    # used only as a fallback for a colour the examples never show.
    learned = {}
    if examples:
        for ex in examples:
            try:
                ei, eo = ex
            except Exception:
                ei, eo = ex["input"], ex["output"]
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            for a, b in zip(ei.flatten().tolist(), eo.flatten().tolist()):
                learned.setdefault(a, b)
    cmap = dict(MAPPING)
    cmap.update(learned)

    # Each input colour is treated as one unit: all of its cells, with the mask
    # taken from I, get one Color op. The masks come from I, so a swapped pair
    # (a<->b) cannot overwrite each other. Ops are ordered so the two halves of
    # each swap pair come one after the other.
    present = set(I.flatten().tolist())
    done = set()
    order = []
    for src in sorted(present):
        if src in done:
            continue
        done.add(src)
        order.append(src)
        partner = cmap.get(src, src)
        if partner in present and partner not in done and cmap.get(partner, partner) == src:
            done.add(partner)
            order.append(partner)

    ops, sels = [], []
    for src in order:
        dst = cmap.get(src, src)
        if dst == src:
            continue
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == src]
        if not cells:
            continue
        ops.append(int(dst))
        sels.append(sel_of(cells))

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
