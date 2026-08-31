"""
ARC Task: 44f52bb0 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"issymm": True}, {"issymm": False}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test case included in plan
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, issymm=None) -> dict:
    if issymm is None:
        issymm = random.choice([True, False])

    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    def hmir(g):  # hmirror = up-down = flipud
        return g[::-1, :]

    def vmir(g):  # vmirror = left-right = fliplr
        return g[:, ::-1]

    def dmir(g):  # dmirror = transpose
        return g.T

    def cmir(g):  # cmirror = anti-transpose
        return g[::-1, ::-1].T

    def apply_fn(fn, g):
        if fn == "identity":
            return g
        if fn == "dmirror":
            return dmir(g)
        if fn == "cmirror":
            return cmir(g)
        if fn == "vmirror":
            return vmir(g)
        if fn == "hmirror":
            return hmir(g)
        if fn == "rot90":
            return np.rot90(g, 3)
        if fn == "rot180":
            return g[::-1, ::-1]
        if fn == "rot270":
            return np.rot90(g, 1)
        return g

    h = random.randint(3, max(3, max_h))
    w = random.randint(3, max(3, max_w))

    ncols = min(random.randint(2, 9), len(remcols))
    ccols = random.sample(remcols, ncols)

    inds = [(i, j) for i in range(h) for j in range(w)]
    gi = np.full((h, w), bgc, dtype=int)
    guard = 0
    while np.array_equal(gi, hmir(gi)):
        guard += 1
        numcells = random.randint(1, h * w - 1)
        cells = random.sample(inds, numcells)
        gi = np.full((h, w), bgc, dtype=int)
        for (a, b) in cells:
            col = random.choice(ccols)
            gi[a, b] = col
            gi[a, w - 1 - b] = col
        if guard > 200:
            break

    if not issymm:
        cands = [(i, j) for i in range(h) for j in range(w // 2)]
        numpert = min(random.randint(1, max(1, h * (w // 2))), len(cands))
        locs = random.sample(cands, numpert)
        for (a, b) in locs:
            col = gi[a, b]
            choices = [c for c in (set(ccols) | {bgc}) if c != col]
            if choices:
                gi[a, b] = random.choice(choices)

    mfs = ["identity", "dmirror", "cmirror", "vmirror", "hmirror", "rot90", "rot180", "rot270"]
    nmfs = random.choice([1, 2])
    for fn in random.sample(mfs, nmfs):
        gi = apply_fn(fn, gi)

    # output determined by actual symmetry of final gi (matches verifier)
    symmetric = np.array_equal(gi, vmir(gi)) or np.array_equal(gi, hmir(gi))
    go = np.full((1, 1), 1 if symmetric else 7, dtype=int)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    # measure symmetry from I (NOT read from O)
    symmetric = np.array_equal(I, np.fliplr(I)) or np.array_equal(I, np.flipud(I))
    color = 1 if symmetric else 7

    ops, sels = [], []
    # crop working canvas down to a single 1x1 cell
    ops.append(33); sels.append([0, 0, 0, 0])
    # paint that single cell with the measured symmetry color
    ops.append(color); sels.append([0, 0, 0, 0])
    ops.append(34); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
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
                                f"for task 44f52bb0"
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
                    f"Failed to build a complete episode for task 44f52bb0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44f52bb0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
