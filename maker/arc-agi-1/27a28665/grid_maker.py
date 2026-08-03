"""
ARC Task: 27a28665 (RE-ARC) — LLM-generated grid_maker
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

# structural variants: (output_color, 3x3 footprint of the mark)
# output color is a FUNCTION of the footprint's largest 4-connected component size
# (size 1->2, 4->3, 5->6, else->1). We keep the generator's 4 footprints to build data,
# but derive_operations MEASURES that connectivity from I, never a table lookup.
MAPPING = [
    (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
    (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
    (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
    (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)}),
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c != bgc])
    n_var = len(MAPPING)
    n_ex = num_examples if num_examples else 5
    idxs = list(range(n_var))
    if n_ex >= n_var:
        examples = list(idxs)
        examples += [random.choice(idxs) for _ in range(n_ex - n_var)]
        random.shuffle(examples)
    else:
        examples = random.sample(idxs, n_ex)
    plan = [{"variant": v} for v in examples]
    plan.append({"variant": random.choice(examples)})  # test drawn from shown variants
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, variant=None) -> dict:
    if variant is None:
        variant = random.randrange(len(MAPPING))
    col, obj = MAPPING[variant]
    h = random.randint(3, max_h)
    w = random.randint(3, max_w)
    fac = random.randint(1, min(h, w) // 3)
    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - 3 * fac)
    locj = random.randint(0, w - 3 * fac)
    for (i, j) in obj:
        for dr in range(fac):
            for dc in range(fac):
                gi[loci + i * fac + dr][locj + j * fac + dc] = objc
    go = [[col]]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = dominant color the canvas was painted with
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # foreground mark cells
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    rs = [r for r, c in fg]
    cs = [c for r, c in fg]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    bw = c1 - c0 + 1
    fac = max(1, bw // 3)  # mark = 3x3 pattern upscaled by fac

    # reconstruct the 3x3 downscaled footprint
    foot = set()
    for i in range(3):
        for j in range(3):
            filled = False
            for dr in range(fac):
                for dc in range(fac):
                    if I[r0 + i * fac + dr, c0 + j * fac + dc] != bgc:
                        filled = True
            if filled:
                foot.add((i, j))

    # largest 4-connected component size within the footprint
    seen = set()
    best = 0
    for cell in foot:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        sz = 0
        while stack:
            r, c = stack.pop()
            sz += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in foot and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        best = max(best, sz)

    if best == 1:
        target = 2
    elif best == 4:
        target = 3
    elif best == 5:
        target = 6
    else:
        target = 1

    ops, sels = [], []
    ops.append(33); sels.append([0, 0, 0, 0])       # collapse canvas to a single 1x1 cell
    ops.append(target); sels.append([0, 0, 0, 0])   # paint measured answer color
    ops.append(34); sels.append([0, 0, 0, 0])       # submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 27a28665"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 27a28665"
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
                                f"for task 27a28665"
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
                    f"Failed to build a complete episode for task 27a28665 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"27a28665-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
