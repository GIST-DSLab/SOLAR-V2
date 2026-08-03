"""
ARC Task: 1caeab9d (RE-ARC) — LLM-generated grid_maker
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
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def sample_colors(num_examples=None) -> dict:
    # rule depends only on object positions (anchor = the color-1 object),
    # so only the background color must be fixed across the episode
    bgc = random.choice([c for c in range(10) if c != 1])
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (6, max_w))
    oh = _unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
    ow = _unifint(diff_lb, diff_ub, (1, max(1, w // 3)))

    bb = [(i, j) for i in range(oh) for j in range(ow)]
    sp = random.choice(bb)
    obj = {sp}
    rem = set(bb) - obj
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice([ncellsd, oh * ow - ncellsd])
    ncells = min(max(0, ncells), oh * ow - 1)
    for _ in range(ncells):
        cands = [c for c in rem
                 if any(abs(c[0] - o[0]) <= 1 and abs(c[1] - o[1]) <= 1 for o in obj)]
        if not cands:
            break
        pick = random.choice(cands)
        obj.add(pick)
        rem.discard(pick)

    mr = min(r for r, _ in obj)
    mc = min(c for _, c in obj)
    obj = {(r - mr, c - mc) for r, c in obj}
    oh = max(r for r, _ in obj) + 1
    ow = max(c for _, c in obj) + 1

    loci = random.randint(0, h - oh)
    numo = _unifint(diff_lb, diff_ub, (2, min(8, max(2, w // ow)))) - 1
    locj = random.randint(0, w - ow)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for r, c in obj:
        gi[loci + r, locj + c] = 1
        go[loci + r, locj + c] = 1

    # object colors exclude 0 (ARCLE treats 0 as "nothing" for object ops)
    remcols = [c for c in range(10) if c not in (0, 1, bgc)]
    random.shuffle(remcols)

    itv = set(range(w)) - set(range(locj, locj + ow))
    for _ in range(numo):
        cands = [j for j in sorted(itv) if all(x in itv for x in range(j, j + ow))]
        if not cands or not remcols:
            break
        locj = random.choice(cands)
        col = remcols.pop()
        loci_i = random.randint(0, h - oh)
        for r, c in obj:
            gi[loci_i + r, locj + c] = col
            go[loci + r, locj + c] = col
        itv -= set(range(locj, locj + ow))

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # anchor = the color-1 object; every other object slides vertically
    # until its bottom row matches the anchor's bottom row
    anchor_bottom = int(np.argwhere(I == 1)[:, 0].max())

    for col in [int(c) for c in np.unique(I) if c != bgc and c != 1]:
        cells = [(int(r), int(c)) for r, c in np.argwhere(I == col)]
        dr = anchor_bottom - max(r for r, _ in cells)
        if dr == 0:
            continue
        step = 1 if dr > 0 else -1
        move_op = 21 if dr > 0 else 20

        cur = cells
        for _ in range(abs(dr)):
            ops.append(move_op)
            sels.append(sel_of(cur))
            cur = [(r + step, c) for r, c in cur]

        # ARCLE zeroes every cell the object passed through but no longer occupies
        if bgc != 0:
            swept = {(r + step * k, c) for k in range(abs(dr) + 1) for r, c in cells}
            vacated = sorted(swept - set(cur))
            if vacated:
                ops.append(int(bgc))
                sels.append(sel_of(vacated))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
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
                                f"for task 1caeab9d"
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
                    f"Failed to build a complete episode for task 1caeab9d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1caeab9d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
