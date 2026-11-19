"""
ARC Task: 1f0c79e5 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 24)))
    inds = {(i, j) for i in range(h) for j in range(w)}
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for _ in range(nobjs):
        cands = [(i, j) for (i, j) in inds
                 if all((i + a, j + b) in inds for a, b in obj)]
        if not cands:
            break
        li, lj = random.choice(cands)
        plcd = [(li + a, lj + b) for a, b in obj]
        nred = unifint(diff_lb, diff_ub, (1, 3))
        reds = random.sample(plcd, nred)
        for r, c in plcd:
            gi[r, c] = objc
        for r, c in reds:
            gi[r, c] = 2
        for (r, c) in reds:
            dr, dc = 2 * (r - li) - 1, 2 * (c - lj) - 1
            for (a, b) in plcd:
                k = 0
                while 0 <= a + k * dr < h and 0 <= b + k * dc < w:
                    go[a + k * dr, b + k * dc] = objc
                    k += 1
        rem = set(plcd)
        for r, c in plcd:
            rem |= {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
        inds -= rem
    return {"input": gi.tolist(), "output": go.tolist()}


def _blocks(I, bgc):
    hi, wi = I.shape
    used, blocks = set(), []
    for r in range(hi - 1):
        for c in range(wi - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if any(x in used for x in cells):
                continue
            if all(I[x] != bgc for x in cells) and any(I[x] == 2 for x in cells):
                used |= set(cells)
                blocks.append((r, c, cells))
    return blocks


def _beam(cells, dr, dc, hi, wi):
    out = set()
    for a, b in cells:
        k = 0
        while 0 <= a + k * dr < hi and 0 <= b + k * dc < wi:
            out.add((a + k * dr, b + k * dc))
            k += 1
    return out


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = canvas colour (blocks occupy a small fraction of the grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # object colour: the non-bg, non-red colour (from I, else from demonstrations)
    objc = None
    for v in sorted(set(I.flatten().tolist())):
        if v != bgc and v != 2:
            objc = v
    if objc is None and examples:
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            eb = Counter(ei.flatten().tolist()).most_common(1)[0][0]
            for v in sorted(set(ei.flatten().tolist())):
                if v != eb and v != 2:
                    objc = v
            if objc is not None:
                break
    if objc is None:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # One beam per red cell: the red corner marks a diagonal direction (away from
    # the block centre); the 2x2 block shoots that way, sweeping a 3-wide diagonal
    # band that starts on the block itself (so the red gets recoloured too).
    beams = []
    for (r0, c0, cells) in _blocks(I, bgc):
        for (r, c) in cells:
            if I[r, c] == 2:
                beams.append(_beam(cells, 2 * (r - r0) - 1, 2 * (c - c0) - 1, hi, wi))

    # drop a beam only if the rule's own picture (from I alone) is unchanged without it
    def replay(bs):
        G = I.copy()
        for b in bs:
            for x in b:
                G[x] = objc
        return G
    full = replay(beams)
    i = 0
    while i < len(beams):
        trial = beams[:i] + beams[i + 1:]
        if np.array_equal(replay(trial), full):
            beams = trial
        else:
            i += 1

    ops, sels = [], []
    for b in beams:
        ops.append(int(objc))
        sels.append(sel_of(sorted(b)))
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
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
                                f"for task 1f0c79e5"
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
                    f"Failed to build a complete episode for task 1f0c79e5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f0c79e5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
