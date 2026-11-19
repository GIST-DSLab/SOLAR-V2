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


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 24)))
    inds = {(i, j) for i in range(h) for j in range(w)}
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for _ in range(nobjs):
        cands = [(i, j) for (i, j) in inds
                 if all((i + a, j + b) in inds for a, b in obj)]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        plcd = [(loc[0] + a, loc[1] + b) for a, b in obj]
        nred = unifint(diff_lb, diff_ub, (1, 3))
        reds = random.sample(plcd, nred)
        for (i, j) in plcd:
            gi[i][j] = objc
        for (i, j) in reds:
            gi[i][j] = 2
        for idx in reds:
            dr = 2 * (idx[0] - loc[0]) - 1
            dc = 2 * (idx[1] - loc[1]) - 1
            for (i, j) in plcd:
                k = 0
                while 0 <= i + k * dr < h and 0 <= j + k * dc < w:
                    go[i + k * dr][j + k * dc] = objc
                    k += 1
        rem = set(plcd)
        for (i, j) in plcd:
            rem |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        inds -= rem
    return {"input": gi, "output": go}


def _components(I, bgc):
    h, w = I.shape
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen \
                            and I[na, nb] != bgc:
                        seen.add((na, nb))
                        stack.append((na, nb))
            comps.append(sorted(comp))
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    objc = None
    for v in sorted(set(I.flatten().tolist())):
        if v != bgc and v != 2:
            objc = v
    if objc is None and examples:
        for ei, _eo in examples:
            ei = np.asarray(ei, dtype=int)
            eb = Counter(ei.flatten().tolist()).most_common(1)[0][0]
            for v in sorted(set(ei.flatten().tolist())):
                if v != eb and v != 2:
                    objc = v
            if objc is not None:
                break
    if objc is None:
        return [34], [sel_of([(r, c) for r in range(hi) for c in range(wi)])]

    # Reading (shared by the maker and the golf entry): each red corner is a marker
    # that travels diagonally away from its 2x2 block; at every step the block's
    # 2x2 footprint is laid down at the marker's new place, so the cell the marker
    # leaves (starting with the red cell itself) turns into body colour.
    # No separate "recolour the reds" step: step 1's footprint already covers it.
    stamps = []  # clipped 2x2 footprints, ray by ray, outward
    for comp in _components(I, bgc):
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        reds = [(r, c) for r, c in comp if I[r, c] == 2]
        for (rr, rc) in reds:
            dr = 2 * (rr - r0) - 1
            dc = 2 * (rc - c0) - 1
            k = 1
            while True:
                cells = [(r0 + k * dr + a, c0 + k * dc + b)
                         for a in (0, 1) for b in (0, 1)]
                cells = [(r, c) for r, c in cells if 0 <= r < hi and 0 <= c < wi]
                if not cells:
                    break
                stamps.append(cells)
                k += 1

    def replay(st):
        G = I.copy()
        for cells in st:
            for r, c in cells:
                G[r, c] = objc
        return G

    full = replay(stamps)
    # drop any footprint the rest of the rule (computed from I alone) already
    # lays down -- such an op would be deletable / a no-op
    i = 0
    while i < len(stamps):
        trial = stamps[:i] + stamps[i + 1:]
        if np.array_equal(replay(trial), full):
            stamps = trial
        else:
            i += 1

    ops, sels = [], []
    for cells in stamps:
        ops.append(int(objc))
        sels.append(sel_of(sorted(cells)))   # the block's footprint at this step
    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
