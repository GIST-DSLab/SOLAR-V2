"""
ARC Task: 48d8fb45 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a fixed role. The marker colour and the object
    # colours do not affect the rule (the object is the one built from two colours).
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * max(0.0, min(1.0, diff_lb))
    b = lo + (hi - lo) * max(0.0, min(1.0, diff_ub))
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    return random.randint(max(lo, a), min(hi, max(b, a)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, 10, max_h)
    w = _unifint(diff_lb, diff_ub, 10, max_w)
    nobjs = _unifint(diff_lb, diff_ub, 2, max(2, (h * w) // 15))
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def nbrs(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def dneigh(p):
        i, j = p
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    tr, maxtr, succ, done = 0, 4 * nobjs, 0, False
    go = None
    placed_target = False
    while tr < maxtr and succ < nobjs:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        bx = {(i, j) for i in range(oh) for j in range(ow)}
        nc = random.randint(3, oh * ow)
        sp = random.choice(sorted(bx))
        bx.discard(sp)
        cells = {sp}
        for _ in range(nc - 1):
            front = set()
            for p in cells:
                front |= nbrs(p)
            opts = sorted((bx - cells) & front)
            cells.add(random.choice(opts))
        is_target = False
        if not done:
            done = True
            is_target = True
            idx = random.choice(sorted(cells))
            coll = random.choice(remcols)
            col2 = random.choice([c for c in remcols if c != coll])
            obj = {idx: coll}
            rest = [p for p in cells if p != idx]
            for p in rest:
                obj[p] = col2
            r0 = min(p[0] for p in rest); c0 = min(p[1] for p in rest)
            r1 = max(p[0] for p in rest); c1 = max(p[1] for p in rest)
            go = [[bgc] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
            for (a, b) in rest:
                go[a - r0][b - c0] = col2
        else:
            col = random.choice(remcols)
            obj = {p: col for p in cells}
        tr += 1
        locopts = sorted(ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow)
        if not locopts:
            continue
        loc = random.choice(locopts)
        plcd = {(a + loc[0], b + loc[1]): v for (a, b), v in obj.items()}
        if set(plcd).issubset(inds):
            for (a, b), v in plcd.items():
                gi[a][b] = v
            succ += 1
            if is_target:
                placed_target = True
            blk = set(plcd)
            for p in list(plcd):
                blk |= nbrs(p)
            inds -= blk
    if not placed_target or succ < 2:
        raise ValueError("generation failed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Background: the canvas colour shared by all inputs of the episode.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _eo in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # Reading: one lone cell is a MARKER -- a cell whose colour differs from a
    # non-background cell touching it (objects never touch each other, so this only
    # happens inside the marked object). The marker points at the object to extract.
    def nb8(a, b):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if (da, db) != (0, 0):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi:
                        yield na, nb

    # The marked object = the 8-connected non-bg component that has two colours.
    seen = np.zeros((hi, wi), dtype=bool)
    target = None
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for na, nb in nb8(a, b):
                        if not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
                if len(set(int(I[a, b]) for a, b in cells)) > 1:
                    target = cells
    if target is None:
        raise ValueError("no marked object")

    ccount = Counter(int(I[a, b]) for a, b in target)
    body_col = ccount.most_common(1)[0][0]
    marker = [(a, b) for a, b in target if int(I[a, b]) != body_col]
    body = [(a, b) for a, b in target if int(I[a, b]) == body_col]

    r0 = min(a for a, b in body); r1 = max(a for a, b in body)
    c0 = min(b for a, b in body); c1 = max(b for a, b in body)

    ops, sels = [], []
    # Remove the marker (it is not part of the object) when it lies in the window
    # we keep; outside the window the crop discards it anyway.
    inside = [(a, b) for a, b in marker if r0 <= a <= r1 and c0 <= b <= c1]
    if inside:
        ops.append(int(bgc)); sels.append(sel_of(inside))

    # Crop to the marked object's body window: the full rectangle is intended.
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 48d8fb45"
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
                                f"for task 48d8fb45"
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
                    f"Failed to build a complete episode for task 48d8fb45 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"48d8fb45-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
