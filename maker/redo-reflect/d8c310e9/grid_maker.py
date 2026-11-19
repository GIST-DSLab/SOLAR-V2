"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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

ORIENTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(rem, numc)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [o for o in ORIENTS] + [random.choice(ORIENTS) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ORIENTS, n_ex)
    plan = [{"orient": o} for o in ex]
    plan.append({"orient": random.choice(ex)})
    return {"bgc": bgc, "ccols": list(ccols), "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, orient=None) -> dict:
    if orient is None:
        orient = random.choice(ORIENTS)
    hlo = min(3, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
    pmax = max(2, (w - 1) // 3)
    p = unifint(diff_lb, diff_ub, (2, pmax))
    ccols = list(ccols)
    obj = set()
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, max(1, h - 1)))
        for ii in range(h - 1, h - numcells - 1, -1):
            loc = (ii, j)
            col = random.choice(ccols)
            obj.add((col, loc))
    gi = canvas(bgc, (h, w))
    obj = frozenset(obj)
    minobj = obj | shift(obj, (0, p))
    addonw = random.randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = random.randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    gi = paint(gi, fullobj)
    go = tuple(e for e in gi)
    for j in range(w // (2 * p) + 2):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    fn = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[orient]
    gi = fn(gi)
    go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]          # bbox == exactly the whole canvas

    # --- identify the repeat axis, whether the pattern grows towards the
    #     negative side (then a reflection brings it into "grow forward" form),
    #     and the period of the strip ---
    found = None
    for axis, fl in (("h", False), ("v", False), ("h", True), ("v", True)):
        if fl:
            G = np.fliplr(I) if axis == "h" else np.flipud(I)
        else:
            G = I
        ext = w if axis == "h" else h
        for P in range(1, ext):
            if axis == "h":
                pred = np.tile(G[:, :P], (1, ext // P + 1))[:, :w]
            else:
                pred = np.tile(G[:P, :], (ext // P + 1, 1))[:h, :]
            back = pred
            if fl:
                back = np.fliplr(pred) if axis == "h" else np.flipud(pred)
            if np.array_equal(back, O):
                found = (axis, fl, P)
                break
        if found:
            break

    if found is None:
        diff = [(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]]
        for col in sorted({int(O[r, c]) for r, c in diff}):
            cells = [(r, c) for r, c in diff if int(O[r, c]) == col]
            ops.append(col)
            sels.append(sel_of(cells))
        ops.append(34)
        sels.append(full)
        return ops, sels

    axis, fl, P = found
    W = I.copy()
    flip_op = 26 if axis == "h" else 27

    # reflect the whole canvas so the strip sits at the leading edge
    if fl:
        ops.append(flip_op)
        sels.append(full)                # whole canvas reflected, background included
        W = np.fliplr(W) if axis == "h" else np.flipud(W)

    ext = w if axis == "h" else h
    tile = (W[:, :P] if axis == "h" else W[:P, :]).copy()

    copied = False
    for origin in range(P, ext, P):
        q = min(P, ext - origin)
        sub = tile[:, :q] if axis == "h" else tile[:q, :]
        newW = W.copy()
        reg = newW[:, origin:origin + q] if axis == "h" else newW[origin:origin + q, :]
        m = sub != 0
        reg[m] = sub[m]
        if np.array_equal(newW, W):
            continue                      # this repeat is already present: nothing to do
        if not copied:
            ops.append(29 if fl else 28)
            sels.append([0, 0, h - 1, P - 1] if axis == "h" else [0, 0, P - 1, w - 1])
            copied = True
        ops.append(30)
        sels.append([0, origin, 0, 0] if axis == "h" else [origin, 0, 0, 0])
        W = newW

    # the strip's colour-0 cells are transparent to Paste: draw them explicitly
    zcells = []
    for origin in range(P, ext, P):
        q = min(P, ext - origin)
        if axis == "h":
            for r in range(h):
                for c in range(q):
                    if tile[r, c] == 0 and W[r, origin + c] != 0:
                        zcells.append((r, origin + c))
        else:
            for r in range(q):
                for c in range(w):
                    if tile[r, c] == 0 and W[origin + r, c] != 0:
                        zcells.append((origin + r, c))
    if zcells:
        ops.append(0)
        sels.append(sel_of(zcells))
        for r, c in zcells:
            W[r, c] = 0

    # reflect back into the original orientation
    if fl:
        ops.append(flip_op)
        sels.append(full)

    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
