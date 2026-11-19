"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background has a role here; each object's colour is the
    # object's own identity, and the rule does not depend on which colour it is.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _connect(a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return {(r1, c) for c in range(min(c1, c2), max(c1, c2) + 1)}
    if c1 == c2:
        return {(r, c1) for r in range(min(r1, r2), max(r1, r2) + 1)}
    if abs(r1 - r2) == abs(c1 - c2):
        n = abs(r1 - r2)
        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1
        return {(r1 + k * dr, c1 + k * dc) for k in range(n + 1)}
    return set()


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    num = unifint(diff_lb, diff_ub, (1, max(1, min(9, (h * w) // 10))))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 6 * num

    def mirror(cells, mf):
        if mf == "id":
            return set(cells)
        if mf == "d":
            return {(c, r) for r, c in cells}
        if not cells:
            return set()
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if mf == "h":
            return {(r0 + r1 - r, c) for r, c in cells}
        if mf == "v":
            return {(r, c0 + c1 - c) for r, c in cells}
        # cmirror: anti-diagonal mirror around the bbox
        # cmirror(x) = vmirror(dmirror(vmirror(x)))
        x = {(r, c0 + c1 - c) for r, c in cells}
        x = {(c, r) for r, c in x}
        xs = [c for _, c in x]
        a, b = min(xs), max(xs)
        return {(r, a + b - c) for r, c in x}

    while tr < maxtr and succ < num:
        d = unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if len(inds) == 0 or len(remcols) == 0:
            tr += 1
            continue
        loci, locj = random.choice(sorted(inds))
        io1 = _connect((loci, locj), (loci + d - 1, locj + d - 1))
        io2 = _connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = set()
        iol = list(io)
        for a in iol:
            for b in iol:
                if a[0] == b[0]:
                    oo |= _connect(a, b)
        mf = random.choice(["id", "d", "c", "h", "v"])
        # mirror with a shared bbox reference: mirror io and oo jointly
        allc = io | oo
        if mf in ("h", "v", "c"):
            rs = [r for r, _ in allc]; cs = [c for _, c in allc]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            def m(cells):
                if mf == "h":
                    return {(r0 + r1 - r, c) for r, c in cells}
                if mf == "v":
                    return {(r, c0 + c1 - c) for r, c in cells}
                x = {(r, c0 + c1 - c) for r, c in cells}
                x = {(c, r) for r, c in x}
                # after dmirror, the bbox columns are r0..r1
                return {(r, r0 + r1 - c) for r, c in x}
            io2_, oo2_ = m(io), m(oo)
        elif mf == "d":
            io2_, oo2_ = {(c, r) for r, c in io}, {(c, r) for r, c in oo}
        else:
            io2_, oo2_ = io, oo
        col = random.choice(remcols)
        if oo2_.issubset(inds):
            for r, c in io2_:
                gi[r, c] = col
            for r, c in oo2_:
                go[r, c] = col
            succ += 1
            inds = inds - oo2_
            remcols = [c for c in remcols if c != col]
        tr += 1
    if succ == 0:
        raise ValueError("no object placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background: the canvas colour the generator paints first (dominant).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Rule (read from I alone): each non-background colour is one object;
    # fill it in by joining every pair of its cells with a straight line.
    # One Color op per object paints that object's whole filled shape at once.
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})
    for col in colors:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        filled = set(cells)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                filled |= _connect(cells[i], cells[j])
        if filled == set(cells):
            continue  # nothing to fill for this object; the op would be a no-op
        ops.append(col)
        sels.append(sel_of(sorted(filled)))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
