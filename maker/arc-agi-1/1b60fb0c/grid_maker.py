"""
ARC Task: 1b60fb0c (RE-ARC) — LLM-generated grid_maker
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
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    def rot90(g):
        return [list(r) for r in zip(*g[::-1])]

    def ofcolor(g, c):
        return {(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v == c}

    def shift(p, d):
        return {(i + d[0], j + d[1]) for i, j in p}

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    odh = unifint(diff_lb, diff_ub, (2, max(2, min(h, w) // 2)))
    loci = random.randint(0, h - 2 * odh)
    locj = random.randint(0, w - 2 * odh)
    quad = [[bgc] * odh for _ in range(odh)]
    ncellsd = unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
    ncells = random.choice((ncellsd, odh ** 2 - ncellsd))
    ncells = min(max(1, ncells), odh ** 2 - 1)
    allc = [(i, j) for i in range(odh) for j in range(odh)]
    cells = random.sample(allc, ncells)
    g1 = [row[:] for row in quad]
    for i, j in cells:
        g1[i][j] = objc
    g2 = rot90(g1)
    g3 = rot90(g2)
    g4 = rot90(g3)
    c1 = shift(ofcolor(g1, objc), (0, 0))
    c2 = shift(ofcolor(g2, objc), (0, odh))
    c3 = shift(ofcolor(g3, objc), (odh, odh))
    c4 = shift(ofcolor(g4, objc), (odh, 0))
    s = random.randint(0, odh)
    c1 = shift(c1, (0, s))
    c2 = shift(c2, (s, 0))
    c3 = shift(c3, (0, -s))
    c4 = shift(c4, (-s, 0))
    cs = [c1, c2, c3, c4]
    ri = random.randrange(4)
    rempart = cs[ri]
    inobj = set()
    for k in range(4):
        if k != ri:
            inobj |= cs[k]
    rempart = rempart - inobj
    inobj = shift(inobj, (loci, locj))
    rempart = shift(rempart, (loci, locj))
    gi = [[bgc] * w for _ in range(h)]
    for i, j in inobj:
        if 0 <= i < h and 0 <= j < w:
            gi[i][j] = objc
    go = [row[:] for row in gi]
    for i, j in rempart:
        if 0 <= i < h and 0 <= j < w:
            go[i][j] = 2
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """The pattern has four-fold rotational symmetry with one quarter missing.
    Mark the visible pattern, give it a quarter turn clockwise about the
    pattern's centre, then put the original pattern back in its own colour:
    what is still marked is exactly the quarter that was missing."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- background colour: read from the demonstrations, never from O ---
    bgc = None
    if examples:
        votes = Counter()
        for ex in examples:
            try:
                ei, eo = ex[0], ex[1]
            except Exception:
                ei, eo = ex["input"], ex["output"]
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            m = eo == 2
            if m.any():
                votes.update(ei[m].tolist())
            else:
                votes.update([Counter(ei.flatten().tolist()).most_common(1)[0][0]])
        present = set(I.flatten().tolist())
        for col, _ in votes.most_common():
            if col in present:
                bgc = int(col)
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    S = frozenset((r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc)
    if not S:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    objc = int(I[next(iter(S))])

    rows = [r for r, _ in S]
    cols = [c for _, c in S]
    rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)

    # --- find the quarter turn that carries the pattern onto itself ----
    # a clockwise quarter turn about some centre is  (r, c) -> (a + c, b - r)
    best = None
    for a in range(-cmin, hi - cmax):
        for b in range(rmax, wi + rmin):
            if (a + b) % 2 == 0:
                continue
            T = frozenset((a + c, b - r) for r, c in S)
            if not (T - S):
                continue
            F = S | T
            if frozenset((a + cc, b - rr) for rr, cc in F) != F:
                continue
            score = len(S & T)
            if best is None or score > best[0]:
                best = (score, a, b, T)

    if best is None:                  # nothing is missing: the grid stays as it is
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    _, a, b, T = best

    F = S | T
    cr2, cc2 = a + b, b - a
    d2 = max(max(abs(2 * r - cr2), abs(2 * c - cc2)) for r, c in F)
    L = d2 + 1
    r0, c0 = (cr2 - d2) // 2, (cc2 - d2) // 2

    Scells = sorted(S)
    # 1. mark the whole visible pattern with the answer colour
    ops.append(2); sels.append(sel_of(Scells))
    # 2. quarter turn clockwise about the pattern's centre.
    #    bbox selection is intended: the WHOLE square region turns, background included
    ops.append(25); sels.append([r0, c0, L - 1, L - 1])
    # 3. restore the original pattern in its own colour
    ops.append(objc); sels.append(sel_of(Scells))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
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
                                f"for task 1b60fb0c"
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
                    f"Failed to build a complete episode for task 1b60fb0c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b60fb0c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
