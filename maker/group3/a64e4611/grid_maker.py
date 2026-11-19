"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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

PLAN_VARIANTS = [{"spi_zero": True}, {"spi_zero": False}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(PLAN_VARIANTS):
        ex = [dict(v) for v in PLAN_VARIANTS]
        ex += [dict(random.choice(PLAN_VARIANTS)) for _ in range(n_ex - len(PLAN_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(PLAN_VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


def _runs(flags):
    out, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1)); s = None
    return out


def _detect(I, bgc=None):
    """Return (bgc, rects): interiors of the bgc band structure, measured from I alone."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    L, top = [], []
    for j in range(w):
        c = I[h - 1, j]
        k = h - 1
        while k >= 0 and I[k, j] == c:
            k -= 1
        L.append(h - 1 - k); top.append(k + 1)
    lmin = h - h // 2
    best = None
    for col in range(10):
        if bgc is not None and col != bgc:
            continue
        flags = [L[j] >= lmin and I[h - 1, j] == col for j in range(w)]
        for a, b in _runs(flags):
            if b - a + 1 >= 3 and (best is None or b - a > best[2] - best[1]):
                best = (col, a, b)
    if best is None:
        return None, []
    bgc, a, b = best
    spi = max(top[a:b + 1])
    rects = [((spi + 1) if spi > 0 else 0, a + 1, h - 1, b - 1)]   # vertical band interior
    for side in ("L", "R"):
        if side == "L":
            region = I[:, :a]; c0, c1 = 0, b - 1
        else:
            region = I[:, b + 1:]; c0, c1 = a + 1, w - 1
        if region.shape[1] == 0:
            continue
        flags = [bool(np.all(region[r] == bgc)) for r in range(h)]
        for s, e in _runs(flags):
            if e - s + 1 >= 3:
                rects.append(((s + 1) if s > 0 else s, c0, e - 1, c1))
    return bgc, rects


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, spi_zero=None) -> dict:
    from random import randint, choice, sample
    if spi_zero is None:
        spi_zero = random.choice([True, False])

    def unif(lo, hi):
        lo, hi = int(lo), int(hi)
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))

    for _ in range(200):
        h = unif(18, max(18, min(30, max_h)))
        w = unif(18, max(18, min(30, max_w)))
        lb, ub = int(0.4 * h * w), int(0.5 * h * w)
        g = np.full((h, w), noisec, dtype=int)
        inds = [(i, j) for i in range(h) for j in range(w)]
        for (i, j) in sample(inds, unif(lb, ub)):
            g[i, j] = bgc
        addn, addb = set(), set()
        for i in range(h - 2):
            for j in range(w - 2):
                blk = g[i:i + 3, j:j + 3]
                if np.all(blk == bgc):
                    addn.add((randint(0, 2) + i, randint(0, 2) + j))
                elif np.all(blk == noisec):
                    addb.add((randint(0, 2) + i, randint(0, 2) + j))
        for (i, j) in addn:
            g[i, j] = noisec
        for (i, j) in addb:
            g[i, j] = bgc
        gi = g.copy(); go = g.copy()
        dim = randint(randint(3, 8), 8)
        if w - dim - 4 < 3:
            continue
        locj = randint(3, w - dim - 4)
        spi = 0 if spi_zero else randint(3, h // 2)
        gi[spi:, locj:locj + dim] = bgc
        go[spi:, locj:locj + dim] = bgc
        go[(spi + 1 if spi > 0 else spi):, locj + 1:locj + dim - 1] = 3

        def hband(sgns):
            startloc = choice((spi, randint(spi + 3, h - 6)))
            hh = randint(3, min(8, h - startloc - 3))
            for sgn in sgns:
                for ii in range(startloc, startloc + hh):
                    cs = range(0, locj + 1) if sgn == -1 else range(locj, w)
                    for j in cs:
                        gi[ii, j] = bgc
                        if go[ii, j] != 3:
                            go[ii, j] = bgc
            for sgn in sgns:
                for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                    if sgn == -1:
                        go[ii, 0:locj + dim - 1] = 3
                    else:
                        go[ii, locj + 1:w] = 3

        sgns = choice(((-1,), (1,), (-1, 1)))
        hband(sgns)
        if len(sgns) == 1 and unif(0, 1) == 1:
            hband((-sgns[0],))
        # keep only instances whose band structure is unambiguous from the input
        dbg, rects = _detect(gi)
        if dbg != bgc:
            continue
        pred = gi.copy()
        for (r0, c0, r1, c1) in rects:
            pred[r0:r1 + 1, c0:c1 + 1] = 3
        if not np.array_equal(pred, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = None
    if examples:
        bgc, _ = _detect(examples[0][0])   # the episode's band colour, read from a demo input
    bgc, rects = _detect(I, bgc)
    if not rects:
        bgc, rects = _detect(I)
    # Unit of work: one ROW at a time. In each row, the band interiors that cross it
    # (vertical band interior + any horizontal band interior, all measured from I)
    # form one contiguous run; paint that run with 3, the colour the rule names.
    ops, sels = [], []
    for r in range(hi):
        cols = set()
        for (r0, c0, r1, c1) in rects:
            if r0 <= r <= r1:
                cols.update(range(c0, c1 + 1))
        if not cols:
            continue
        ops.append(3)
        sels.append(sel_of([(r, c) for c in sorted(cols)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
