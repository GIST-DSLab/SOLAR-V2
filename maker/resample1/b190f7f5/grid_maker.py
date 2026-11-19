"""
ARC Task: b190f7f5 (RE-ARC) — LLM-generated grid_maker
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
import random, math
import numpy as np
from maker.sel_helpers import sel_of

def sample_colors(num_examples=None) -> dict:
    bgc = random.randint(0, 9)
    srccol = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "srccol": srccol}

def _valid_splits(gi):
    I = np.asarray(gi); hi, wi = I.shape; n = 0
    parts = []
    if hi % 2 == 0: parts.append((I[:hi // 2], I[hi // 2:]))
    if wi % 2 == 0: parts.append((I[:, :wi // 2], I[:, wi // 2:]))
    for A, B in parts:
        pa, pb = set(A.flatten().tolist()), set(B.flatten().tolist())
        if len(pa & pb) == 1 and min(len(pa), len(pb)) == 2 and max(len(pa), len(pb)) >= 3:
            n += 1
    return n

def _generate(diff_lb, diff_ub, max_h, max_w, bgc=None, srccol=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a: b = a
        lo = a + round((b - a) * lb); hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    if bgc is None: bgc = random.randint(0, 9)
    cols = [c for c in range(10) if c != bgc]
    if srccol is None or srccol == bgc: srccol = random.choice(cols)
    hmax = max(2, min(5, math.isqrt(max_h), max_h // 2))
    wmax = max(2, min(5, math.isqrt(max_w), max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcd, h * w - numcd)); numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc2 = random.choice((numcd2, h * w - numcd2)); numc2 = min(max(2, numc2), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    srclocs = random.sample(inds, numc)
    remcols = [c for c in cols if c != srccol]
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    trglocs = random.sample(inds, numc2)
    ccols = random.sample(remcols, numcols)
    fixc1 = random.choice(ccols)
    trgcols = [fixc1, random.choice([c for c in ccols if c != fixc1])] + [random.choice(ccols) for _ in trglocs[2:]]
    gisrc = [[bgc] * w for _ in range(h)]
    for i, j in srclocs: gisrc[i][j] = srccol
    gitrg = [[bgc] * w for _ in range(h)]
    for (i, j), c in zip(trglocs, trgcols): gitrg[i][j] = c
    ordd = random.choice(([gisrc, gitrg], [gitrg, gisrc]))
    if random.random() < 0.5:
        gi = [ordd[0][i] + ordd[1][i] for i in range(h)]
    else:
        gi = [list(r) for r in ordd[0]] + [list(r) for r in ordd[1]]
    go = [[bgc] * (w * w) for _ in range(h * h)]
    for a, b in trglocs:
        for i, j in srclocs:
            go[a * h + i][b * w + j] = gitrg[a][b]
    return {"input": [list(r) for r in gi], "output": go}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, srccol=None) -> dict:
    d = None
    for _ in range(100):
        d = _generate(diff_lb, diff_ub, max_h, max_w, bgc, srccol)
        if _valid_splits(d["input"]) == 1:
            return d
    return d

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cands = []
    if hi % 2 == 0:
        h, w = hi // 2, wi
        cands.append(((0, 0), (h, 0), h, w))
    if wi % 2 == 0:
        h, w = hi, wi // 2
        cands.append(((0, 0), (0, w), h, w))
    best = None
    for (p, q, h, w) in cands:
        A = I[p[0]:p[0] + h, p[1]:p[1] + w]; B = I[q[0]:q[0] + h, q[1]:q[1] + w]
        pa, pb = set(A.flatten().tolist()), set(B.flatten().tolist())
        score = 0
        if len(pa & pb) == 1: score += 1
        if min(len(pa), len(pb)) == 2 and max(len(pa), len(pb)) >= 3: score += 1
        if min(h, w) > 1: score += 1
        if best is None or score > best[0]:
            src, trg = (A, B) if len(pa) <= len(pb) else (B, A)
            best = (score, src, trg, h, w, (pa & pb))
    _, src, trg, h, w, common = best
    bgc = next(iter(common)) if common else int(np.bincount(I.flatten()).argmax())
    srclocs = [(i, j) for i in range(h) for j in range(w) if src[i, j] != bgc]
    ho, wo = h * h, w * w
    ops, sels = [], []
    # expand canvas to h^2 x w^2 (whole rectangle intended)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # lay the background base over the whole new canvas (whole rectangle intended)
    if bgc != 0:
        ops.append(bgc); sels.append([0, 0, ho - 1, wo - 1])
    else:
        rem = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != 0]
        if rem:
            ops.append(0); sels.append(sel_of(rem))
    # stamp the source shape into each target cell's block, coloured by that cell
    for a in range(h):
        for b in range(w):
            col = int(trg[a, b])
            if col == bgc or not srclocs:
                continue
            cells = [(a * h + i, b * w + j) for i, j in srclocs]
            ops.append(col); sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
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
                                f"for task b190f7f5"
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
                    f"Failed to build a complete episode for task b190f7f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b190f7f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
