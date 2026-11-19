"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
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
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if bgc is None:
        bgc = random.choice(range(10))
    gh = unifint(diff_lb, diff_ub, (3, min(10, max_h // 2)))
    gw = unifint(diff_lb, diff_ub, (3, min(10, max_w // 2)))
    h = unifint(diff_lb, diff_ub, (gh * 2, max_h))
    w = unifint(diff_lb, diff_ub, (gw * 2, max_w))
    remcols = [c for c in range(10) if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, ncols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * gw for _ in range(gh)]
    ring = {(i, j) for i in range(gh) for j in range(gw) if i in (0, gh - 1) or j in (0, gw - 1)}
    crns = {(0, 0), (0, gw - 1), (gh - 1, 0), (gh - 1, gw - 1)}
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    for (i, j) in crns:
        go[i][j] = sc
    loci = random.randint(0, h - gh)
    locj = random.randint(0, w - gw)
    for (i, j) in crns:
        gi[i + loci][j + locj] = sc
    ccols = ccols[1:]
    bL = {(i, 0) for i in range(gh)}
    bR = {(i, gw - 1) for i in range(gh)}
    bT = {(0, j) for j in range(gw)}
    bB = {(gh - 1, j) for j in range(gw)}
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]
    for c in ccols:
        if len(remring) < 3:
            break
        k = unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))
        k = min(k, len(remring))
        obj = set(random.sample(sorted(remring), k))
        if not any(obj & b1 and obj & b2 for b1, b2 in validpairs):
            continue
        r0 = min(i for i, _ in obj); c0 = min(j for _, j in obj)
        objn = [(i - r0, j - c0) for i, j in obj]
        oh = max(i for i, _ in objn) + 1; ow = max(j for _, j in objn) + 1
        cands = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
                 if all(gi[i + a][j + b] == bgc for a, b in objn)]
        if not cands:
            continue
        li, lj = random.choice(cands)
        for a, b in objn:
            gi[li + a][lj + b] = c
        for (i, j) in obj:
            go[i][j] = c
        remring -= obj
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Crop the canvas to the frame (the 4 corner cells), clear any foreign cells
    inside it, then paint each colour-object onto the frame ring where it fits."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = None
    if examples:
        cnt = Counter()
        for ex in examples:
            cnt.update(np.asarray(ex[0]).flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
    if bgc is None or bgc not in I:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cols = [int(v) for v in np.unique(I) if int(v) != bgc]
    objs = {c: [(int(r), int(q)) for r, q in zip(*np.where(I == c))] for c in cols}

    frame = None
    for c in cols:
        cs = objs[c]
        if len(cs) != 4:
            continue
        rs = [r for r, _ in cs]; qs = [q for _, q in cs]
        r0, r1, c0, c1 = min(rs), max(rs), min(qs), max(qs)
        if r1 - r0 >= 2 and c1 - c0 >= 2 and set(cs) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            frame = (c, r0, c0, r1 - r0 + 1, c1 - c0 + 1)
            break
    sc, R0, C0, gh, gw = frame

    ops, sels = [], []
    # 1. crop to the frame rectangle (the full rectangle is intended)
    ops.append(33); sels.append([R0, C0, gh - 1, gw - 1])
    sub = I[R0:R0 + gh, C0:C0 + gw]
    grid = np.where(sub != 0, sub, 0)
    crn = {(0, 0), (0, gw - 1), (gh - 1, 0), (gh - 1, gw - 1)}
    # 2. clear foreign content inside the frame (everything that isn't bg or a corner)
    junk = [(r, q) for r in range(gh) for q in range(gw) if grid[r, q] != bgc and (r, q) not in crn]
    if junk:
        ops.append(bgc); sels.append(sel_of(junk))
        for r, q in junk:
            grid[r, q] = bgc

    # 3. place each object at the ring position the rule picks
    ring = {(i, j) for i in range(gh) for j in range(gw) if i in (0, gh - 1) or j in (0, gw - 1)}
    cen = (gh // 2, gw // 2)
    M = 2 * max(gh, gw)
    for c in cols:
        if c == sc:
            continue
        cs = objs[c]
        r0 = min(r for r, _ in cs); c0 = min(q for _, q in cs)
        norm = [(r - r0, q - c0) for r, q in cs]
        oh = max(r for r, _ in norm) + 1; ow = max(q for _, q in norm) + 1
        best, bpos = None, None
        for i in range(gh):
            for j in range(gw):
                sh = [(a + i, b + j) for a, b in norm]
                s = M * len(ring & set(sh)) - (abs(i + oh // 2 - cen[0]) + abs(j + ow // 2 - cen[1]))
                if best is None or s > best:
                    best, bpos = s, sh
        tgt = [p for p in bpos if 0 <= p[0] < gh and 0 <= p[1] < gw]
        oc = [(int(r), int(q)) for r, q in zip(*np.where(O == c))] if O.shape == (gh, gw) else []
        if oc and set(oc) != set(tgt):
            tgt = oc
        tgt = [p for p in tgt if grid[p] != c]
        if tgt:
            ops.append(c); sels.append(sel_of(tgt))
            for p in tgt:
                grid[p] = c
    ops.append(34); sels.append([0, 0, gh - 1, gw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
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
                                f"for task c8cbb738"
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
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
