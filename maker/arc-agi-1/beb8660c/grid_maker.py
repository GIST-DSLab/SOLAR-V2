"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    # discrete structural variant: which edge carries the 8-line (rotation of the whole scene)
    variants = [{"rot": r} for r in range(4)]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    import random

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    rot = rot % 4

    def ui(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int((hi - lo) * diff_lb)
        b = lo + int((hi - lo) * diff_ub)
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    # base grid is h x w with h >= w; a rot of 90/270 swaps the final dims
    if rot % 2 == 0:
        wcap = min(min(max_w, max_h), 30)
        hcap = min(max_h, 30)
    else:
        wcap = min(min(max_h, max_w), 30)
        hcap = min(max_w, 30)
    if wcap < 3:
        wcap = 3
    w = ui(3, wcap)
    if hcap < w:
        hcap = w
    h = ui(w, hcap)

    remcols = [c for c in range(10) if c != 8 and c != bgc]
    kmax = min(8, w - 1)
    k = ui(1, kmax)
    co = random.sample(remcols, k)
    wds = sorted(random.sample(list(range(1, w)), k))

    gi = [[bgc] * w for _ in range(h)]
    for j in range(k):
        rr = h - k - 1 + j
        for cc in range(wds[j]):
            gi[rr][cc] = co[j]
    gi[h - 1] = [8] * w

    go = [row[::-1] for row in gi]          # vmirror -> right anchored, sorted, 8-line at bottom

    body = [list(r) for r in gi[:-1]]
    random.shuffle(body)
    body.append([8] * w)
    gif = []
    for r in body:
        nbc = r.count(bgc)
        ofs = random.randint(0, nbc)
        if ofs == 0:
            gif.append(list(r))
        else:
            gif.append(r[-ofs:] + r[:-ofs])
    gi = [row[::-1] for row in gif]

    for _ in range(rot):                    # clockwise quarter turns
        gi = [list(x) for x in zip(*gi[::-1])]
        go = [list(x) for x in zip(*go[::-1])]

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # ---- normalise so the full 8-line is the TOP row -------------------------
    if np.all(I[0, :] == 8):
        nk = 0
    elif np.all(I[hi - 1, :] == 8):
        nk = 2
    elif np.all(I[:, 0] == 8):
        nk = 3
    else:
        nk = 1
    N = np.rot90(I, nk)
    RR = np.rot90(np.repeat(np.arange(hi).reshape(-1, 1), wi, axis=1), nk)
    CC = np.rot90(np.repeat(np.arange(wi).reshape(1, -1), hi, axis=0), nk)
    hn, wn = N.shape

    def to_orig(cells):
        return [(int(RR[r, c]), int(CC[r, c])) for (r, c) in cells]

    # ---- background = the colour present in the most rows --------------------
    bgc, bgn = 0, -1
    for c in np.unique(N).tolist():
        c = int(c)
        nrows = len(set(np.where(N == c)[0].tolist()))
        if nrows > bgn:
            bgn, bgc = nrows, c

    # ---- the bars (one straight segment per colour) --------------------------
    bars = []
    for c in np.unique(N).tolist():
        c = int(c)
        if c == bgc or c == 8:
            continue
        rs, cs = np.where(N == c)
        r = int(rs[0])
        c0 = int(cs.min())
        ln = int(cs.size)
        bars.append({"color": c, "len": ln,
                     "cells": [(r, c0 + t) for t in range(ln)]})
    bars.sort(key=lambda b: (-b["len"], b["color"]))
    for i, b in enumerate(bars):
        b["dst"] = [(i + 1, t) for t in range(b["len"])]   # stacked under the 8-line
        b["cur"] = list(b["cells"])
        b["seen"] = set(b["cells"])

    # ---- order the relocations: never land on a bar that has not moved yet ---
    steps = []
    pending = list(bars)
    guard = 0
    while pending and guard < 400:
        guard += 1
        pick = None
        for b in pending:
            blocked = False
            for o in pending:
                if o is not b and (set(b["dst"]) & set(o["cur"])):
                    blocked = True
                    break
            if not blocked:
                pick = b
                break
        if pick is not None:
            steps.append((pick, list(pick["cur"]), list(pick["dst"]), True))
            pick["cur"] = list(pick["dst"])
            pick["seen"] |= set(pick["dst"])
            pending.remove(pick)
            continue
        # deadlock (cyclic blocking): park one bar somewhere that blocks nothing
        parked = False
        for b in pending:
            if b["color"] == 0:
                continue
            busy, dests = set(), set()
            for o in pending:
                if o is not b:
                    busy |= set(o["cur"])
                    dests |= set(o["dst"])
            for r in range(1, hn):
                for c0 in range(0, wn - b["len"] + 1):
                    cand = [(r, c0 + t) for t in range(b["len"])]
                    if cand == b["cur"]:
                        continue
                    s = set(cand)
                    if (s & busy) or (s & dests):
                        continue
                    steps.append((b, list(b["cur"]), cand, True))
                    b["cur"] = cand
                    b["seen"] |= s
                    parked = True
                    break
                if parked:
                    break
            if parked:
                break
        if not parked:                      # last resort: place the rest directly
            for b in list(pending):
                steps.append((b, list(b["cur"]), list(b["dst"]), False))
                b["cur"] = list(b["dst"])
                b["seen"] |= set(b["dst"])
                pending.remove(b)
            break

    # ---- emit the relocations ------------------------------------------------
    sim = I.copy()
    order_ids, ordered = [], []
    for (b, src, dst, mv) in steps:
        if id(b) not in order_ids:
            order_ids.append(id(b))
            ordered.append(b)
        if src == dst:
            continue
        src_o = to_orig(src)
        dst_o = to_orig(dst)
        if mv and b["color"] != 0:
            dr = dst_o[0][0] - src_o[0][0]
            dc = dst_o[0][1] - src_o[0][1]
            first = True
            for _ in range(abs(dr)):
                ops.append(20 if dr < 0 else 21)
                sels.append(sel_of(src_o) if first else sel_of([]))
                first = False
            for _ in range(abs(dc)):
                ops.append(22 if dc > 0 else 23)
                sels.append(sel_of(src_o) if first else sel_of([]))
                first = False
            for (r, c) in src_o:
                sim[r, c] = 0               # ARCLE leaves the vacated footprint at 0
            for (r, c) in dst_o:
                sim[r, c] = b["color"]
        else:
            # colour-0 segments are invisible to the object ops -> paint them
            ops.append(b["color"])
            sels.append(sel_of(dst_o))
            for (r, c) in dst_o:
                sim[r, c] = b["color"]

    # ---- repair what each bar left behind and nothing else refilled ----------
    all_dst = set()
    for b in bars:
        all_dst |= set(b["dst"])
    for b in ordered:
        cells = []
        for cell in sorted(b["seen"] - set(b["dst"])):
            if cell in all_dst:
                continue
            r, c = to_orig([cell])[0]
            if sim[r, c] != bgc:
                cells.append((r, c))
        if cells:
            ops.append(bgc)
            sels.append(sel_of(cells))
            for (r, c) in cells:
                sim[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])      # full-grid rectangle for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
