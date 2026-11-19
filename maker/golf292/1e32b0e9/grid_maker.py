"""
ARC Task: 1e32b0e9 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, fgc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def _unif(diff_lb, diff_ub, lo, hi):
    d = random.uniform(diff_lb, diff_ub)
    return max(lo, min(hi, lo + int(round(d * (hi - lo)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=1, fgc=2, **kw) -> dict:
    h = _unif(diff_lb, diff_ub, 4, 6)
    w = _unif(diff_lb, diff_ub, 4, 6)
    nh = _unif(diff_lb, diff_ub, 1, 4)
    nw = _unif(diff_lb, diff_ub, 1 if nh > 1 else 2, 3)
    while nh > 1 and h * nh + nh - 1 > max_h:
        nh -= 1
    while nw > 1 and w * nw + nw - 1 > max_w:
        nw -= 1
    if nh * nw < 2:
        if w * 2 + 1 <= max_w:
            nw = 2
        else:
            nh = 2
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    gi = [[linc] * fullw for _ in range(fullh)]
    locs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(locs)
    remlocs = [l for l in locs if l != srcloc]
    ncells = _unif(diff_lb, diff_ub, 0, (h - 2) * (w - 2) - 1)
    inds = [(i, j) for i in range(h - 2) for j in range(w - 2)]
    sp = random.choice(inds)
    shp = {sp}
    for _ in range(ncells):
        cand = [p for p in inds if p not in shp and
                any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 or
                    (abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1) for q in shp)]
        if not cand:
            break
        shp.add(random.choice(cand))
    shp = [(i + 1, j + 1) for (i, j) in shp]
    for (a, b) in locs:
        for i in range(h):
            for j in range(w):
                gi[a + i][b + j] = bgc
    go = [row[:] for row in gi]
    for (i, j) in shp:
        gi[srcloc[0] + i][srcloc[1] + j] = fgc
        go[srcloc[0] + i][srcloc[1] + j] = fgc
    for (a, b) in remlocs:
        nleft = random.randint(0, len(shp))
        sub = random.sample(shp, nleft)
        for (i, j) in shp:
            go[a + i][b + j] = linc
        for (i, j) in sub:
            gi[a + i][b + j] = fgc
            go[a + i][b + j] = fgc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # separator colour: least common colour among fully-constant lines
    cand = []
    for r in range(H):
        if len(set(I[r, :].tolist())) == 1:
            cand.append(int(I[r, 0]))
    for c in range(W):
        if len(set(I[:, c].tolist())) == 1:
            cand.append(int(I[0, c]))
    cnt = Counter(cand)
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    srows = set(r for r in range(H) if all(int(v) == linc for v in I[r, :]))
    scols = set(c for c in range(W) if all(int(v) == linc for v in I[:, c]))

    def bands(n, seps):
        res, cur = [], []
        for i in range(n):
            if i in seps:
                if cur:
                    res.append(cur[0]); cur = []
            else:
                cur.append(i)
        if cur:
            res.append(cur[0])
        return res

    rb = bands(H, srows)
    cb = bands(W, scols)
    bh = next((r for r in range(rb[0], H) if r in srows), H) - rb[0]
    bw = next((c for c in range(cb[0], W) if c in scols), W) - cb[0]

    inner = Counter()
    for r0 in rb:
        for c0 in cb:
            inner.update(I[r0:r0 + bh, c0:c0 + bw].flatten().tolist())
    bgc = inner.most_common(1)[0][0]
    fgs = [k for k in inner if k != bgc]
    if not fgs:
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels
    fgc = max(fgs, key=lambda k: inner[k])

    blocks = []
    for r0 in rb:
        for c0 in cb:
            rel = {(r, c) for r in range(bh) for c in range(bw)
                   if int(I[r0 + r, c0 + c]) == fgc}
            blocks.append((r0, c0, rel))

    # source block: its shape contains every other block's cells
    src = None
    for b in blocks:
        if all(o[2] <= b[2] for o in blocks):
            src = b
            break
    if src is None:
        src = max(blocks, key=lambda b: len(b[2]))
    shape = src[2]

    # one stroke: the source shape stamped at every block origin, in the line colour,
    # except where the block already holds the foreground colour in I
    fg_in_I = {(r, c) for r in range(H) for c in range(W) if int(I[r, c]) == fgc}
    stamp = set()
    for (r0, c0, _) in blocks:
        for (r, c) in shape:
            stamp.add((r0 + r, c0 + c))
    target = sorted(stamp - fg_in_I)
    if target:
        ops.append(int(linc)); sels.append(sel_of(target))

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e32b0e9"
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
                                f"for task 1e32b0e9"
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
                    f"Failed to build a complete episode for task 1e32b0e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e32b0e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
