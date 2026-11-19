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


# ----------------------------------------------------------------------------
# 1. colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """bgc / linc / fgc are the three roles the generator samples at random.

    All three carry meaning (background inside a block, the separator-line colour
    that the missing cells get stamped with, the mark colour), so all three are
    fixed once per episode.  Any of them may legitimately be 0 — the route below
    never lets a 0-coloured cell travel inside a clipboard."""
    cols = list(range(10))
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (4, max(4, min(6, max_h))))
    w = unifint(diff_lb, diff_ub, (4, max(4, min(6, max_w))))

    nh_ub = max(1, min(4, (max_h + 1) // (h + 1)))
    nh = unifint(diff_lb, diff_ub, (1, nh_ub))
    nw_ub = max(1, min(3, (max_w + 1) // (w + 1)))
    nw_lb = 1 if nh > 1 else 2
    if nw_lb > nw_ub:
        nw_lb = nw_ub
    nw = unifint(diff_lb, diff_ub, (nw_lb, nw_ub))
    if nh == 1 and nw == 1:                      # need at least two blocks
        if nh_ub >= 2:
            nh = 2
        elif nw_ub >= 2:
            nw = 2

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))

    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(sorted(llocs))
    srcloc = choice(llocs)
    remlocs = remove(srcloc, llocs)

    ncells = unifint(diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    smallc2 = canvas(bgc, (h - 2, w - 2))
    inds = asindices(smallc2)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))

    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    go = tuple(e for e in gi)

    ncc = ncells + 1
    remlocs = totuple(remlocs)
    for k, rl in enumerate(remlocs):
        if k == 0:
            nleft = randint(0, ncc - 1)      # guarantees at least one real change
        else:
            nleft = randint(0, ncc)
        subobj = sample(totuple(shp), nleft)
        sg2 = asobject(fill(smallc, fgc, subobj))
        gi = paint(gi, shift(sg2, rl))
        go = paint(go, shift(gg2, rl))
        go = fill(go, fgc, shift(subobj, rl))

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """Rule (read off I alone): the grid is cut by constant lines of colour `linc`
    into equal blocks.  One block carries the complete mark-shape; every other
    block carries a subset of it.  Each incomplete block gets the shape completed,
    the cells it was missing drawn in `linc`.

    The ROUTE through that rule is chosen from the input: which block first,
    which way to sweep, whether to draw only what is missing or to lay the whole
    template down and re-draw the marks on top, whether to stamp one block and
    copy it to the rest.  Same input -> same route; different inputs differ."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- separator colour: colours of fully constant lines, least common -----
    cand = []
    for r in range(H):
        if len(set(I[r, :].tolist())) == 1:
            cand.append(int(I[r, 0]))
    for c in range(W):
        if len(set(I[:, c].tolist())) == 1:
            cand.append(int(I[0, c]))
    if not cand:
        return [34], [[0, 0, H - 1, W - 1]]
    cnt = Counter(cand)
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    srows = set(r for r in range(H) if all(int(v) == linc for v in I[r, :]))
    scols = set(c for c in range(W) if all(int(v) == linc for v in I[:, c]))

    def bands(n, seps):
        res, cur = [], []
        for i in range(n):
            if i in seps:
                if cur:
                    res.append((cur[0], len(cur)))
                    cur = []
            else:
                cur.append(i)
        if cur:
            res.append((cur[0], len(cur)))
        return res

    rbands = bands(H, srows)
    cbands = bands(W, scols)

    # --- background / mark colour inside the blocks --------------------------
    inner = []
    for (r0, bh) in rbands:
        for (c0, bw) in cbands:
            for r in range(r0, r0 + bh):
                for c in range(c0, c0 + bw):
                    inner.append(int(I[r, c]))
    icnt = Counter(inner)
    bgc = icnt.most_common(1)[0][0]
    fg_candidates = [k for k in icnt if k != bgc]
    if not fg_candidates:
        return [34], [[0, 0, H - 1, W - 1]]
    fgc = max(fg_candidates, key=lambda k: (icnt[k], -k))

    # --- per-block mark sets, in block-local coordinates ---------------------
    blocks = []
    for (r0, bh) in rbands:
        for (c0, bw) in cbands:
            rel = set()
            for r in range(bh):
                for c in range(bw):
                    if int(I[r0 + r, c0 + c]) == fgc:
                        rel.add((r, c))
            blocks.append({"r0": r0, "c0": c0, "bh": bh, "bw": bw, "rel": rel})

    # --- the complete one: its marks contain every other block's marks -------
    src_i = None
    for i, b in enumerate(blocks):
        if all(o["rel"] <= b["rel"] for o in blocks):
            src_i = i
            break
    if src_i is None:
        src_i = max(range(len(blocks)), key=lambda i: len(blocks[i]["rel"]))
    src = blocks[src_i]
    shape = set(src["rel"])
    shape_sorted = sorted(shape)

    targets = [b for i, b in enumerate(blocks)
               if i != src_i and (shape - b["rel"])]
    if not targets:
        return [34], [[0, 0, H - 1, W - 1]]

    # --- route, read off the input -------------------------------------------
    fgcount = int((I == fgc).sum())
    style = (int(linc) * 2 + int(fgc) + fgcount) % 4
    orderk = (int(bgc) * 3 + len(blocks) + src["r0"] + src["c0"]) % 4
    downward = ((int(bgc) + int(fgc)) % 2 == 0)

    if orderk == 0:                                    # reading order
        targets.sort(key=lambda b: (b["r0"], b["c0"]))
    elif orderk == 1:                                  # column by column
        targets.sort(key=lambda b: (b["c0"], b["r0"]))
    elif orderk == 2:                                  # backwards from the end
        targets.sort(key=lambda b: (-b["r0"], -b["c0"]))
    else:                                              # outwards from the model
        targets.sort(key=lambda b: (abs(b["r0"] - src["r0"]) + abs(b["c0"] - src["c0"]),
                                    b["r0"], b["c0"]))

    def abs_cells(b, rels):
        return [(b["r0"] + r, b["c0"] + c) for (r, c) in rels]

    same_size = len(set((b["bh"], b["bw"]) for b in blocks)) == 1
    # A clipboard never carries a 0 cell, so the stamp may only be copied when
    # the stamp colour itself is non-zero; otherwise the Paste would deliver
    # nothing and the route falls back to drawing each block by hand.
    copy_ok = (style == 2 and same_size and int(linc) != 0 and len(targets) >= 3)
    if style == 2 and not copy_ok:
        style = 1

    if style == 0:
        # draw straight onto the grid: only the marks each block lacks
        for b in targets:
            ops.append(int(linc))
            sels.append(sel_of(abs_cells(b, sorted(shape - b["rel"]))))

    elif style == 1:
        # lay the whole template down in linc, then redraw the marks that were
        # already there on top of it
        for b in targets:
            ops.append(int(linc))
            sels.append(sel_of(abs_cells(b, shape_sorted)))
            present = sorted(b["rel"])
            if present:
                ops.append(int(fgc))
                sels.append(sel_of(abs_cells(b, present)))

    elif style == 2:
        # stamp the template once, copy that block onto the others
        t0 = targets[0]
        ops.append(int(linc))
        sels.append(sel_of(abs_cells(t0, shape_sorted)))
        # whole block rectangle on purpose (background included) -> bbox form
        ops.append(29)
        sels.append([t0["r0"], t0["c0"], t0["bh"] - 1, t0["bw"] - 1])
        present0 = sorted(t0["rel"])
        if present0:
            ops.append(int(fgc))
            sels.append(sel_of(abs_cells(t0, present0)))
        for b in targets[1:]:
            ops.append(30)
            sels.append([b["r0"], b["c0"], 0, 0])
            # any cell of the stamped block that has to read 0 but could not
            # travel in the clipboard is put down by hand (none when bgc != 0)
            blind = []
            if int(bgc) == 0:
                for r in range(b["bh"]):
                    for c in range(b["bw"]):
                        if (r, c) in shape:
                            continue
                        if int(I[b["r0"] + r, b["c0"] + c]) != 0:
                            blind.append((b["r0"] + r, b["c0"] + c))
            if blind:
                ops.append(0)
                sels.append(sel_of(blind))
            present = sorted(b["rel"])
            if present:
                ops.append(int(fgc))
                sels.append(sel_of(abs_cells(b, present)))

    else:
        # fill each block in line by line
        for b in targets:
            miss = shape - b["rel"]
            rows = sorted({r for (r, _) in miss}, reverse=not downward)
            for r in rows:
                line = sorted((r, c) for (rr, c) in miss for _ in (0,) if rr == r)
                ops.append(int(linc))
                sels.append(sel_of(abs_cells(b, line)))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
