"""
ARC Task: 780d0b14 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    nh = unifint(diff_lb, diff_ub, (2, 6))
    nw = unifint(diff_lb, diff_ub, (2, 6))
    nh = max(2, min(nh, (max_h + 1) // 3))
    nw = max(2, min(nw, (max_w + 1) // 3))
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (3, 9))
    ccols = sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(diff_lb, diff_ub, (3 * nh - 1, max(3 * nh - 1, max_h)))
    w = unifint(diff_lb, diff_ub, (3 * nw - 1, max(3 * nw - 1, max_w)))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        hdist[randint(0, nh - 1)] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        wdist[randint(0, nw - 1)] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    gi = dmirror(merge(tuple(repeat(r, c) + (repeat(bgc, h),) for r, c in zip(dmirror(gi), wdist)))[:-1])
    objs = objects(gi, T, F, F)
    bgobjs = colorfilter(objs, bgc)
    objs = objs - bgobjs
    base = gi
    for _ in range(200):
        gi = base
        for obj in objs:
            gi = fill(gi, bgc, sample(totuple(toindices(obj)), unifint(diff_lb, diff_ub, (1, len(obj) // 2))))
        # the separator lines must stay the only all-background rows/columns
        nfr = sum(1 for row in gi if len(set(row)) == 1)
        nfc = sum(1 for col in dmirror(gi) if len(set(col)) == 1)
        if nfr == nh - 1 and nfc == nw - 1:
            return {'input': gi, 'output': go}
    raise ValueError('noise erased a whole line')


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Reading: the grid is a lattice of blocks cut by full-length bgc lines.
    # Collapse it in two passes: first every band of rows is merged (union of
    # its rows) into one row, then every band of columns of that result is
    # merged into one column.  Union is done with transparent Copy/Paste, so
    # the background is first made "empty" (0).
    frow = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    fcol = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bgc = int(I[frow[0], 0]) if frow else int(I[0, fcol[0]])

    def bands(n, fl):
        res, start = [], None
        for i in range(n):
            if i in fl:
                if start is not None:
                    res.append((start, i - 1)); start = None
            elif start is None:
                start = i
        if start is not None:
            res.append((start, n - 1))
        return res

    rbands = bands(hi, set(frow))
    cbands = bands(wi, set(fcol))
    nh, nw = len(rbands), len(cbands)

    ops, sels = [], []
    if bgc != 0:
        # every background cell becomes empty so pasting never stamps it
        ops.append(0); sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]))

    # pass 1: merge each row band into row i
    for i, (r0, r1) in enumerate(rbands):
        if i > 0:
            # row i belongs to an already-merged band (or a frontier): wipe it
            ops.append(0); sels.append(sel_of([(i, c) for c in range(wi)]))
        for r in range(r0, r1 + 1):
            if i == 0 and r == 0:
                continue
            ops.append(29); sels.append([r, 0, 0, wi - 1])   # the whole row r (full rectangle)
            ops.append(30); sels.append([i, 0, 0, 0])
    # pass 2: merge each column band (within the nh merged rows) into column j
    for j, (c0, c1) in enumerate(cbands):
        if j > 0:
            ops.append(0); sels.append(sel_of([(r, j) for r in range(nh)]))
        for c in range(c0, c1 + 1):
            if j == 0 and c == 0:
                continue
            ops.append(29); sels.append([0, c, nh - 1, 0])   # column c of the merged rows (full rectangle)
            ops.append(30); sels.append([0, j, 0, 0])
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])

    # ---- drop every action that has no effect (judged on I only) ----
    def run(ol, sl):
        g = np.zeros((30, 30), dtype=int); g[:hi, :wi] = I
        dim = [hi, wi]; clip = None
        for op, s in zip(ol, sl):
            if op <= 9:
                for r, c in s["cells"]:
                    g[r, c] = op
            elif op == 29:
                r, c, h, w = s
                clip = g[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                if clip is None:
                    continue
                r, c = s[0], s[1]
                ch, cw = clip.shape
                ch = min(ch, 30 - r); cw = min(cw, 30 - c)
                sub = g[r:r + ch, c:c + cw]
                m = clip[:ch, :cw] != 0
                sub[m] = clip[:ch, :cw][m]
            elif op == 33:
                r, c, h, w = s
                blk = g[r:r + h + 1, c:c + w + 1].copy()
                g = np.zeros((30, 30), dtype=int)
                g[:h + 1, :w + 1] = blk
                dim = [h + 1, w + 1]
        return g[:dim[0], :dim[1]].copy()

    def strip_noop_cells(ol, sl):
        # shrink Color selections to cells that actually change at that moment
        g = np.zeros((30, 30), dtype=int); g[:hi, :wi] = I
        out_o, out_s = [], []
        for k, (op, s) in enumerate(zip(ol, sl)):
            if op <= 9:
                cells = [(r, c) for r, c in s["cells"] if g[r, c] != op]
                if not cells:
                    continue
                s = sel_of(cells)
            out_o.append(op); out_s.append(s)
            g2 = run(out_o, out_s)
            g = np.zeros((30, 30), dtype=int); g[:g2.shape[0], :g2.shape[1]] = g2
        return out_o, out_s

    ops, sels = strip_noop_cells(ops, sels)
    target = run(ops, sels)
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            tr_o = ops[:k] + ops[k + 1:]; tr_s = sels[:k] + sels[k + 1:]
            res = run(tr_o, tr_s)
            if res.shape == target.shape and np.array_equal(res, target):
                ops, sels = tr_o, tr_s
                changed = True
                break

    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
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
                                f"for task 780d0b14"
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
                    f"Failed to build a complete episode for task 780d0b14 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"780d0b14-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
