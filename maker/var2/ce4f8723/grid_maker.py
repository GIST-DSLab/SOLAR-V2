"""
ARC Task: ce4f8723 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]      # 3 is the fill colour, never sampled
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    cola = random.choice(rem)
    colb = random.choice([c for c in rem if c != cola])

    # the bar being a column (halves side by side) or a row (halves stacked) is a
    # discrete structural variant -> both must be shown before the test instance
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barcol": barcol, "cola": cola, "colb": colb,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, cola, colb,
             mirrored=None) -> dict:
    if mirrored is None:
        mirrored = choice((True, False))

    if mirrored:                       # input ends up (2w+1, h) after dmirror
        h_ub = max(2, min(30, max_w))
        w_ub = max(2, min(14, (max_h - 1) // 2))
    else:                              # input is (h, 2w+1)
        h_ub = max(2, min(30, max_h))
        w_ub = max(2, min(14, (max_w - 1) // 2))

    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    deva = choice((+1, -1)) * deva
    devb = choice((+1, -1)) * devb
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 3, set(a) | set(b))
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Overlay the two halves of the input onto the first one: a cell marked in
    EITHER half becomes the fill colour, a cell plain in both keeps the shared
    background colour.

    The ROUTE is read off the instance itself, so one instance always gives one
    route and the variety comes from the instances differing:
      * crop the canvas down first, or paint on the full grid and crop last
        -> whether the bar is a row (halves stacked) or a column;
      * lay the fill colour over the whole canvas and put the plain cells back,
        or draw only the marks -> whichever of the two sets is the smaller;
      * which half's marks go down first -> the busier half is dealt with first;
      * sweep in rows or in columns -> the long way across the answer canvas;
      * sweep forwards or backwards -> start at the end where the bulk sits;
      * blob by blob or one straight sweep -> how sparse the set is.
    Nothing here looks at O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ── the separator: the one uniform line down the middle ──────────────────
    stacked = (hi % 2 == 1) and (len(set(I[hi // 2].tolist())) == 1)
    if stacked:                      # bar is a row -> halves are top / bottom
        k = hi // 2
        A, B = I[:k, :], I[k + 1:, :]
    else:                            # bar is a column -> halves are left / right
        k = wi // 2
        A, B = I[:, :k], I[:, k + 1:]
    kh, kw = A.shape                 # the answer canvas, already sitting at (0,0)

    # ── colour roles: the shared colour is the background of both halves ─────
    palA = set(A.flatten().tolist())
    palB = set(B.flatten().tolist())
    common = sorted(palA & palB)
    if len(common) == 1:
        bgc = common[0]
    else:
        from collections import Counter
        cnt = Counter(A.flatten().tolist()) + Counter(B.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]

    # the colour the marks are redrawn in — read off the demonstrations
    fillc = 3
    for pair in (examples or []):
        try:
            ein, eout = pair[0], pair[1]
        except Exception:
            continue
        new = (set(np.asarray(eout).flatten().tolist())
               - set(np.asarray(ein).flatten().tolist()))
        if len(new) == 1:
            fillc = int(new.pop())
            break

    mA = (A != bgc)
    mB = (B != bgc)
    nA, nB = int(mA.sum()), int(mB.sum())
    union = mA | mB
    n_marked = int(union.sum())
    n_plain = kh * kw - n_marked

    # ── route choices, all read from the input ───────────────────────────────
    lay_base = n_marked > n_plain    # mostly fill -> flood the canvas, punch holes
    crop_first = stacked             # stacked halves: shrink to the answer first
    by_rows = kw >= kh               # sweep the long way across the canvas

    ops, sels = [], []

    def key_of(rc, rev):
        r, c = rc
        pri, sec = (r, c) if by_rows else (c, r)
        return (-pri, -sec) if rev else (pri, sec)

    def components(cells, rev):
        left = set(cells)
        out = []
        for seed in sorted(cells, key=lambda p: key_of(p, rev)):
            if seed not in left:
                continue
            left.discard(seed)
            stack, comp = [seed], []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = (r + dr, c + dc)
                    if p in left:
                        left.discard(p)
                        stack.append(p)
            out.append(comp)
        return out

    def emit_group(cells, color, rev):
        """Paint one blob (or one whole field) line by line, each straight run
        of touching cells in one stroke."""
        lines = {}
        for (r, c) in cells:
            pri, sec = (r, c) if by_rows else (c, r)
            lines.setdefault(pri, []).append(sec)
        step = -1 if rev else 1
        for pri in sorted(lines, reverse=rev):
            secs = sorted(lines[pri], reverse=rev)
            start = prev = secs[0]
            runs = []
            for s in secs[1:]:
                if s == prev + step:
                    prev = s
                    continue
                runs.append((start, prev))
                start = prev = s
            runs.append((start, prev))
            for a, b in runs:
                lo, hi_ = (b, a) if rev else (a, b)
                run = [((pri, s) if by_rows else (s, pri)) for s in range(lo, hi_ + 1)]
                ops.append(color)            # Color<color> on exactly this run
                sels.append(sel_of(run))

    def emit(cells, color):
        if not cells:
            return                            # nothing of this kind here
        span = (kh if by_rows else kw) - 1
        bulk = sum((r if by_rows else c) for r, c in cells) / float(len(cells))
        rev = span > 0 and bulk > span / 2.0  # start where the work is
        # a sparse set reads as separate blobs — deal with them one blob at a
        # time; a dense one is just a field to sweep straight through
        groups = components(cells, rev) if 4 * len(cells) <= kh * kw else [cells]
        for g in groups:
            emit_group(g, color, rev)

    # full rectangle (background included) -> a bbox is exactly the intended cells
    canvas_box = [0, 0, kh - 1, kw - 1]

    if crop_first:
        ops.append(33)
        sels.append(canvas_box)

    if lay_base:
        # lay the fill colour over the whole answer canvas, then put the cells
        # that stay plain back in the background colour with an explicit
        # Color<bgc> — correct even when bgc is 0
        ops.append(fillc)
        sels.append(canvas_box)
        emit([(r, c) for r in range(kh) for c in range(kw) if not union[r, c]], bgc)
    else:
        # draw the marks: the busier half's marks first, then the ones the other
        # half adds on top (cells already carrying the fill colour are skipped)
        first, second = (mB, mA) if nB > nA else (mA, mB)
        emit([(r, c) for r in range(kh) for c in range(kw) if first[r, c]], fillc)
        emit([(r, c) for r in range(kh) for c in range(kw)
              if second[r, c] and not first[r, c]], fillc)

    if not crop_first:
        ops.append(33)
        sels.append(canvas_box)

    ops.append(34)
    sels.append(canvas_box)
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
                        f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
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
                                f"for task ce4f8723"
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
                    f"Failed to build a complete episode for task ce4f8723 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce4f8723-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
