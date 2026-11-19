"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# structural variants: the generator either leaves the bars horizontal or
# dmirrors (transposes) the whole thing, making the bars vertical.
# ----------------------------------------------------------------------------
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # only `dotc` is a color role the rule depends on (the dot / line colour);
    # the bar colours are drawn per bar and carry no rule meaning.
    dotc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"dotc": dotc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(a + (b - a) * diff_lb)
    hi = int(a + (b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, dotc, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])

    # build in "horizontal bars" space; if it will be transposed the final
    # dimensions swap, so swap the limits up front.
    lim_h = max_w if transposed else max_h
    lim_w = max_h if transposed else max_w
    lim_h = max(6, min(30, int(lim_h)))
    lim_w = max(3, min(30, int(lim_w)))

    cols = list(range(10))

    nbarsd = _unifint(diff_lb, diff_ub, (1, 4))
    nbars = random.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, lim_h // 2)
    nbars = max(3, nbars)

    h = _unifint(diff_lb, diff_ub, (nbars, lim_h))
    w = _unifint(diff_lb, diff_ub, (3, lim_w))

    barsizes = [2] * nbars
    while sum(barsizes) < h and sum(barsizes) < lim_h:
        barsizes[random.randint(0, nbars - 1)] += 1

    remcols = [c for c in cols if c != dotc]
    lastcol = -1

    nloclbs = [random.choice((0, 1)) for _ in range(nbars)]
    if sum(nloclbs) < 2:
        i1, i2 = random.sample(range(nbars), 2)
        nloclbs[i1] = 1
        nloclbs[i2] = 1

    gi = []
    go = []
    for bs, nloclb in zip(barsizes, nloclbs):
        choices = [c for c in remcols if c != lastcol]
        col = random.choice(choices)
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        hi_nl = max(nloclb, w // 2)
        nl = _unifint(diff_lb, diff_ub, (nloclb, hi_nl))
        nl = max(0, min(nl, w))
        chlocs = random.sample(range(w), nl)
        for jj in chlocs:
            ri = random.randint(0, bs - 1)
            gim[ri][jj] = dotc
            for rr in range(bs):          # vfrontier inside the bar
                gom[rr][jj] = dotc
        lastcol = col
        gi.extend(gim)
        go.extend(gom)

    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


# ----------------------------------------------------------------------------
# derivation  (reads I only; O is never inspected)
# ----------------------------------------------------------------------------
def _analyze(G):
    """G: 2-D int array laid out as horizontal bars.
    Returns (dotc, [(r0, r1), ...]) or None."""
    h, w = G.shape
    rowsets = [set(int(v) for v in G[r].tolist()) for r in range(h)]
    if any(len(s) > 2 for s in rowsets):
        return None
    two = [s for s in rowsets if len(s) == 2]
    if not two:
        return None
    inter = set(two[0])
    for s in two[1:]:
        inter &= s
    if not inter:
        return None
    counts = Counter(int(v) for v in G.flatten().tolist())
    for cand in sorted(inter, key=lambda c: counts[c]):
        rowcol = []
        ok = True
        for s in rowsets:
            rest = s - {cand}
            if len(rest) != 1:
                ok = False
                break
            rowcol.append(next(iter(rest)))
        if not ok:
            continue
        bands = []
        start = 0
        for r in range(1, h + 1):
            if r == h or rowcol[r] != rowcol[r - 1]:
                bands.append((start, r - 1))
                start = r
        if len(bands) < 3:
            continue
        if any(b1 - b0 + 1 < 2 for b0, b1 in bands):
            continue
        return cand, bands
    return None


def _analyze_lenient(G):
    """Fallback, still input-only: majority colour per row gives the bars,
    rarest colour is the dot colour."""
    h, w = G.shape
    counts = Counter(int(v) for v in G.flatten().tolist())
    dotc = min(counts, key=lambda c: counts[c])
    rowcol = []
    for r in range(h):
        cc = Counter(int(v) for v in G[r].tolist())
        if len(cc) > 1 and cc.most_common(1)[0][0] == dotc:
            cc.pop(dotc)
        rowcol.append(cc.most_common(1)[0][0])
    bands = []
    start = 0
    for r in range(1, h + 1):
        if r == h or rowcol[r] != rowcol[r - 1]:
            bands.append((start, r - 1))
            start = r
    merged = []
    for b in bands:
        if merged and (b[1] - b[0] + 1) < 2:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return dotc, merged


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # 1) orientation: are the bars rows or columns of I?  (decided from I alone)
    horizontal = True
    res = _analyze(I)
    if res is None:
        res = _analyze(I.T)
        if res is not None:
            horizontal = False
    if res is None:
        rows_bad = sum(1 for r in range(hi) if len(set(I[r].tolist())) > 2)
        cols_bad = sum(1 for c in range(wi) if len(set(I[:, c].tolist())) > 2)
        horizontal = rows_bad <= cols_bad
        res = _analyze_lenient(I if horizontal else I.T)

    dotc, bands = res
    color_op = int(dotc) % 10

    # 2) every dot grows into a full line across its own bar
    if horizontal:
        for (r0, r1) in bands:
            dot_cols = sorted({c for r in range(r0, r1 + 1)
                               for c in range(wi) if I[r, c] == dotc})
            for c in dot_cols:
                cells = [(r, c) for r in range(r0, r1 + 1)]
                ops.append(color_op)
                sels.append(sel_of(cells))
    else:
        for (c0, c1) in bands:
            dot_rows = sorted({r for c in range(c0, c1 + 1)
                               for r in range(hi) if I[r, c] == dotc})
            for r in dot_rows:
                cells = [(r, c) for c in range(c0, c1 + 1)]
                ops.append(color_op)
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
