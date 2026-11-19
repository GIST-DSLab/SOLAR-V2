"""
ARC Task: 363442ee (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo)); hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(hi_i, lo_i)))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    dotcol = random.choice(rem)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, dotcol) -> dict:
    M = min(max_h, max_w, 30)
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    w = _unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    while h > 1 and 2 * h > M:
        h -= 2
    while w > 1 and 3 * w + 1 > M:
        w -= 2
    nremh = _unifint(diff_lb, diff_ub, (2, max(2, M // h)))
    nremw = _unifint(diff_lb, diff_ub, (2, max(2, (M - w - 1) // w)))
    rsh, rsw = nremh * h, nremw * w
    remcols = [c for c in cols if c not in (bgc, barcol)]
    rsi = np.full((rsh, rsw), bgc, dtype=int)
    rso = rsi.copy()
    ls = np.full(((nremh - 1) * h, w), bgc, dtype=int)
    bar = np.full((rsh, 1), barcol, dtype=int)
    dotcands = [(i, j) for i in range(0, rsh, h) for j in range(0, rsw, w)]
    dev = _unifint(diff_lb, diff_ub, (1, max(1, len(dotcands) // 2)))
    ndots = random.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(dotcands, ndots)
    nfull = _unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = random.sample(remcols, nfull)
    ulc = np.array([[random.choice(fullremcols) for _ in range(w)] for _ in range(h)], dtype=int)
    for (i, j) in dots:
        rsi[i + h // 2, j + w // 2] = dotcol
        rso[i:i + h, j:j + w] = ulc
    left = np.vstack([ulc, ls])
    gi = np.hstack([left, bar, rsi])
    go = np.hstack([left, bar, rso])
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, -1),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Rule: a solid multi-coloured key block sits on one side of a full-length bar;
    every isolated dot on the other side becomes a copy of the key block centred
    on it.  Route: copy the key block once (CopyI); if it contains colour 0
    (which Paste cannot write) lay ONE Color0 base over all landing rectangles
    (the cells the copies land on, known from the dots in I); then Paste the
    block once per dot.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            cnt[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        if cnt:
            cand = cnt.most_common(1)[0][0]
            if cand in set(I.flatten().tolist()):
                bgc = cand

    # the bar: a full-length uniform line of a non-background colour
    bar = np.zeros((hi, wi), dtype=bool)
    for r in range(hi):
        v = int(I[r, 0])
        if v != bgc and np.all(I[r, :] == v):
            bar[r, :] = True
    for c in range(wi):
        v = int(I[0, c])
        if v != bgc and np.all(I[:, c] == v):
            bar[:, c] = True

    mask = (I != bgc) & (~bar)
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if mask[r0, c0] and not seen[r0, c0]:
                stack = [(r0, c0)]; seen[r0, c0] = True; cells = []
                while stack:
                    rr, cc = stack.pop(); cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True; stack.append((nr, nc))
                comps.append(cells)

    if len(comps) < 2:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    comps.sort(key=len, reverse=True)
    block = comps[0]                      # key block: the big solid rectangle
    br = min(r for r, _ in block); bc = min(c for _, c in block)
    bh = max(r for r, _ in block) - br + 1
    bw = max(c for _, c in block) - bc + 1

    centers = sorted(((min(r for r, _ in cp) + max(r for r, _ in cp)) // 2,
                      (min(c for _, c in cp) + max(c for _, c in cp)) // 2)
                     for cp in comps[1:])
    tops = [(r - bh // 2, c - bw // 2) for r, c in centers]

    blockvals = I[br:br + bh, bc:bc + bw]
    has_zero = bool(np.any(blockvals == 0))
    has_nonzero = bool(np.any(blockvals != 0))

    # the key block is a full rectangle -> its bbox is exactly its cell set
    if has_nonzero:
        ops.append(28); sels.append([br, bc, bh - 1, bw - 1])

    # base layer: one Color0 over every landing rectangle (Paste can't write 0)
    if has_zero:
        land = [(r, c) for (t, l) in tops
                for r in range(t, t + bh) for c in range(l, l + bw)
                if 0 <= r < hi and 0 <= c < wi]
        if any(I[r, c] != 0 for r, c in land):
            ops.append(0); sels.append(sel_of(land))

    # one copy per dot, landing centred on the dot
    if has_nonzero:
        for (t, l) in tops:
            ops.append(30); sels.append(sel_of([(t, l)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 363442ee"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 363442ee"
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
                                f"for task 363442ee"
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
                    f"Failed to build a complete episode for task 363442ee "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"363442ee-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
