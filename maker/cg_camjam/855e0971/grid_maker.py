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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The dot colour is the one thing the rule paints with, so it is fixed for
    # the whole episode. Bar colours only describe the layout and may vary.
    dotc = random.choice(range(10))
    return {"dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, dotc=None, transpose=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))

    cols = list(range(10))
    if dotc is None:
        dotc = random.choice(cols)
    if transpose is None:
        transpose = random.choice((True, False))
    # Bars run along the rows here. Transposing swaps which limit applies to h and w.
    lim_h = max_w if transpose else max_h
    lim_w = max_h if transpose else max_w

    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = random.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, max(3, lim_h // 2))
    h = unifint(diff_lb, diff_ub, (max(nbars * 2, nbars), max(lim_h, nbars * 2)))
    w = unifint(diff_lb, diff_ub, (3, max(3, lim_w)))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = random.randint(0, nbars - 1)
        barsizes[j] += 1
    remcols = [c for c in cols if c != dotc]
    lastcol = -1
    nloclbs = [random.choice((0, 1)) for _ in range(nbars)]
    if sum(nloclbs) < 2:
        l1, l2 = random.sample(range(nbars), 2)
        nloclbs[l1] = 1
        nloclbs[l2] = 1
    gi_rows, go_rows = [], []
    for bs, nloclb in zip(barsizes, nloclbs):
        col = random.choice([c for c in remcols if c != lastcol])
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        chlocs = random.sample(range(w), nl)
        for jj in chlocs:
            i = random.randint(0, bs - 1)
            gim[i][jj] = dotc
            for k in range(bs):
                gom[k][jj] = dotc
        lastcol = col
        gi_rows += gim
        go_rows += gom
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _orientation_horizontal(I):
    # As in the verifier: the bars are horizontal when every row has fewer than 3 colours.
    return all(len(set(row.tolist())) < 3 for row in I)


def _dot_color(I, examples):
    # Read the dot colour from the demonstrations: it is the only colour that
    # appears in the cells the rule changed.
    if examples:
        cands = None
        for ex in examples:
            EI = np.asarray(ex[0], dtype=int)
            EO = np.asarray(ex[1], dtype=int)
            if EI.shape != EO.shape:
                continue
            ch = set(EO[EI != EO].tolist())
            if not ch:
                continue
            cands = ch if cands is None else (cands & ch)
        if cands:
            inI = [c for c in cands if (I == c).any()]
            if inI:
                return inI[0]
    # Fallback: the colour with the most cells that touch no cell of their own colour.
    hi, wi = I.shape
    cnt = Counter()
    for r in range(hi):
        for c in range(wi):
            v = I[r, c]
            iso = True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < hi and 0 <= cc < wi and I[rr, cc] == v:
                    iso = False
                    break
            if iso:
                cnt[int(v)] += 1
    return cnt.most_common(1)[0][0] if cnt else int(I[0, 0])


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    horizontal = _orientation_horizontal(I)
    G = I.copy() if horizontal else I.T.copy()   # work with the bars as rows
    H, W = G.shape
    dotc = _dot_color(I, examples)

    def to_orig(cells):
        return cells if horizontal else [(c, r) for r, c in cells]

    # A bar is a maximal run of consecutive rows that share the same non-dot colour.
    def bar_color(row):
        others = [v for v in row.tolist() if v != dotc]
        return Counter(others).most_common(1)[0][0] if others else None

    bcs = [bar_color(G[r]) for r in range(H)]
    bands = []
    start = 0
    for r in range(1, H + 1):
        if r == H or bcs[r] != bcs[start]:
            bands.append((start, r - 1))
            start = r

    # The dot colour spreads one row at a time inside each bar: first downward,
    # then upward. Each step paints the cells next to a dot-coloured cell in the
    # row just reached. The cells are read from the grid as it stands at that step.
    for (r0, r1) in bands:
        for r in range(r0, r1):            # downward sweep
            cells = [(r + 1, c) for c in range(W)
                     if G[r, c] == dotc and G[r + 1, c] != dotc]
            if cells:
                for rr, cc in cells:
                    G[rr, cc] = dotc
                ops.append(int(dotc) % 10)
                sels.append(sel_of(to_orig(cells)))
        for r in range(r1, r0, -1):        # upward sweep
            cells = [(r - 1, c) for c in range(W)
                     if G[r, c] == dotc and G[r - 1, c] != dotc]
            if cells:
                for rr, cc in cells:
                    G[rr, cc] = dotc
                ops.append(int(dotc) % 10)
                sels.append(sel_of(to_orig(cells)))

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
