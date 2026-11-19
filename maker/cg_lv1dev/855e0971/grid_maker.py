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
from maker.sel_helpers import sel_of

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Only the dot colour is a role colour. Bar colours are just "whatever each
    # bar is" and stay random inside generate().
    dotc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        examples = [dict(v) for v in ORIENTS]
        examples += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, dotc=None, transpose=None) -> dict:
    from random import choice, randint, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if dotc is None:
        dotc = random.choice(range(10))
    if transpose is None:
        transpose = random.choice((True, False))

    # In the transposed case the generated height becomes the width.
    lim_h = max_w if transpose else max_h
    lim_w = max_h if transpose else max_w

    cols = list(range(10))
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, max(3, lim_h // 2))
    h = unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, lim_h)))
    h = min(h, lim_h)
    w = unifint(diff_lb, diff_ub, (3, max(3, lim_w)))
    w = min(w, lim_w)
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = randint(0, nbars - 1)
        barsizes[j] += 1
    remcols = [c for c in cols if c != dotc]
    lastcol = -1
    nloclbs = [choice((0, 1)) for _ in range(nbars)]
    if sum(nloclbs) < 2:
        l1, l2 = sample(range(nbars), 2)
        nloclbs[l1] = 1
        nloclbs[l2] = 1
    gi_rows, go_rows = [], []
    for bs, nloclb in zip(barsizes, nloclbs):
        col = choice([c for c in remcols if c != lastcol])
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        nl = min(nl, w)
        for jj in sample(range(w), nl):
            gim[randint(0, bs - 1)][jj] = dotc
            for rr in range(bs):
                gom[rr][jj] = dotc
        lastcol = col
        gi_rows += gim
        go_rows += gom
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _bands(G):
    # Horizontal-bar hypothesis: there is one colour d (the dot colour) such that
    # every row is "one bar colour + maybe d". Bars are maximal runs of rows that
    # share the same bar colour (the generator never puts two bars of the same
    # colour next to each other).
    G = np.asarray(G)
    for d in sorted(set(G.flatten().tolist())):
        rest = []
        ok = True
        for row in G:
            s = set(row.tolist()) - {d}
            if len(s) != 1:
                ok = False
                break
            rest.append(s.pop())
        if not ok:
            continue
        bands, s = [], 0
        for i in range(1, len(rest) + 1):
            if i == len(rest) or rest[i] != rest[s]:
                bands.append((s, i - 1))
                s = i
        return d, bands
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Orientation and dot colour come from I alone: bars run along rows if every
    # row is one bar colour plus dots; otherwise they run along columns.
    horizontal = True
    res = _bands(I)
    if res is None:
        res = _bands(I.T)
        horizontal = False
    dotc, bands = res
    G = I if horizontal else I.T

    # Row-wise reading: each line of a bar becomes the union of that bar's lines.
    # So every line of the bar takes the dot colour at every position where the
    # bar has a dot anywhere. Work bar by bar, one line at a time.
    for (r0, r1) in bands:
        dot_pos = sorted({c for r in range(r0, r1 + 1)
                          for c in range(G.shape[1]) if G[r, c] == dotc})
        for r in range(r0, r1 + 1):
            # skip only cells that already hold the dot colour (no-op cells)
            cells = [(r, c) for c in dot_pos if G[r, c] != dotc]
            if not cells:
                continue
            if not horizontal:
                cells = [(c, r) for (r, c) in cells]
            ops.append(int(dotc))
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
