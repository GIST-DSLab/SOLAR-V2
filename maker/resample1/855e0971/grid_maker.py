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


def sample_colors(num_examples=None) -> dict:
    dotc = random.randint(0, 9)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, dotc=None, transpose=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if dotc is None:
        dotc = random.randint(0, 9)
    if transpose is None:
        transpose = random.choice([True, False])
    # stacked dimension (bars stacked along) and length dimension
    Hlim = max_w if transpose else max_h
    Wlim = max_h if transpose else max_w
    Hlim = min(Hlim, 30)
    Wlim = min(Wlim, 30)
    cols = list(range(10))
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = random.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = min(nbars, max(3, Hlim // 2))
    h = unifint(diff_lb, diff_ub, (nbars, Hlim))
    h = max(h, 2 * nbars)
    h = min(h, Hlim) if Hlim >= 2 * nbars else h
    w = unifint(diff_lb, diff_ub, (3, Wlim))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[random.randint(0, nbars - 1)] += 1
    remcols = [c for c in cols if c != dotc]
    lastcol = -1
    nloclbs = [random.choice((0, 1)) for _ in barsizes]
    if sum(nloclbs) < 2:
        l1, l2 = random.sample(range(len(nloclbs)), 2)
        nloclbs[l1] = 1
        nloclbs[l2] = 1
    gi_rows, go_rows = [], []
    for bs, nlb in zip(barsizes, nloclbs):
        col = random.choice([c for c in remcols if c != lastcol])
        gim = np.full((bs, w), col, dtype=int)
        gom = np.full((bs, w), col, dtype=int)
        nl = unifint(diff_lb, diff_ub, (nlb, max(nlb, w // 2)))
        for jj in random.sample(range(w), nl):
            r = random.randint(0, bs - 1)
            gim[r, jj] = dotc
            gom[:, jj] = dotc
        lastcol = col
        gi_rows.append(gim)
        go_rows.append(gom)
    gi = np.vstack(gi_rows)
    go = np.vstack(go_rows)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _bars_rows(G, dotc):
    """If bars are horizontal (row runs) treating dotc as dots, return list of (r0, r1); else None."""
    h, w = G.shape
    rowcol = []
    for r in range(h):
        vals = set(int(v) for v in G[r] if v != dotc)
        if len(vals) != 1:
            return None
        rowcol.append(vals.pop())
    if dotc in rowcol:
        return None
    bars = []
    s = 0
    for r in range(1, h + 1):
        if r == h or rowcol[r] != rowcol[s]:
            bars.append((s, r - 1))
            s = r
    if len(bars) < 2:
        return None
    return bars


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    dotc = None
    if examples:
        from collections import Counter
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                diff = ei != eo
                cnt.update(eo[diff].tolist())
        if cnt:
            dotc = cnt.most_common(1)[0][0]

    candidates = [dotc] if dotc is not None else []
    candidates += [c for c in range(10) if c != dotc and c in set(I.flatten().tolist())]

    orient = None
    bars = None
    for c in candidates:
        b = _bars_rows(I, c)
        if b is not None:
            dotc, orient, bars = c, "rows", b
            break
        b = _bars_rows(I.T, c)
        if b is not None:
            dotc, orient, bars = c, "cols", b
            break

    ops, sels = [], []
    if bars is not None:
        G = I if orient == "rows" else I.T
        gh, gw = G.shape
        for (r0, r1) in bars:
            for c in range(gw):
                if any(G[r, c] == dotc for r in range(r0, r1 + 1)):
                    line = [(r, c) for r in range(r0, r1 + 1)]
                    if all(G[r, c] == dotc for r, c in line):
                        continue
                    if orient == "cols":
                        line = [(cc, rr) for rr, cc in line]
                    ops.append(dotc)
                    sels.append(sel_of(line))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
