"""
ARC Task: 9f236235 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    frontcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "frontcol": frontcol}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + round((b - a) * lb)
    hi = a + round((b - a) * ub)
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, frontcol=1, **kw) -> dict:
    cols = list(range(10))
    numh = _unifint(diff_lb, diff_ub, (2, min(14, (max_h + 1) // 2)))
    numw = _unifint(diff_lb, diff_ub, (2, min(14, (max_w + 1) // 2)))
    h = _unifint(diff_lb, diff_ub, (1, max(1, (max_h + 1) // numh - 1)))
    w = _unifint(diff_lb, diff_ub, (1, max(1, (max_w + 1) // numw - 1)))
    remcols = [c for c in cols if c != frontcol]
    numcells = _unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = [(a, b) for a in range(numh) for b in range(numw)]
    locs = set(random.sample(cands, numcells))
    gi = np.full((h * numh + numh - 1, w * numw + numw - 1), frontcol, dtype=int)
    go = np.full((numh, numw), bgc, dtype=int)
    for (a, b) in cands:
        col = random.choice(remcols) if (a, b) in locs else bgc
        gi[a * (h + 1):a * (h + 1) + h, b * (w + 1):b * (w + 1) + w] = col
        go[a, b] = col
    go = go[:, ::-1]
    return {"input": gi.tolist(), "output": go.tolist()}


def _starts(lines, n):
    st = []
    for i in range(n):
        if i not in lines and (i == 0 or (i - 1) in lines):
            st.append(i)
    return st


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Frontier colour (read from I, like the golf reading): the colour that fills
    # both a complete row and a complete column. Cell colours never equal it.
    urow = {int(I[r, 0]) for r in range(hi) if len(set(I[r].tolist())) == 1}
    ucol = {int(I[0, c]) for c in range(wi) if len(set(I[:, c].tolist())) == 1}
    f = sorted(urow & ucol)[0]
    frows = {r for r in range(hi) if (I[r] == f).all()}
    fcols = {c for c in range(wi) if (I[:, c] == f).all()}
    rs = _starts(frows, hi)
    cs = _starts(fcols, wi)
    nh, nw = len(rs), len(cs)
    # one value per block (blocks are the units between frontier lines)
    S = np.array([[I[r, c] for c in cs] for r in rs], dtype=int)

    def plan(flip):
        G = I[:, ::-1] if flip else I
        T = S[:, ::-1] if flip else S
        cur = G[:nh, :nw]
        byc = {}
        for r in range(nh):
            for c in range(nw):
                if cur[r, c] != T[r, c]:
                    byc.setdefault(int(T[r, c]), []).append((r, c))
        return byc

    # Mirror the whole board left<->right first (the block lattice is mirror-symmetric,
    # so the full-grid rectangle maps blocks onto blocks).
    flip = not np.array_equal(I, I[:, ::-1])
    byc = plan(True)
    if flip:
        alt = I[:nh, :nw].copy()
        for col, cells in byc.items():
            for r, c in cells:
                alt[r, c] = col
        if np.array_equal(alt, S[:, ::-1]):
            flip = False  # the flip would be idle
    if not flip:
        byc = plan(False)

    ops, sels = [], []
    if flip:
        ops.append(26); sels.append([0, 0, hi - 1, wi - 1])   # whole board = exact rectangle
    # collapse the canvas to one cell per block
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])
    # write each block's colour into its cell, one Color op per colour class
    for col in sorted(byc, key=lambda k: (-len(byc[k]), k)):
        ops.append(col); sels.append(sel_of(sorted(byc[col])))
    if not flip and not np.array_equal(S, S[:, ::-1]):
        ops.append(26); sels.append([0, 0, nh - 1, nw - 1])   # whole small canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task 9f236235"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9f236235"
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
                                f"for task 9f236235"
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
                    f"Failed to build a complete episode for task 9f236235 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9f236235-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
