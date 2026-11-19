"""
ARC Task: c444b776 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if bgc is None or linc is None:
        bgc, linc = random.sample(range(10), 2)
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    while True:
        h = unifint(diff_lb, diff_ub, (2, 9))
        w = unifint(diff_lb, diff_ub, (2, 9))
        nh = unifint(diff_lb, diff_ub, (1, 3))
        nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    gi = np.full((fullh, fullw), linc, dtype=int)
    small = np.full((h, w), bgc, dtype=int)
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, numcels):
        small[i, j] = random.choice(ccols)
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    src = random.choice(llocs)
    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = bgc
    gi[src[0]:src[0] + h, src[1]:src[1] + w] = small
    go = gi.copy()
    for (a, b) in llocs:
        go[a:a + h, b:b + w] = small
    return {"input": gi.tolist(), "output": go.tolist()}


def _line_colors(G):
    G = np.asarray(G, dtype=int)
    cands = set()
    for r in range(G.shape[0]):
        if len(set(G[r].tolist())) == 1:
            cands.add(int(G[r, 0]))
    for c in range(G.shape[1]):
        if len(set(G[:, c].tolist())) == 1:
            cands.add(int(G[0, c]))
    return cands


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Episode conventions (bgc, divider colour) read from the demonstrations' inputs.
    cnt = Counter(I.flatten().tolist())
    ex_inputs = [np.asarray(e[0], dtype=int) for e in (examples or [])]
    for X in ex_inputs:
        cnt.update(X.flatten().tolist())
    bgc = max(cnt, key=cnt.get)
    cands = _line_colors(I) - {bgc}
    for X in ex_inputs:
        inter = cands & _line_colors(X)
        if inter:
            cands = inter
    # Divider colour: the candidate forming the most full lines in I
    linc = None
    best = -1
    for c in sorted(cands):
        n = sum(all(I[r, :] == c) for r in range(hi)) + sum(all(I[:, k] == c) for k in range(wi))
        if n > best:
            best, linc = n, c

    div_rows = [r for r in range(hi) if linc is not None and all(I[r, :] == linc)]
    div_cols = [c for c in range(wi) if linc is not None and all(I[:, c] == linc)]

    def bands(divs, n):
        out, prev = [], 0
        for d in divs:
            if d > prev:
                out.append((prev, d - 1))
            prev = d + 1
        if prev < n:
            out.append((prev, n - 1))
        return out

    row_bands = bands(div_rows, hi)
    col_bands = bands(div_cols, wi)

    src = None
    for (r0, r1) in row_bands:
        for (c0, c1) in col_bands:
            if np.any(I[r0:r1 + 1, c0:c1 + 1] != bgc):
                src = (r0, r1, c0, c1)
                break
        if src is not None:
            break
    if src is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    sr0, sr1, sc0, sc1 = src
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    # Cells of the source panel holding colour 0 (Paste cannot write them).
    zero_src = [(dr, dc) for dr in range(sh) for dc in range(sw)
                if I[sr0 + dr, sc0 + dc] == 0] if bgc != 0 else []

    # Does the copy carry anything Paste can write? (fg cells that are not 0)
    paste_fg = bool(np.any((I[sr0:sr1 + 1, sc0:sc1 + 1] != bgc) & (I[sr0:sr1 + 1, sc0:sc1 + 1] != 0)))

    # Phase 1 (one axis): spread the source panel along its own panel-row.
    others_in_row = [(c0, c1) for (c0, c1) in col_bands if (c0, c1) != (sc0, sc1)]
    if others_in_row:
        if paste_fg:
            ops.append(28); sels.append([sr0, sc0, sh - 1, sw - 1])  # full source panel rectangle
        for (c0, c1) in others_in_row:
            if paste_fg:
                ops.append(30); sels.append(sel_of([(sr0, c0)]))
            if zero_src:
                ops.append(0); sels.append(sel_of([(sr0 + dr, c0 + dc) for dr, dc in zero_src]))

    # Phase 2 (other axis): the filled panel-row strip (full width, dividers included)
    # is copied onto every other panel-row.
    others_rows = [(r0, r1) for (r0, r1) in row_bands if (r0, r1) != (sr0, sr1)]
    if others_rows:
        # The strip as it stands after phase 1 (built from I and the copy just made).
        strip = I.copy()
        for (c0, c1) in others_in_row:
            strip[sr0:sr1 + 1, c0:c1 + 1] = I[sr0:sr1 + 1, sc0:sc1 + 1]
        zero_strip = [(dr, c) for dr in range(sh) for c in range(wi)
                      if strip[sr0 + dr, c] == 0] if bgc != 0 else []
        if paste_fg:
            ops.append(29); sels.append([sr0, 0, sh - 1, wi - 1])  # whole strip rectangle
        for (r0, r1) in others_rows:
            if paste_fg:
                ops.append(30); sels.append(sel_of([(r0, 0)]))
            if zero_strip:
                cells = [(r0 + dr, c) for dr, c in zero_strip if I[r0 + dr, c] != 0]
                if cells:
                    ops.append(0); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task c444b776"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c444b776"
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
                                f"for task c444b776"
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
                    f"Failed to build a complete episode for task c444b776 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c444b776-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
