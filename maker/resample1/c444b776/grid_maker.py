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
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    return int(round(random.uniform(lo, hi)))


def sample_colors(num_examples=None) -> dict:
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=1) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc, linc)]
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (2, 9))
        w = _unifint(diff_lb, diff_ub, (2, 9))
        nh = _unifint(diff_lb, diff_ub, (1, 3))
        nw = _unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h, w, nh, nw = 2, 2, 1, 2
        fullh, fullw = 2, 5
    gi = np.full((fullh, fullw), linc, dtype=int)
    numcol = _unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcol)
    numcels = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(inds, numcels)
    small = np.full((h, w), bgc, dtype=int)
    for (i, j) in cels:
        small[i, j] = random.choice(ccols)
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)
    go = gi.copy()
    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = small if (a, b) == srcloc else bgc
        go[a:a + h, b:b + w] = small
    return {"input": gi.tolist(), "output": go.tolist()}


def _segments(n, lines):
    segs, start = [], 0
    for k in sorted(lines) + [n]:
        if k > start:
            segs.append((start, k - start))
        start = k + 1
    return segs


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    # background: the majority colour (empty cells dominate by construction)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # line colour: colour of a uniform full row/col that is not bgc
    linc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1 and I[r, 0] != bgc:
            linc = int(I[r, 0])
    for c in range(wi):
        if len(set(I[:, c].tolist())) == 1 and I[0, c] != bgc:
            linc = int(I[0, c])
    line_rows = [r for r in range(hi) if linc is not None and all(I[r] == linc)]
    line_cols = [c for c in range(wi) if linc is not None and all(I[:, c] == linc)]
    rsegs = _segments(hi, line_rows)
    csegs = _segments(wi, line_cols)
    cells = [(r0, rh, c0, cw) for (r0, rh) in rsegs for (c0, cw) in csegs]
    src = None
    for (r0, rh, c0, cw) in cells:
        if np.any(I[r0:r0 + rh, c0:c0 + cw] != bgc):
            src = (r0, rh, c0, cw)
            break
    ops, sels = [], []
    if src is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    r0, rh, c0, cw = src
    patch = I[r0:r0 + rh, c0:c0 + cw]
    G = I.copy()
    dests = [cl for cl in cells if cl != src]
    # Copy the source cell from the input
    if np.any(patch != 0):
        ops.append(28); sels.append([r0, c0, rh - 1, cw - 1])  # full source cell rectangle
        for (dr, _, dc, _) in dests:
            new = G.copy()
            sub = new[dr:dr + rh, dc:dc + cw]
            m = patch != 0
            sub[m] = patch[m]
            if not np.array_equal(new, G):
                ops.append(30); sels.append(sel_of([(dr, dc)]))
                G = new
    # zero-coloured content is not carried by Paste; paint it explicitly
    if bgc != 0:
        zeros = [(i, j) for i in range(rh) for j in range(cw) if patch[i, j] == 0]
        if zeros:
            for (dr, _, dc, _) in dests:
                cl = [(dr + i, dc + j) for (i, j) in zeros if G[dr + i, dc + j] != 0]
                if cl:
                    ops.append(0); sels.append(sel_of(cl))
                    for (r, c) in cl:
                        G[r, c] = 0
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
