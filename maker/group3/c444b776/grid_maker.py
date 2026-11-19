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


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d + random.uniform(-0.5, 0.5))))
    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    nh = unifint(diff_lb, diff_ub, (1, 3))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    hmax = max(2, min(9, (max_h - (nh - 1)) // nh))
    wmax = max(2, min(9, (max_w - (nw - 1)) // nw))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    remcols = [c for c in cols if c not in (bgc, linc)]
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    gi = np.full((fullh, fullw), linc, dtype=int)
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    inds = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(inds, numcels)
    small = np.full((h, w), bgc, dtype=int)
    for (i, j) in cels:
        small[i, j] = random.choice(ccols)
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)
    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = bgc
    gi[srcloc[0]:srcloc[0] + h, srcloc[1]:srcloc[1] + w] = small
    go = gi.copy()
    for (a, b) in llocs:
        go[a:a + h, b:b + w] = small
    return {"input": gi.tolist(), "output": go.tolist()}


def _frontier_colors(G):
    G = np.asarray(G, dtype=int)
    fs = set()
    for r in range(G.shape[0]):
        if len(set(G[r].tolist())) == 1:
            fs.add(int(G[r, 0]))
    for c in range(G.shape[1]):
        if len(set(G[:, c].tolist())) == 1:
            fs.add(int(G[0, c]))
    return fs


def _bands(lines, n):
    out, prev = [], 0
    for x in lines:
        if x > prev:
            out.append((prev, x - 1))
        prev = x + 1
    if prev < n:
        out.append((prev, n - 1))
    return out


def _structure(I, bgc, linc):
    hi, wi = I.shape
    dr = [r for r in range(hi) if np.all(I[r] == linc)]
    dc = [c for c in range(wi) if np.all(I[:, c] == linc)]
    rb, cb = _bands(dr, hi), _bands(dc, wi)
    if len(rb) * len(cb) < 2:
        return None
    if len({b - a for a, b in rb}) != 1 or len({b - a for a, b in cb}) != 1:
        return None
    cells = [(r0, c0, r1 - r0 + 1, c1 - c0 + 1) for (r0, r1) in rb for (c0, c1) in cb]
    nonempty = [cl for cl in cells
                if np.any(I[cl[0]:cl[0] + cl[2], cl[1]:cl[1] + cl[3]] != bgc)]
    if len(nonempty) != 1:
        return None
    return cells, nonempty[0]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    exs = [np.asarray(e[0], dtype=int) for e in (examples or [])]
    # background: the colour the empty cells are drawn in (shared across the episode)
    cnt = Counter(I.flatten().tolist())
    for E in exs:
        cnt.update(E.flatten().tolist())
    bgc = max(cnt, key=cnt.get)
    if bgc not in I:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # line colour: a full-row/col colour other than bgc; prefer one the examples agree on,
    # and require it to cut the grid into equal cells with exactly one non-empty cell
    cands = sorted(_frontier_colors(I) - {bgc},
                   key=lambda k: -sum(k in _frontier_colors(E) for E in exs))
    st = None
    for linc in cands:
        st = _structure(I, bgc, linc)
        if st is not None:
            break
    ops, sels = [], []
    if st is not None:
        cells, src = st
        sr, sc, sh, sw = src
        targets = [cl for cl in cells if cl != src]
        sub = I[sr:sr + sh, sc:sc + sw]
        colours = []
        for v in sub.flatten().tolist():
            if v != bgc and v not in colours:
                colours.append(v)
        # Unit of work = one COLOUR: for each colour in the source cell, paint every
        # pixel of that colour into the same relative position of every empty cell
        # at once (handles colour 0 naturally, no clipboard transparency issue).
        for k in colours:
            rel = [(i, j) for i in range(sh) for j in range(sw) if sub[i, j] == k]
            sel = [(tr + i, tc + j) for (tr, tc, _, _) in targets for (i, j) in rel]
            ops.append(int(k)); sels.append(sel_of(sel))
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
