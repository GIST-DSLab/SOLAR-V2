"""
ARC Task: 40853293 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, defaultdict
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    # The rule depends only on where same-coloured pairs of dots sit, not on which
    # colours they are, so only the background colour is fixed for the episode.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    nlines = _unifint(diff_lb, diff_ub, (2, min(8, (h * w) // 2)))
    nhori = random.randint(1, nlines - 1)
    nverti = nlines - nhori
    ilocs = random.sample(range(h), min(nhori, h))
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for ii in ilocs:
        llen = _unifint(diff_lb, diff_ub, (2, w - 1))
        js = random.randint(0, w - llen)
        je = js + llen - 1
        col = random.choice(remcols)
        remcols.remove(col)
        gi[ii][js] = col
        gi[ii][je] = col
        for j in range(js, je + 1):
            go[ii][j] = col
    jlocs = [j for j in range(w) if sum(1 for i in range(h) if gi[i][j] == bgc) > 1]
    nverti = min(nverti, len(jlocs))
    jlocs = random.sample(jlocs, nverti)
    for jj in jlocs:
        jc = [i for i in range(h) if gi[i][jj] == bgc]
        kk = len(jc)
        llen = _unifint(diff_lb, diff_ub, (2, kk))
        sp = random.randint(0, kk - llen)
        ep = sp + llen - 1
        a, b = jc[sp], jc[ep]
        col = random.choice(remcols)
        remcols.remove(col)
        gi[a][jj] = col
        gi[b][jj] = col
        for i in range(a, b + 1):
            go[i][jj] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Each line colour is used by exactly two dots, so the background is the
    # majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = defaultdict(list)
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells[v].append((r, c))

    horiz, vert = [], []
    for col, pts in cells.items():
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(pts)
        if r0 == r1:
            horiz.append((r0, min(c0, c1), max(c0, c1), col))
        elif c0 == c1:
            vert.append((r0, r1, c0, col))

    ops, sels = [], []
    G = I.copy()

    # This follows the reference's order. Vertical lines go down first:
    # each pair of dots in a column is joined by painting its whole segment,
    # which runs over any crossing cells.
    for r0, r1, c, col in sorted(vert, key=lambda t: t[2]):
        seg = [(r, c) for r in range(r0, r1 + 1)]
        if any(G[r, cc] != col for r, cc in seg):
            ops.append(int(col)); sels.append(sel_of(seg))
            for r, cc in seg:
                G[r, cc] = col

    # Horizontal lines then fill only the background gaps along their row segment.
    # Cells already taken by a vertical line stay as they are, so the vertical wins.
    for r, c0, c1, col in sorted(horiz, key=lambda t: t[0]):
        gaps = [(r, c) for c in range(c0, c1 + 1) if G[r, c] == bgc]
        if gaps:
            ops.append(int(col)); sels.append(sel_of(gaps))
            for rr, c in gaps:
                G[rr, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 40853293"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 40853293"
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
                                f"for task 40853293"
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
                    f"Failed to build a complete episode for task 40853293 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"40853293-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
