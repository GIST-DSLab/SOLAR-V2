"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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

# Reading shared with the golf entry and the maker: every line running across
# the stripes is cleaned to its dominant colour. Colours play no role in the
# rule, so no colour is fixed. The only structural variant is the stripe
# orientation (the generator's random dmirror), and the instance plan makes
# sure both orientations appear in the examples.
ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    L = min(30, max_w if transpose else max_h)    # extent across the stripes
    Wm = min(30, max_h if transpose else max_w)   # extent along the stripes
    cols = list(range(10))
    nr = _unifint(diff_lb, diff_ub, (1, max(1, min(10, L // 2))))
    w = _unifint(diff_lb, diff_ub, (4, max(4, Wm)))
    scols = random.sample(cols, nr)
    heights = [2] * nr
    numexp = _unifint(diff_lb, diff_ub, (0, max(0, L - 2 * nr)))
    for _ in range(numexp):
        heights[random.randint(0, nr - 1)] += 1
    gi_rows, go_rows = [], []
    for col, a in zip(scols, heights):
        sg = [[col] * w for _ in range(a)]
        ub = (a * w) // 2 - 1
        nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = [(i, j) for i in range(a) for j in range(w)]
        oc = [c for c in cols if c != col]
        sg2 = [row[:] for row in sg]
        for (i, j) in random.sample(inds, nnoise):
            sg2[i][j] = random.choice(oc)
        for idx in (0, a - 1):
            while sum(e == col for e in sg2[idx]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idx]) if e != col]
                sg2[idx][random.choice(locs)] = col
        gi_rows += sg2
        go_rows += sg
    gi = np.array(gi_rows)
    go = np.array(go_rows)
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _segment(G):
    # Split G's rows into contiguous bands (each at least 2 rows, neighbouring
    # bands in different colours), choosing the split that maximises the number
    # of cells matching their band's colour. Returns (bands, score).
    n = G.shape[0]
    pre = np.zeros((n + 1, 10), dtype=int)
    for i in range(n):
        pre[i + 1] = pre[i] + np.bincount(G[i], minlength=10)
    NEG = -10 ** 9
    best = [[NEG] * 10 for _ in range(n + 1)]
    back = [[None] * 10 for _ in range(n + 1)]
    for j in range(2, n + 1):
        for c in range(10):
            for i in range(0, j - 1):
                gain = int(pre[j][c] - pre[i][c])
                if i == 0:
                    cand, prev = gain, None
                else:
                    prev, cand = None, NEG
                    for c2 in range(10):
                        if c2 != c and best[i][c2] > NEG and best[i][c2] + gain > cand:
                            cand = best[i][c2] + gain
                            prev = c2
                    if prev is None:
                        continue
                if cand > best[j][c]:
                    best[j][c] = cand
                    back[j][c] = (i, prev)
    c = int(np.argmax(best[n]))
    score = best[n][c]
    bands = []
    j = n
    while j > 0:
        i, prev = back[j][c]
        bands.append((i, j - 1, c))
        j, c = i, prev
    return bands[::-1], score


def derive_operations(I, O=None, examples=None):
    # The golf entry and the maker read the task the same way: noise is wiped
    # and each line across the stripes takes its dominant colour. They split the
    # work differently. The golf entry decides each LINE (row/column) on its
    # own, while the maker paints a whole band at once. Here each line is its
    # own unit: one Color op per line that still holds noise. The line's colour
    # is its band's colour, measured from I alone. Using the band rather than
    # the line by itself keeps this correct when noise ties with or outnumbers
    # the stripe colour inside a single middle line.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    if hi < 2 or wi < 2:
        vertical = hi < 2
    else:
        _, sh = _segment(I)
        _, sv = _segment(I.T)
        vertical = sv > sh
    G = I.T if vertical else I
    bands, _ = _segment(G)

    ops, sels = [], []
    for (s, e, col) in bands:
        for k in range(s, e + 1):
            if np.all(G[k] == col):
                continue                                   # line already clean
            ops.append(int(col))
            if vertical:
                sels.append(sel_of([(r, k) for r in range(hi)]))   # whole column k
            else:
                sels.append(sel_of([(k, c) for c in range(wi)]))   # whole row k

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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
