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

# Horizontal vs vertical stripes (the generator's random dmirror) is the one
# structural variant; it is planned so the examples show both.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Stripe colours are random per instance and the rule does not depend on
    # them (no background role), so only the orientation is planned.
    n = num_examples if num_examples else 3
    if n >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if transpose is None:
        transpose = random.choice((True, False))
    row_lim = max_w if transpose else max_h
    col_lim = max_h if transpose else max_w

    cols = list(range(10))
    nr = unifint(diff_lb, diff_ub, (1, max(1, min(10, row_lim // 2))))
    w = unifint(diff_lb, diff_ub, (4, max(4, col_lim)))
    scols = random.sample(cols, nr)
    sgs = [[[c] * w for _ in range(2)] for c in scols]
    numexp = unifint(diff_lb, diff_ub, (0, max(0, row_lim - 2 * nr)))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))

    gi, go = [], []
    for idx, col in enumerate(scols):
        sg = [list(r) for r in sgs[idx]]
        a, b = len(sg), w
        nnoise = unifint(diff_lb, diff_ub, (0, max(0, (a * b) // 2 - 1)))
        inds = [(x, y) for x in range(a) for y in range(b)]
        oc = [c for c in cols if c != col]
        for (x, y) in random.sample(inds, nnoise):
            sg[x][y] = random.choice(oc)
        for ix in (0, -1):
            while sum(e == col for e in sg[ix]) < w // 2:
                locs = [j for j, e in enumerate(sg[ix]) if e != col]
                sg[ix][random.choice(locs)] = col
        gi += sg
        go += [list(r) for r in sgs[idx]]

    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _segment(G):
    """Split the rows of G into contiguous bands (each >= 2 rows, neighbouring
    bands of different colour) maximising the number of cells that agree with
    their band's colour. Returns (bands, score); bands = [(start, end, colour)]."""
    G = np.asarray(G, dtype=int)
    n = G.shape[0]
    cnt = np.zeros((n + 1, 10), dtype=int)
    for i in range(n):
        cnt[i + 1] = cnt[i] + np.bincount(G[i], minlength=10)
    dp = [dict() for _ in range(n + 1)]
    dp[0][-1] = (0, None)
    for e in range(2, n + 1):
        for s in range(0, e - 1):
            if not dp[s]:
                continue
            seg = cnt[e] - cnt[s]
            for col in range(10):
                prev = [(v[0], pc) for pc, v in dp[s].items() if pc != col]
                if not prev:
                    continue
                sc, pc = max(prev)
                sc += int(seg[col])
                if col not in dp[e] or sc > dp[e][col][0]:
                    dp[e][col] = (sc, (s, pc))
    col = max(dp[n], key=lambda c: dp[n][c][0])
    score = dp[n][col][0]
    bands, e = [], n
    while e > 0:
        _, (s, pc) = dp[e][col]
        bands.append((s, e - 1, col))
        e, col = s, pc
    return bands[::-1], score


def derive_operations(I, O=None, examples=None):
    # The golf reading (each cell = majority of its row/column cross) and the
    # maker's reading agree: the grid is a stack of stripes along one axis and
    # each stripe takes its dominant colour. Painting line by line (each row's
    # own majority) is NOT safe here: the generator can noise an interior line
    # of a >=3-thick stripe almost completely, so that line's own majority
    # lies. The stripe (band) is the unit, so each band gets one Color op.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # orientation, measured from I only: the axis whose band split fits best
    if hi < 2 or wi < 2:
        vertical = hi < 2
    else:
        _, sh = _segment(I)
        _, sv = _segment(I.T)
        vertical = sv > sh

    G = I.T if vertical else I
    if G.shape[0] >= 2:
        bands, _ = _segment(G)
    else:
        bands = [(0, G.shape[0] - 1, int(np.bincount(G.flatten(), minlength=10).argmax()))]

    # paint every band with its own dominant colour, one Color op per band.
    # Each selection is the exact full rectangle of the band (rows s..e or
    # columns s..e), which is precisely the region being repainted.
    for (s, e, col) in bands:
        if np.all(G[s:e + 1, :] == col):
            continue  # band already clean: an op here would change nothing
        ops.append(int(col))
        if vertical:
            sels.append([0, s, hi - 1, e - s])   # full rectangle: columns s..e
        else:
            sels.append([s, 0, e - s, wi - 1])   # full rectangle: rows s..e

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
