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
from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are exact full band rectangles)


ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Band colours are sampled freely per instance; the rule (paint each band
    # with its dominant colour) does not depend on any fixed colour role.
    # Only the orientation is a discrete structural variant -> plan it.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + round(d * (b - a))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    # rows of the band-stack go along H (normal) or W (transposed)
    row_lim = max_w if transpose else max_h
    col_lim = max_h if transpose else max_w
    row_lim = max(2, min(30, row_lim))
    col_lim = max(4, min(30, col_lim))
    cols = list(range(10))
    nr = _unifint(diff_lb, diff_ub, (1, min(10, row_lim // 2)))
    w = _unifint(diff_lb, diff_ub, (4, col_lim))
    scols = random.sample(cols, nr)
    heights = [2] * nr
    numexp = _unifint(diff_lb, diff_ub, (0, row_lim - 2 * nr))
    for _ in range(numexp):
        heights[random.randint(0, nr - 1)] += 1
    gi_rows, go_rows = [], []
    for col, a in zip(scols, heights):
        sg = [[col] * w for _ in range(a)]
        sgo = [row[:] for row in sg]
        ub = (a * w) // 2 - 1
        nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = [(i, j) for i in range(a) for j in range(w)]
        oc = [c for c in cols if c != col]
        for (i, j) in random.sample(inds, nnoise):
            sg[i][j] = random.choice(oc)
        for idxx in [0, a - 1]:
            while sum(e == col for e in sg[idxx]) < w // 2:
                locs = [j for j, e in enumerate(sg[idxx]) if e != col]
                sg[idxx][random.choice(locs)] = col
        gi_rows += sg
        go_rows += sgo
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _segment(G):
    """Optimal split of G's rows into bands (each >= 2 rows) maximising the
    number of cells equal to their band's dominant colour."""
    n, w = G.shape
    pref = np.zeros((10, n + 1), dtype=int)
    for c in range(10):
        pref[c, 1:] = np.cumsum((G == c).sum(axis=1))
    NEG = -10 ** 9
    dp = [NEG] * (n + 1)
    back = [None] * (n + 1)
    dp[0] = 0
    for i in range(2, n + 1):
        for j in range(0, i - 1):
            if dp[j] == NEG:
                continue
            cnts = pref[:, i] - pref[:, j]
            c = int(cnts.argmax())
            v = dp[j] + int(cnts[c])
            if v > dp[i]:
                dp[i] = v
                back[i] = (j, c)
    if dp[n] == NEG:
        c = int(np.bincount(G.flatten(), minlength=10).argmax())
        return [(0, n - 1, c)], int((G == c).sum())
    segs = []
    i = n
    while i > 0:
        j, c = back[i]
        segs.append((j, i - 1, c))
        i = j
    segs.reverse()
    merged = []
    for s, e, c in segs:
        if merged and merged[-1][2] == c:
            merged[-1] = (merged[-1][0], e, c)
        else:
            merged.append((s, e, c))
    return merged, dp[n]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # orientation measured from I alone: which axis' band split explains more cells
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

    # one Color op per band: paint the whole band (an exact full rectangle)
    # with its dominant colour, wiping out the noise in one stroke.
    for (s, e, col) in bands:
        if np.all(G[s:e + 1, :] == col):
            continue  # already clean; painting would be a no-op
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
