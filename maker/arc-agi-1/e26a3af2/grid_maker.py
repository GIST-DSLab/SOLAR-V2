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


def sample_colors(num_examples=None) -> dict:
    # Every band colour is drawn fresh for each instance, and the rule does not
    # depend on colour. The only discrete structural case is the orientation
    # (horizontal or vertical bands), so the plan covers both.
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    cols = list(range(10))
    long_max = max_w if transpose else max_h   # axis the bands are stacked along
    wid_max = max_h if transpose else max_w
    nr = _unifint(diff_lb, diff_ub, (1, max(1, min(10, long_max // 2))))
    w = _unifint(diff_lb, diff_ub, (4, max(4, wid_max)))
    scols = random.sample(cols, nr)
    heights = [2] * nr
    numexp = _unifint(diff_lb, diff_ub, (0, max(0, long_max - 2 * nr)))
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
        for idxx in [0, -1]:
            while sum(e == col for e in sg2[idxx]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                sg2[idxx][random.choice(locs)] = col
        gi_rows += sg2
        go_rows += sg
    gi = np.array(gi_rows, dtype=int)
    go = np.array(go_rows, dtype=int)
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _segment(G):
    # Split the rows into contiguous bands, each at least 2 rows and one colour.
    # Pick the split that leaves the most cells already holding their band's
    # colour. Noise counts are taken over the whole band, not row by row, so a
    # middle row that is mostly noise cannot break a band apart.
    h, w = G.shape
    cnt = np.zeros((h + 1, 10), dtype=int)
    for r in range(h):
        cnt[r + 1] = cnt[r] + np.bincount(G[r], minlength=10)
    NEG = -10 ** 9
    best = [NEG] * (h + 1)
    back = [None] * (h + 1)
    best[0] = 0
    for e in range(2, h + 1):
        for s in range(0, e - 1):
            if best[s] == NEG:
                continue
            seg = cnt[e] - cnt[s]
            col = int(np.argmax(seg))
            sc = best[s] + int(seg[col])
            if sc > best[e]:
                best[e] = sc
                back[e] = (s, col)
    bands = []
    e = h
    while e > 0:
        s, col = back[e]
        bands.append((s, e - 1, col))
        e = s
    bands.reverse()
    merged = []
    for b in bands:
        if merged and merged[-1][2] == b[2]:
            merged[-1] = (merged[-1][0], b[1], b[2])
        else:
            merged.append(b)
    return merged, best[h]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- orientation, measured from I only: the axis whose band split fits best
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

    # --- paint every band with its own colour, one Color op per band ---
    # Color<n> writes n straight onto the grid, including n == 0, so a band
    # whose colour is 0 needs no special handling. No Paste/Move/Copy is used.
    for (s, e, col) in bands:
        if np.all(G[s:e + 1, :] == col):
            continue                      # band already uniform: nothing to do
        ops.append(int(col))
        if vertical:
            sels.append([0, s, hi - 1, e - s])   # exact full rectangle: columns s..e
        else:
            sels.append([s, 0, e - s, wi - 1])   # exact full rectangle: rows s..e

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
