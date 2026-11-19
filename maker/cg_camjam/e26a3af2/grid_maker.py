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
from collections import Counter
from maker.sel_helpers import sel_of

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Every band colour is sampled independently and the rule never depends on a
    # colour role, so no colour is fixed. Only orientation is planned.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return random.randint(min(lo, hi), max(lo, hi))


def _line_modes(G):
    return [Counter(row.tolist()).most_common(1)[0][0] for row in G]


def _is_vertical(I):
    # stripes run along the axis whose lines' majority colours are more varied
    rm = _line_modes(I)
    cm = _line_modes(I.T)
    return len(set(cm)) > len(set(rm))


def _solve(I):
    vertical = _is_vertical(I)
    G = I.T if vertical else I
    out = np.array([[m] * G.shape[1] for m in _line_modes(G)])
    return out.T if vertical else out


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    nlines_max = min(30, max_w if transpose else max_h)
    wcap = min(30, max_h if transpose else max_w)
    cols = list(range(10))
    for _ in range(200):
        nr = _unifint(diff_lb, diff_ub, (1, min(10, nlines_max // 2)))
        w = _unifint(diff_lb, diff_ub, (4, wcap))
        scols = random.sample(cols, nr)
        heights = [2] * nr
        numexp = _unifint(diff_lb, diff_ub, (0, nlines_max - 2 * nr))
        for _k in range(numexp):
            heights[random.randint(0, nr - 1)] += 1
        gi_rows, go_rows = [], []
        for col, a in zip(scols, heights):
            sg = [[col] * w for _ in range(a)]
            go_rows += [list(r) for r in sg]
            ub = (a * w) // 2 - 1
            nnoise = _unifint(diff_lb, diff_ub, (0, ub))
            inds = [(i, j) for i in range(a) for j in range(w)]
            oc = [c for c in cols if c != col]
            for (i, j) in random.sample(inds, nnoise):
                sg[i][j] = random.choice(oc)
            for idx in (0, a - 1):
                while sum(e == col for e in sg[idx]) < w // 2:
                    locs = [j for j, e in enumerate(sg[idx]) if e != col]
                    sg[idx][random.choice(locs)] = col
            gi_rows += sg
        gi = np.array(gi_rows)
        go = np.array(go_rows)
        if transpose:
            gi, go = gi.T, go.T
        # keep only instances where every line's majority colour is its band colour
        if np.array_equal(_solve(gi), go):
            return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("could not generate consistent instance")


def derive_operations(I, O=None, examples=None):
    # Reading: every single line (a row, or a column when the stripes are
    # vertical) is restored to its own most frequent colour. The unit is one
    # line, not a whole band: each noisy line gets one Color op over the
    # entire line; lines that are already uniform are left alone.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vertical = _is_vertical(I)
    ops, sels = [], []
    G = I.T if vertical else I
    for k, row in enumerate(G):
        m = Counter(row.tolist()).most_common(1)[0][0]
        if np.all(row == m):
            continue  # line already clean: painting it would be a no-op
        if vertical:
            cells = [(r, k) for r in range(hi)]   # the whole column k
        else:
            cells = [(k, c) for c in range(wi)]   # the whole row k
        ops.append(int(m)); sels.append(sel_of(cells))
    ops.append(34); sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
