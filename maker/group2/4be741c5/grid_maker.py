"""
ARC Task: 4be741c5 (RE-ARC) — LLM-generated grid_maker
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
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, d))))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(max(a, lo), min(b, max(hi, a)))


def sample_colors(num_examples=None) -> dict:
    # colours are arbitrary per instance (the rule only reads band order);
    # the structural variant is the band orientation (transpose or not).
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    # pre-mirror grid is h x w; after transpose it becomes w x h
    lim_h = max_w if transpose else max_h
    lim_w = max_h if transpose else max_w
    lim_h = max(4, min(30, lim_h))
    lim_w = max(6, min(30, lim_w))
    h = _unifint(diff_lb, diff_ub, (4, lim_h))
    w = _unifint(diff_lb, diff_ub, (6, lim_w))
    numcolors = _unifint(diff_lb, diff_ub, (2, max(2, w // 3)))
    ccols = random.sample(range(10), numcolors)
    # columns sequence
    seq = []
    for c in ccols:
        seq += [c, c, c]
    while len(seq) < w:
        idx = random.randint(0, len(seq) - 1)
        seq = seq[:idx] + seq[idx:idx + 1] + seq[idx:]
    gi = [list(seq) for _ in range(h)]
    ndist = _unifint(diff_lb, diff_ub, (0, 3 * h * numcolors))
    for _ in range(ndist):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b + 1] and gi[a][b + 2] == gi[a][b + 3]:
                    options.append((a, b, gi[a][b], gi[a][b + 2]))
        if not options:
            break
        a, b, c1, c2 = random.choice(options)
        if random.choice((True, False)):
            gi[a][b + 1] = c2
        else:
            gi[a][b + 2] = c1
    gi = np.array(gi, dtype=int)
    go = np.array([ccols], dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Bands run vertically (read row 0) unless row 0 is a single colour,
    # in which case they run horizontally (read column 0).
    row0 = I[0, :].tolist()
    vertical_out = len(set(row0)) == 1
    line = I[:, 0].tolist() if vertical_out else row0
    n = len(line)

    # runs of the reading line: each band is one contiguous run, in order
    runs = []  # (colour, start, end)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and line[j + 1] == line[i]:
            j += 1
        runs.append((line[i], i, j))
        i = j + 1
    k = len(runs)

    def cell(t):
        return (t, 0) if vertical_out else (0, t)

    # Unit of work: a whole band segment on the reading line.
    # Band j is stretched back along the line so it starts at index j
    # (painting the band's colour over the cells from j to the band's own end),
    # pushing the strip of bands together into the first k cells.
    for j, (col, s, e) in enumerate(runs):
        if s == j:
            continue  # band already starts at its rank; nothing visible to do
        cells = [cell(t) for t in range(j, e + 1)]
        ops.append(int(col))
        sels.append(sel_of(cells))

    # keep only the first k cells of the reading line
    ops.append(33)
    sels.append([0, 0, k - 1, 0] if vertical_out else [0, 0, 0, k - 1])

    ho, wo = (k, 1) if vertical_out else (1, k)
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
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
                                f"for task 4be741c5"
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
                    f"Failed to build a complete episode for task 4be741c5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4be741c5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
