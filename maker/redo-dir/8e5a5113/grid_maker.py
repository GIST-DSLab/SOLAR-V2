"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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

# The grid is a strip of d x d panels separated by 1-cell bars. Exactly one panel
# carries the pattern in I; in O every panel carries that pattern turned one more
# quarter-turn clockwise in reading order. The strip may be laid out horizontally
# (bars are columns) or vertically (bars are rows) — a discrete structural variant,
# so both orientations are planned per episode.
VARIANTS = [{"portrait": True}, {"portrait": False}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    barc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, portrait=None) -> dict:
    if portrait is None:
        portrait = choice((True, False))
    long_max = max_h if portrait else max_w     # strip direction
    short_max = max_w if portrait else max_h    # panel side direction
    colopts = interval(0, 10, 1)
    remcols = remove(barc, remove(bgc, colopts))
    d_hi = max(2, min(9, short_max, (long_max - 1) // 2))
    d = unifint(diff_lb, diff_ub, (2, d_hi))
    k = 4 if d < 7 else 3
    n_hi = max(2, min(k, (long_max + 1) // (d + 1)))
    num = unifint(diff_lb, diff_ub, (2, n_hi))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ncols = min(ncols, len(remcols))
    patcols = sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncells = unifint(diff_lb, diff_ub, (1, d * d - 1))
    indsss = sample(inds, ncells)
    for ij in indsss:
        c = fill(c, choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = choice(interval(0, num, 1))
    gi = c if fillinidx == 0 else bgcanv
    go = c
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if portrait:
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Layout: h = n*d + (n-1) along the strip, d across it (n >= 2 => strip is longer).
    portrait = h > w
    d = w if portrait else h
    n = ((h + 1) // (d + 1)) if portrait else ((w + 1) // (d + 1))

    def origin(k):
        return (k * (d + 1), 0) if portrait else (0, k * (d + 1))

    def patch(G, k):
        r, c = origin(k)
        return G[r:r + d, c:c + d]

    # Source panel = the one panel of I that is not a uniform blank.
    src, best = 0, -1
    for k in range(n):
        ncol = len(set(patch(I, k).flatten().tolist()))
        if ncol > best:
            best, src = ncol, k
    P = patch(I, src)

    # Background = the colour filling the blank panels.
    bgc = 0
    for k in range(n):
        if k != src:
            bgc = int(patch(I, k)[0, 0])
            break

    def cw(a):
        return np.rot90(a, k=3)   # clockwise quarter turn

    ops, sels = [], []
    sr, sc = origin(src)
    # CopyI the source panel — selection is exactly that d x d rectangle.
    ops.append(28); sels.append([sr, sc, d - 1, d - 1])

    for t in range(n):
        if t == src:
            continue
        T = patch(O, t)
        # How many clockwise quarter turns carry the source panel onto this one.
        m, cur = None, P
        for mm in range(4):
            if np.array_equal(cur, T):
                m = mm
                break
            cur = cw(cur)
        if m is None:
            m = (t - src) % 4
        tr, tc = origin(t)
        rect = [tr, tc, d - 1, d - 1]   # exactly the target panel rectangle

        # Paste never writes the copied cells that hold colour 0, so clear the
        # blank panel first whenever the pattern contains 0s on a non-0 background.
        if bgc != 0 and (P == 0).any():
            ops.append(0); sels.append(list(rect))
        # Replicate the pattern into this panel...
        ops.append(30); sels.append([tr, tc, 0, 0])
        # ...then turn it in place (square selection: rotation is exact).
        if m == 1:
            ops.append(25); sels.append(list(rect))
        elif m == 2:
            ops.append(25); sels.append(list(rect))
            ops.append(25); sels.append(list(rect))
        elif m == 3:
            ops.append(24); sels.append(list(rect))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
