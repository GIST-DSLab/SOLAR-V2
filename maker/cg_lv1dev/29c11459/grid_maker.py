"""
ARC Task: 29c11459 (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c != 5]
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    ncols = random.randint(2, len(rem))
    ccols = random.sample(rem, ncols)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None,
             transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c != 5])
    if ccols is None:
        rem = [c for c in range(10) if c not in (5, bgc)]
        ccols = random.sample(rem, random.randint(2, len(rem)))
    ccols = list(ccols)
    # pre-mirror dims: h rows, w (odd) cols; after transpose dims swap
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    lim_h = max(5, min(30, lim_h))
    lim_w = max(5, min(29, lim_w))
    h = random.randint(5, lim_h)
    w = random.randint(5, lim_w)
    if w % 2 == 0:
        opts = [x for x in (w - 1, w + 1) if 5 <= x <= lim_w]
        w = random.choice(opts)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    nlocs = random.randint(1, h)
    locs = random.sample(range(h), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = random.sample(range(h), nlocs)
    aforb = bforb = -1
    pairs = []
    for _ in range(nlocs):
        ac = random.choice([c for c in ccols if c != aforb]); aforb = ac
        bc = random.choice([c for c in ccols if c != bforb]); bforb = bc
        pairs.append((ac, bc))
    mid = w // 2
    for (a, b), loc in zip(pairs, sorted(locs)):
        gi[loc][0] = a
        gi[loc][w - 1] = b
        for c in range(0, mid):
            go[loc][c] = a
        for c in range(mid + 1, w):
            go[loc][c] = b
        go[loc][mid] = 5
    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour; only border endpoints are coloured
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    nz = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    cols_used = set(c for _, c in nz)
    ops, sels = [], []

    if cols_used <= {0, wi - 1}:
        # endpoints on left/right borders: each marked row is one unit
        mid = wi // 2
        for r in range(hi):
            a, b = int(I[r, 0]), int(I[r, wi - 1])
            if a == bgc or b == bgc:
                continue
            # left endpoint extends to just before the midpoint (full row segment)
            ops.append(a); sels.append(sel_of([(r, c) for c in range(0, mid)]))
            # meeting point marker
            ops.append(5); sels.append(sel_of([(r, mid)]))
            # right endpoint extends back to just after the midpoint
            ops.append(b); sels.append(sel_of([(r, c) for c in range(mid + 1, wi)]))
    else:
        # endpoints on top/bottom borders: each marked column is one unit
        mid = hi // 2
        for c in range(wi):
            a, b = int(I[0, c]), int(I[hi - 1, c])
            if a == bgc or b == bgc:
                continue
            ops.append(a); sels.append(sel_of([(r, c) for r in range(0, mid)]))
            ops.append(5); sels.append(sel_of([(mid, c)]))
            ops.append(b); sels.append(sel_of([(r, c) for r in range(mid + 1, hi)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 29c11459"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29c11459"
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
                                f"for task 29c11459"
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
                    f"Failed to build a complete episode for task 29c11459 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29c11459-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
