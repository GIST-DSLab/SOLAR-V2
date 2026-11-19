"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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

PLAN = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(PLAN):
        ex = [dict(v) for v in PLAN] + [dict(random.choice(PLAN)) for _ in range(n_ex - len(PLAN))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(PLAN, n_ex)]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def generate(diff_lb, diff_ub, max_h, max_w, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    along = max_w if transpose else max_h
    across = max_h if transpose else max_w
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, min(15, along - 1)))
    w = unifint(diff_lb, diff_ub, (1, min(30, across)))
    cols = []
    lastc = -1
    for k in range(h):
        c = choice(remove(lastc, fullcols))
        cols.append(c)
        lastc = c
    go = tuple((c,) for c in cols)
    gi = tuple(repeat(c, w) for c in cols)
    numinserts = unifint(diff_lb, diff_ub, (1, min(30, along) - h))
    for k in range(numinserts):
        loc = randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # Reading: drop every line equal to the line before it, on both axes.
    # Across the stripes every line is identical, so that axis collapses to a
    # single line (the final crop); along the stripes each duplicate stripe is
    # deleted by sliding the rest of the grid over it.
    row_mode = len(set(I[0].tolist())) == 1
    seq = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]
    runs = []
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    n = len(runs)
    # Work from the LAST stripe back to the first, so the block that slides
    # over a run's duplicates is already de-duplicated and travels as one unit.
    starts, p0 = [], 0
    for v, L in runs:
        starts.append(p0)
        p0 += L
    line = list(seq)      # live values along the stripe axis
    clen = len(seq)       # live extent along the stripe axis
    for k in range(n - 1, -1, -1):
        v, L = runs[k]
        e = L - 1
        if e == 0:
            continue
        st = starts[k]
        s = st + L
        B = clen - s
        if B <= 0:
            clen = st + 1            # trailing duplicates: the crop drops them
            continue
        block = line[s:clen]
        # Move is transparent for 0: a 0 stripe sliding onto a non-0 duplicate
        # would leave the duplicate showing, so wipe the run's duplicates first.
        if v != 0 and 0 in block[:e]:
            zpos = list(range(st + 1, st + 1 + e))
            if row_mode:
                cells = [(r, c) for r in zpos for c in range(wi)]
            else:
                cells = [(r, c) for r in range(hi) for c in zpos]
            ops.append(0); sels.append(sel_of(cells))
        if any(x != 0 for x in block):
            # grab the whole live block after the run (full rectangle,
            # background included -> bbox intended), then keep sliding it.
            if row_mode:
                ops.append(20); sels.append([s, 0, clen - 1 - s, wi - 1])
            else:
                ops.append(23); sels.append([0, s, hi - 1, clen - 1 - s])
            for _ in range(e - 1):
                ops.append(20 if row_mode else 23); sels.append(sel_of([]))
        line[st + 1:st + 1 + B] = block
        clen = st + 1 + B
    final = [0, 0, n - 1, 0] if row_mode else [0, 0, 0, n - 1]
    ops.append(33); sels.append(final)
    ops.append(34); sels.append(final)
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
