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


def sample_colors(num_examples=None) -> dict:
    # Stripe colours are all random and the rule ignores colour, so no colour is fixed.
    # Orientation (rows vs columns) is a discrete case: plan it so the examples show both.
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, mirrored=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    if mirrored is None:
        mirrored = random.choice((True, False))
    # stripe axis length bound / cross-axis width bound
    lmax = max_w if mirrored else max_h
    wmax = max_h if mirrored else max_w
    lmax = min(lmax, 30)
    wmax = min(wmax, 30)
    if lmax < 3:
        lmax = 3
    fullcols = list(range(10))
    h = unifint(diff_lb, diff_ub, (2, min(15, lmax - 1)))
    w = unifint(diff_lb, diff_ub, (1, wmax))
    cols = []
    lastc = -1
    for _ in range(h):
        c = random.choice([x for x in fullcols if x != lastc])
        cols.append(c)
        lastc = c
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = unifint(diff_lb, diff_ub, (1, lmax - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    gi = [list(r) for r in gi]
    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Stripes run along whole rows (first row uniform) or whole columns.
    row_mode = len(set(I[0].tolist())) == 1
    seq = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]

    runs = []
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    n = len(runs)

    def cell(i):
        return (i, 0) if row_mode else (0, i)

    # A 0-stripe only needs protecting if it will actually be slid by a Move
    # (Move treats 0 as empty): i.e. some earlier run has duplicates to collapse.
    present = set(int(v) for v in np.unique(I))
    need_fix = False
    seen_dup = False
    for v, L in runs:
        if v == 0 and seen_dup:
            need_fix = True
            break
        if L > 1:
            seen_dup = True
    spare = None
    line = list(seq)                    # live values along the stripe axis
    if need_fix:
        spare = next((c for c in range(1, 10) if c not in present), None)
        if spare is not None:
            zcells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]
            ops.append(spare)
            sels.append(sel_of(zcells))
            line = [spare if v == 0 else v for v in line]

    # Collapse each duplicated run by sliding the whole remaining block (full
    # rectangle of live stripes, all cells intended) toward it one step at a time.
    # Without a spare colour, black lines in the block are not carried: where
    # they should land the duplicate's colour still shows; that is repaired below.
    clen = hi if row_mode else wi
    cur = 0
    for (v, L) in runs:
        excess = L - 1
        if excess > 0:
            s = cur + L
            for _ in range(excess):
                if s > clen - 1:
                    break
                if any(line[j] != 0 for j in range(s, clen)):
                    new = list(line)
                    for j in range(s, clen):
                        new[j] = 0
                    for j in range(s, clen):
                        if line[j] != 0:
                            new[j - 1] = line[j]
                    if new != line:
                        if row_mode:
                            ops.append(20)
                            sels.append([s, 0, clen - 1 - s, wi - 1])
                        else:
                            ops.append(23)
                            sels.append([0, s, hi - 1, clen - 1 - s])
                        line = new
                # (a block of pure black has nothing to carry: no Move, the
                #  grid stays as it is and the black is repaired below)
                s -= 1
                clen -= 1
            if s > clen - 1:
                clen = cur + 1
        cur += 1

    # Keep one cell per distinct stripe.
    if row_mode:
        final = [0, 0, n - 1, 0]
    else:
        final = [0, 0, 0, n - 1]
    ops.append(33)
    sels.append(final)

    # Put black back on the kept stripes that are black in I but do not show it.
    vals = [v for v, _ in runs]
    zfix = [cell(i) for i in range(n) if vals[i] == 0 and line[i] != 0]
    if zfix:
        ops.append(0)
        sels.append(sel_of(zfix))

    ops.append(34)
    sels.append(final)
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
