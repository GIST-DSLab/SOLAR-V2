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

VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # Rule is purely structural (collapse repeated stripes); stripe colours stay random.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, transposed=None, **_ignored) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transposed is None:
        transposed = random.choice([True, False])
    len_cap = max(3, min(30, max_w if transposed else max_h))
    thick_cap = max(1, min(30, max_h if transposed else max_w))

    fullcols = list(range(10))
    h = unifint(diff_lb, diff_ub, (2, max(2, min(15, len_cap - 1))))
    h = max(2, min(h, len_cap - 1))
    w = unifint(diff_lb, diff_ub, (1, thick_cap))
    w = max(1, min(w, thick_cap))
    cols, lastc = [], -1
    for _ in range(h):
        c = random.choice([x for x in fullcols if x != lastc])
        cols.append(c)
        lastc = c
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = unifint(diff_lb, diff_ub, (1, max(1, len_cap - h)))
    numinserts = max(1, min(numinserts, len_cap - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + [list(gi[loc])] + gi[loc + 1:]
    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared by the maker and the golf entry): the grid is a sequence
    of full-width stripes; every repeated stripe is squeezed out by sliding the
    rest of the band back over it, then one cell per distinct stripe is kept.
    Cropping to a single line first would be invisible in the result (the final
    crop keeps that same line), so the whole band slides and the crop comes last."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    row_mode = len(set(I[0].tolist())) == 1          # horizontal stripes -> answer is a column
    line = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]
    L0 = len(line)
    T = wi if row_mode else hi                         # stripe thickness (cross extent)

    def stripe(j):                                     # all cells of stripe position j
        return [(j, t) for t in range(T)] if row_mode else [(t, j) for t in range(T)]

    def stripes(js):
        out = []
        for j in js:
            out += stripe(j)
        return out

    runs = []
    for v in line:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    n = len(runs)

    move_op = 20 if row_mode else 23                  # MoveU / MoveL along the stripe axis
    target = [v for v, _ in runs]

    # one slide per repeated run: the whole band past the run moves back over
    # its repeats (s = first stripe past the run, live = current band end)
    chains = []
    live, cur = L0, 0
    for (v, L) in runs:
        ex = L - 1
        if ex > 0:
            s = cur + L
            if s <= live - 1:
                chains.append([list(range(s, live)), ex])
                live -= ex
            else:
                live = cur + 1                        # trailing repeats: the crop drops them
        cur += 1

    def simulate(steps_per_chain):
        """ARCLE Move semantics on the stripe sequence: the first step grabs the
        non-black stripes of the slab over a snapshot with them cleared; black is
        never carried. Returns (sequence, every step visible?)."""
        g = list(line)
        for (slab, _ex), k in zip(chains, steps_per_chain):
            if k == 0:
                continue
            obj = {j: g[j] for j in slab if g[j] != 0}
            bg = list(g)
            for j in obj:
                bg[j] = 0
            prev = g
            for t in range(1, k + 1):
                comp = list(bg)
                for j, col in obj.items():
                    comp[j - t] = col
                if comp == prev:
                    return g, False
                prev = comp
            g = prev
        return g, True

    def finish(g):
        out = [g[j] if j < len(g) else 0 for j in range(n)]
        for j in range(n):                            # black stripes repainted at the end
            if target[j] == 0:
                out[j] = 0
        return out

    steps = []
    for ci, (_s, ex) in enumerate(chains):            # a slide that cannot show is skipped
        _g, vis = simulate(steps + [ex])
        steps.append(ex if vis else 0)
    # Drop trailing steps of a slide whose effect never reaches the kept strip.
    for ci in range(len(chains) - 1, -1, -1):
        while steps[ci] > 0:
            trial = list(steps)
            trial[ci] -= 1
            g_t, vis = simulate(trial)
            if vis and finish(g_t) == target:
                steps = trial
            else:
                break
    g, _ = simulate(steps)

    for (slab, _ex), k in zip(chains, steps):
        for t in range(k):
            ops.append(move_op)
            sels.append(sel_of(stripes(slab)) if t == 0 else sel_of([]))   # grab once, then continue

    # keep one cell per distinct stripe (full rectangle: first cross-line, n long)
    final = [0, 0, n - 1, 0] if row_mode else [0, 0, 0, n - 1]
    ops.append(33); sels.append(final)
    # Move/Crop never carry black: every black stripe of the squeezed sequence
    # that still shows a stale colour is painted back to 0 in one go.
    zcells = [(j, 0) if row_mode else (0, j)
              for j in range(n) if runs[j][0] == 0 and g[j] != 0]
    if zcells:
        ops.append(0); sels.append(sel_of(zcells))
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
