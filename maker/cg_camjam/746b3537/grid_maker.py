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
    # Stripe colours are drawn freely and the rule only collapses repeated
    # lines, so no colour role is fixed; only the stripe orientation is planned.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, transposed=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transposed is None:
        transposed = random.choice([True, False])
    len_cap = min(30, max_w if transposed else max_h)   # extent along stripe axis
    thick_cap = min(30, max_h if transposed else max_w)
    len_cap = max(3, len_cap)
    thick_cap = max(1, thick_cap)
    h = unifint(diff_lb, diff_ub, (2, min(15, len_cap - 1)))
    w = unifint(diff_lb, diff_ub, (1, thick_cap))
    cols = []
    lastc = -1
    for _ in range(h):
        c = random.choice([x for x in range(10) if x != lastc])
        cols.append(c)
        lastc = c
    gi = [c for c in cols]
    numinserts = unifint(diff_lb, diff_ub, (1, max(1, len_cap - h)))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    I = np.array([[c] * w for c in gi], dtype=int)
    O = np.array([[c] for c in cols], dtype=int)
    if transposed:
        I = I.T
        O = O.T
    return {"input": I.tolist(), "output": O.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Reading: drop every row equal to the one above it, then drop every
    # column equal to the one left of it.  Both axes are treated alike.
    def runs_of(lines):
        runs = []
        for v in lines:
            if runs and runs[-1][0] == v:
                runs[-1][1] += 1
            else:
                runs.append([v, 1])
        return runs

    row_runs = runs_of([tuple(r) for r in I.tolist()])
    after_rows = np.array([r[0] for r in row_runs], dtype=int)
    col_runs = runs_of([tuple(c) for c in after_rows.T.tolist()])
    n_r, n_c = len(row_runs), len(col_runs)

    def plan_axis(runs, axis, H, W):
        """Moves sliding everything past each duplicate back by one line."""
        out = []
        clen = H if axis == 0 else W
        cur = 0
        for (_v, L) in runs:
            s = cur + L
            for _ in range(L - 1):
                if s > clen - 1:
                    break
                if axis == 0:   # whole rectangle of rows s..clen-1 slides up
                    out.append((20, [s, 0, clen - 1 - s, W - 1]))
                else:           # whole rectangle of cols s..clen-1 slides left
                    out.append((23, [0, s, H - 1, clen - 1 - s]))
                s -= 1
                clen -= 1
            cur += 1
        return out

    # rows first (live rectangle shrinks to n_r rows once row duplicates are
    # squeezed out), then columns over those n_r rows; one crop drops the tail.
    body = []
    body += plan_axis(row_runs, 0, hi, wi)
    body += plan_axis(col_runs, 1, n_r, wi)
    if (n_r, n_c) != (hi, wi):
        body.append((33, [0, 0, n_r - 1, n_c - 1]))     # one line per run

    target = after_rows[:, [0] + [sum(r[1] for r in col_runs[:k]) for k in range(1, n_c)]] \
        if n_c else after_rows

    # --- tiny simulator of ARCLE Move / Crop (0 = empty) ----------------------
    def sim(grid0, steps):
        g = grid0.copy()
        states = [g.copy()]
        for op, sel in steps:
            r, c, h, w = sel
            if op in (20, 23):
                H, W = g.shape
                mask = np.zeros_like(g, dtype=bool)
                mask[r:r + h + 1, c:c + w + 1] = True
                obj = mask & (g != 0)
                bg = g.copy()
                bg[obj] = 0
                dr, dc = (-1, 0) if op == 20 else (0, -1)
                ng = bg
                for (y, x) in zip(*np.nonzero(obj)):
                    ny, nx = y + dr, x + dc
                    if 0 <= ny < H and 0 <= nx < W:
                        ng[ny, nx] = g[y, x]
                g = ng
            elif op == 33:
                ng = np.zeros((h + 1, w + 1), dtype=int)
                sub = g[r:r + h + 1, c:c + w + 1]
                ng[:sub.shape[0], :sub.shape[1]] = sub
                g = ng
            states.append(g.copy())
        return g, states

    def effective(grid0, steps):
        # keep only the steps that visibly change the grid
        g, states = sim(grid0, steps)
        keep = [st for st, a, b in zip(steps, states, states[1:])
                if not (a.shape == b.shape and np.array_equal(a, b))]
        return keep, g

    # Move / Crop treat 0 as empty, so a 0-coloured stripe is not carried by
    # the slide: the line it should occupy may still show its old neighbour.
    # The 0 stripes' places are known from I's run order -> paint them black.
    zero_lines = []
    if n_r > 1 or n_c == 1:
        for k in range(n_r):
            if all(v == 0 for v in row_runs[k][0]):
                zero_lines.append([(k, c) for c in range(n_c)])
    if n_c > 1:
        for k in range(n_c):
            if all(v == 0 for v in col_runs[k][0]):
                zero_lines.append([(r, k) for r in range(n_r)])
    zero_cells = sorted({p for line in zero_lines for p in line})

    def finish(steps):
        g, _ = sim(I, steps)
        if g.shape != target.shape:
            return None, None
        fix = [p for p in zero_cells if g[p] != 0]
        g2 = g.copy()
        for p in fix:
            g2[p] = 0
        return (fix if np.array_equal(g2, target) else None), g

    plain, _ = effective(I, body)
    # A slide whose only visible work lands on a 0-stripe's place (which the
    # black repaint covers anyway) or past the crop is not part of the rule.
    changed = True
    while changed:
        changed = False
        for k in range(len(plain)):
            if plain[k][0] == 33:
                continue
            trial, _ = effective(I, plain[:k] + plain[k + 1:])
            fx, _ = finish(trial)
            if fx is not None:
                plain = trial
                changed = True
                break
    fix, _ = finish(plain)

    ops, sels = [], []
    for op, sel in plain:
        ops.append(op); sels.append(sel)
    if fix:
        ops.append(0); sels.append(sel_of(fix))

    ops.append(34); sels.append([0, 0, n_r - 1, n_c - 1])
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
