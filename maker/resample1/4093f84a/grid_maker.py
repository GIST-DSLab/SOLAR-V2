"""
ARC Task: 4093f84a (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    bgc, barc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    mh, mw = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (7, max(7, mh)))
    w = _unifint(diff_lb, diff_ub, (7, max(7, mw)))
    loci1, loci2 = sorted(random.sample(range(2, h - 2), 2))
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci1:loci2 + 1, :] = barc
    go = gi.copy()
    num1 = _unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = _unifint(diff_lb, diff_ub, (1, w // 2))
    for l1 in random.sample(range(w), num1):
        k = _unifint(diff_lb, diff_ub, (1, loci1 - 1))
        for r in random.sample(range(0, loci1), k):
            gi[r, l1] = dotc
        go[loci1 - k:loci1, l1] = barc
    for l2 in random.sample(range(w), num2):
        k = _unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        for r in random.sample(range(loci2 + 1, h), k):
            gi[r, l2] = dotc
        go[loci2 + 1:loci2 + 1 + k, l2] = barc
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    cols = sorted(set(I.flatten().tolist()))

    def bbox_area(c):
        rr, cc = np.where(I == c)
        return (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)

    def full_line_color(c):
        rows = [r for r in range(h) if (I[r] == c).all()]
        colsf = [x for x in range(w) if (I[:, x] == c).all()]
        n = int((I == c).sum())
        return bool((rows and n == len(rows) * w) or (colsf and n == len(colsf) * h))

    # bar colour: made only of full lines, with the tightest bounding box
    cands = [c for c in cols if full_line_color(c)] or cols
    barc = min(cands, key=bbox_area)
    rest = [c for c in cols if c != barc]
    bgc = max(rest, key=lambda c: int((I == c).sum()))
    dl = [c for c in rest if c != bgc]
    dotc = dl[0] if dl else None

    r0 = int(np.where(I == barc)[0][0])
    horizontal = bool((I[r0] == barc).all())
    G = I if horizontal else I.T
    H, W = G.shape
    bar_rows = [r for r in range(H) if (G[r] == barc).all()]
    r1, r2 = min(bar_rows), max(bar_rows)
    conv = (lambda r, c: (r, c)) if horizontal else (lambda r, c: (c, r))

    ops, sels = [], []
    groups = []
    for c in range(W):
        top = [r for r in range(0, r1) if G[r, c] == dotc]
        if top:
            groups.append((top, list(range(r1 - len(top), r1)), c))
        bot = [r for r in range(r2 + 1, H) if G[r, c] == dotc]
        if bot:
            groups.append((bot, list(range(r2 + 1, r2 + 1 + len(bot))), c))
    for dots, run, c in groups:
        # dots fall onto the bar: clear those not already in the stacked run,
        # then paint the run (k cells adjacent to the bar) with the bar colour
        erase = [r for r in dots if r not in run]
        if erase:
            ops.append(bgc); sels.append(sel_of([conv(r, c) for r in erase]))
        ops.append(barc); sels.append(sel_of([conv(r, c) for r in run]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
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
                                f"for task 4093f84a"
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
                    f"Failed to build a complete episode for task 4093f84a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4093f84a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
