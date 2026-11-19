"""
ARC Task: b7249182 (RE-ARC) — LLM-generated grid_maker
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
    bgc, ca, cb = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None,
             transpose=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None or ca is None or cb is None:
        bgc, ca, cb = random.sample(range(10), 3)
    if transpose is None:
        transpose = random.choice([True, False])

    # if transposed, the grid's h/w swap at the end
    H_lim = max_w if transpose else max_h
    W_lim = max_h if transpose else max_w
    H_lim = max(7, min(30, H_lim))
    W_lim = max(5, min(30, W_lim))

    h = unifint(diff_lb, diff_ub, (7, H_lim))
    w = unifint(diff_lb, diff_ub, (5, W_lim))
    ih = unifint(diff_lb, diff_ub, (3, (h - 1) // 2))

    sub = np.full((ih, 5), bgc, dtype=int)
    for r in range(0, ih - 1):
        sub[r, 2] = ca
    sub[ih - 2, :] = ca
    sub[ih - 1, 0] = ca
    sub[ih - 1, 4] = ca
    subb = np.where(sub == ca, cb, sub)
    fig = np.vstack([sub, subb[::-1]])

    loci = random.randint(0, h - 2 * ih)
    locj = random.randint(0, w - 5)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, locj + 2] = ca
    gi[loci + 2 * ih - 1, locj + 2] = cb
    go[loci:loci + 2 * ih, locj:locj + 5] = fig
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour that fills every demonstration input's canvas
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if cnt.most_common(1)[0][0] in set(I.flatten().tolist()):
        cand = cnt.most_common(1)[0][0]
        if (I == cand).sum() >= hi * wi - 2:
            bgc = cand

    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    (ra, ca_), (rb, cb_) = marks[0], marks[1]
    if ca_ == cb_:
        vertical, fixed = True, ca_
        p_lo, p_hi = min(ra, rb), max(ra, rb)
    else:
        vertical, fixed = False, ra
        p_lo, p_hi = min(ca_, cb_), max(ca_, cb_)

    def cell(p, s):
        return (p, s) if vertical else (s, p)

    ih = (p_hi - p_lo + 1) // 2
    cur = I.copy()
    ops, sels = [], []

    def draw(cells, color):
        todo = sorted({rc for rc in cells if cur[rc] != color})
        if todo:
            ops.append(color)
            sels.append(sel_of(todo))
            for rc in todo:
                cur[rc] = color

    # Each marker grows ITS OWN half identically (the 4-rotation reading):
    # from the marker, a stem toward the meeting point reaching the bar row,
    # then the 5-wide bar across, then the two tips beyond the bar.
    for marker, step in ((p_lo, 1), (p_hi, -1)):
        color = int(I[cell(marker, fixed)])
        bar_p = marker + step * (ih - 2)
        tip_p = marker + step * (ih - 1)
        stem = [cell(marker + step * k, fixed) for k in range(1, ih - 1)]
        bar = [cell(bar_p, s) for s in range(fixed - 2, fixed + 3)]
        tips = [cell(tip_p, fixed - 2), cell(tip_p, fixed + 2)]
        draw(stem, color)
        draw(bar, color)
        draw(tips, color)

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
                        f"num_examples+1 ({num_examples + 1}) for task b7249182"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b7249182"
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
                                f"for task b7249182"
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
                    f"Failed to build a complete episode for task b7249182 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b7249182-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
