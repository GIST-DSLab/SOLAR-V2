"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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

H_OUT = 10
VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ncols = random.randint(1, 9)
    ccols = random.sample(rem, ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, mirror=None) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
    ccols = list(ccols)
    if mirror is None:
        mirror = random.choice((True, False))
    hcap = max(2, min(6, max_h))
    wcap = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, hcap))
        w = unifint(diff_lb, diff_ub, (8, wcap))
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w // 2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (H_OUT, w))
        hoffs = randint(0, ow // 2 + 1)
        for k in range(H_OUT // oh + 1):
            go = paint(go, shift(plcd, (k * oh, k * hoffs)))
        if len(palette(go[h:])) <= 1:
            continue
        gi = go[:h]
        if mirror:
            gi = vmirror(gi)
            go = vmirror(go)
        # keep only instances whose repetition step is readable from the input
        if _extend(gi) == go:
            break
    return {"input": gi, "output": go}


def _period(I):
    """The repetition step (dr, dc), measured from the input exactly as the
    rule defines it: the shift under which the drawn cells overlap themselves
    most, with every overlapping cell agreeing in colour."""
    G = tuple(tuple(int(v) for v in row) for row in I)
    x1 = merge(fgpartition(G))
    x14 = toindices(x1)
    cands = remove(ORIGIN, product(interval(1, 6, 1), interval(-10, 10, 1)))
    ok = sfilter(cands, lambda v: len(intersection(x1, shift(x1, v))) ==
                 len(intersection(x14, shift(x14, v))) and
                 len(intersection(x1, shift(x1, v))) > 0)
    best = valmax(ok, lambda v: len(intersection(x1, shift(x1, v))))
    top = sfilter(ok, lambda v: len(intersection(x1, shift(x1, v))) == best)
    return argmax(top, lambda v: v[0] * v[1])


def _extend(gi):
    """The 10-row strip the measured step produces (used to reject ambiguous
    instances in generate)."""
    G = tuple(tuple(int(v) for v in row) for row in gi)
    x1 = merge(fgpartition(G))
    v = _period(G)
    out = canvas(mostcolor(G), (H_OUT, len(G[0])))
    for k in range(H_OUT):
        out = paint(out, shift(x1, multiply(v, k)))
    return out


def derive_operations(I, O=None, examples=None):
    """Rule: the input is the top of a 10-row strip in which a motif repeats
    with step (dr, dc).  Grow the canvas to 10 rows, lay the background on the
    added rows, then extend the repetition: copy the last dr finished rows and
    paste them dr rows lower, shifted by dc -- one copy per period."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    G = tuple(tuple(int(v) for v in row) for row in I)
    bgc = int(mostcolor(G))
    dr, dc = _period(I)

    ops, sels = [], []
    # 1. grow the canvas to the rule's 10 rows (full rectangle)
    ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
    cur = np.zeros((H_OUT, w), dtype=int)
    cur[:h] = I
    # 2. lay the background over the rows the resize added (full rectangle)
    if bgc != 0:
        ops.append(bgc); sels.append([h, 0, H_OUT - 1 - h, w - 1])
        cur[h:] = bgc

    # 3. continue the repetition, one period (dr rows) at a time
    sw = w - abs(dc)                      # width of the band that stays on-grid
    sc = 0 if dc >= 0 else -dc            # source column
    tc = dc if dc >= 0 else 0             # target column
    n = h
    while n < H_OUT and sw > 0:
        L = min(dr, H_OUT - n)
        sr = n - dr
        src = cur[sr:sr + L, sc:sc + sw].copy()
        new = cur.copy()
        new[n:n + L, tc:tc + sw] = src
        if not np.array_equal(new, cur):
            ops.append(29); sels.append([sr, sc, L - 1, sw - 1])        # CopyO band
            if bgc != 0 and (src == 0).any():
                # Paste cannot carry 0s: clear the landing rectangle first
                ops.append(0); sels.append([n, tc, L - 1, sw - 1])
            ops.append(30); sels.append([n, tc, 0, 0])                  # Paste
            cur = new
        n += L

    ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
