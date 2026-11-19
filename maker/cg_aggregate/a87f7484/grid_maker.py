"""
ARC Task: a87f7484 (RE-ARC) — LLM-generated grid_maker
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

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    # stacked dimension (num*h) and the other dimension (w)
    stack_lim = max_w if transpose else max_h
    side_lim = max_h if transpose else max_w
    h = _unifint(diff_lb, diff_ub, (2, min(5, stack_lim // 3)))
    w = _unifint(diff_lb, diff_ub, (2, side_lim))
    num = _unifint(diff_lb, diff_ub, (3, min(stack_lim // h, 9)))
    remcols = [c for c in range(10) if c != bgc]
    ccols = random.sample(remcols, num)
    ncd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    origlocs = random.sample(inds, nc)
    canbrem = set(origlocs)
    canbeadd = set(inds) - set(origlocs)
    otherlocs = set(origlocs)
    nchangesinv = _unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for _ in range(nchanges):
        if random.choice((True, False)):
            if len(canbrem) > 1:
                ch = random.choice(sorted(canbrem))
                otherlocs.discard(ch); canbrem.discard(ch)
            elif len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd))
                otherlocs.add(ch); canbeadd.discard(ch)
        else:
            if len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd))
                otherlocs.add(ch); canbeadd.discard(ch)
            elif len(canbrem) > 1:
                ch = random.choice(sorted(canbrem))
                otherlocs.discard(ch); canbrem.discard(ch)

    def mk(col, locs):
        g = np.full((h, w), bgc, dtype=int)
        for (i, j) in locs:
            g[i, j] = col
        return g

    go = mk(ccols[0], origlocs)
    grids = [go] + [mk(cc, otherlocs) for cc in ccols[1:]]
    random.shuffle(grids)
    gi = np.vstack(grids)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def _panel_check(panels):
    # background = the one colour present in every panel
    common = set(np.unique(panels[0]).tolist())
    for p in panels[1:]:
        common &= set(np.unique(p).tolist())
    if len(common) != 1:
        return None
    bg = next(iter(common))
    # each panel: background + exactly one foreground colour
    for p in panels:
        if len(np.unique(p)) != 2:
            return None
    masks = [(p != bg) for p in panels]
    keys = [m.tobytes() for m in masks]
    cnt = Counter(keys)
    if len(cnt) != 2:
        return None
    odd_keys = [k for k, v in cnt.items() if v == 1]
    if len(odd_keys) != 1 or (len(panels) > 2 and max(cnt.values()) != len(panels) - 1):
        return None
    return keys.index(odd_keys[0])


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    found = None
    # panels stacked as row bands
    for num in range(3, hi + 1):
        if hi % num:
            continue
        s = hi // num
        odd = _panel_check([I[k * s:(k + 1) * s, :] for k in range(num)])
        if odd is not None:
            found = ('h', s, odd)
            break
    if found is None:
        for num in range(3, wi + 1):
            if wi % num:
                continue
            s = wi // num
            odd = _panel_check([I[:, k * s:(k + 1) * s] for k in range(num)])
            if odd is not None:
                found = ('v', s, odd)
                break

    ops, sels = [], []
    if found is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    orient, s, odd = found
    # Crop the whole odd panel rectangle (background included) — full rectangle intended.
    if orient == 'h':
        ops.append(33); sels.append([odd * s, 0, s - 1, wi - 1])
        ho, wo = s, wi
    else:
        ops.append(33); sels.append([0, odd * s, hi - 1, s - 1])
        ho, wo = hi, s
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a87f7484"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a87f7484"
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
                                f"for task a87f7484"
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
                    f"Failed to build a complete episode for task a87f7484 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a87f7484-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
