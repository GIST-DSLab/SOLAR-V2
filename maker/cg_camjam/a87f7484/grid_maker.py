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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    perm = [c for c in cols if c != bgc]
    random.shuffle(perm)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "ccols": perm, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccols=None, mirror=None):
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
        random.shuffle(ccols)
    if mirror is None:
        mirror = random.choice((True, False))
    # pre-mirror: panels stacked along height (h*num), panel width w
    stack_max = max_w if mirror else max_h
    wide_max = max_h if mirror else max_w
    h = _unif(diff_lb, diff_ub, (2, min(5, stack_max // 3)))
    w = _unif(diff_lb, diff_ub, (2, wide_max))
    num = _unif(diff_lb, diff_ub, (3, min(stack_max // h, 9, len(ccols))))
    cc = list(ccols[:num])
    ncd = _unif(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    origlocs = set(random.sample(inds, nc))
    canbrem = set(origlocs)
    canbeadd = set(inds) - origlocs
    other = set(origlocs)
    nchinv = _unif(diff_lb, diff_ub, (0, h * w - 1))
    nch = h * w - nchinv
    for _ in range(nch):
        if random.choice((True, False)):
            if len(canbrem) > 1:
                ch = random.choice(sorted(canbrem)); other.discard(ch); canbrem.discard(ch)
            elif len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd)); other.add(ch); canbeadd.discard(ch)
        else:
            if len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd)); other.add(ch); canbeadd.discard(ch)
            elif len(canbrem) > 1:
                ch = random.choice(sorted(canbrem)); other.discard(ch); canbrem.discard(ch)

    def mk(col, locs):
        g = [[bgc] * w for _ in range(h)]
        for i, j in locs:
            g[i][j] = col
        return g

    go = mk(cc[0], origlocs)
    grids = [go] + [mk(c, other) for c in cc[1:]]
    random.shuffle(grids)
    gi = [row for g in grids for row in g]
    if mirror:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def _odd_panel(panels):
    # background = the one colour present in every panel
    common = set(np.unique(panels[0]).tolist())
    for p in panels[1:]:
        common &= set(np.unique(p).tolist())
    if len(common) != 1:
        return None
    bg = common.pop()
    pats = []
    for p in panels:
        if len(np.unique(p)) != 2:
            return None
        pats.append(frozenset(zip(*np.nonzero(p != bg))))
    if len(set(pats)) != 2:
        return None
    odd = [k for k, s in enumerate(pats) if pats.count(s) == 1]
    if len(odd) != 1 or pats.count(pats[(odd[0] + 1) % len(pats)]) != len(pats) - 1:
        return None
    return odd[0]


def derive_operations(I, O=None, examples=None):
    # Reading (shared by maker and golf entry): the input is a strip of equal
    # panels repeating one foreground pattern; one panel's pattern differs.
    # That odd panel, taken whole, is the answer -> a single crop.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cands = []
    for num in range(3, wi + 1):
        if wi % num == 0:
            s = wi // num
            odd = _odd_panel([I[:, k * s:(k + 1) * s] for k in range(num)])
            if odd is not None:
                cands.append((num, 1, 'v', s, odd))
    for num in range(3, hi + 1):
        if hi % num == 0:
            s = hi // num
            odd = _odd_panel([I[k * s:(k + 1) * s, :] for k in range(num)])
            if odd is not None:
                cands.append((num, 0, 'h', s, odd))
    cands.sort()
    num, _, orient, s, odd = cands[-1]
    ops, sels = [], []
    # Crop to the full odd-panel rectangle (background included) -> output
    if orient == 'v':
        ops.append(33); sels.append([0, odd * s, hi - 1, s - 1]); ho, wo = hi, s
    else:
        ops.append(33); sels.append([odd * s, 0, s - 1, wi - 1]); ho, wo = s, wi
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
