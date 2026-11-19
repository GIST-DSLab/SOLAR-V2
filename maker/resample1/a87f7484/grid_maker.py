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

VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, transposed=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if bgc is None:
        bgc = random.choice(range(10))
    if transposed is None:
        transposed = random.choice([True, False])
    long_cap = min(30, max_w if transposed else max_h)
    wide_cap = min(30, max_h if transposed else max_w)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(5, long_cap // 3))))
    w = unifint(diff_lb, diff_ub, (2, max(2, wide_cap)))
    num = unifint(diff_lb, diff_ub, (3, max(3, min(long_cap // h, 9))))
    remcols = [c for c in range(10) if c != bgc]
    ccols = random.sample(remcols, num)
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    origlocs = random.sample(inds, nc)
    canbrem = set(origlocs)
    canbeadd = set(inds) - set(origlocs)
    otherlocs = set(origlocs)
    nchangesinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for _ in range(nchanges):
        if random.choice((True, False)):
            if len(canbrem) > 1:
                ch = random.choice(sorted(canbrem)); otherlocs.discard(ch); canbrem.discard(ch)
            elif len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd)); otherlocs.add(ch); canbeadd.discard(ch)
        else:
            if len(canbeadd) > 1:
                ch = random.choice(sorted(canbeadd)); otherlocs.add(ch); canbeadd.discard(ch)
            elif len(canbrem) > 1:
                ch = random.choice(sorted(canbrem)); otherlocs.discard(ch); canbrem.discard(ch)

    def tile(col, locs):
        g = [[bgc] * w for _ in range(h)]
        for (i, j) in locs:
            g[i][j] = col
        return g

    go = tile(ccols[0], origlocs)
    grids = [go] + [tile(cc, otherlocs) for cc in ccols[1:]]
    random.shuffle(grids)
    gi = [row for g in grids for row in g]
    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    found = None
    for axis in (0, 1):
        n = H if axis == 0 else W
        for k in range(3, n + 1):
            if n % k:
                continue
            t = n // k
            parts = []
            for i in range(k):
                if axis == 0:
                    parts.append((i * t, 0, t, W, I[i * t:(i + 1) * t, :]))
                else:
                    parts.append((0, i * t, H, t, I[:, i * t:(i + 1) * t]))
            pals = [set(np.unique(p[4]).tolist()) for p in parts]
            if any(len(p) != 2 for p in pals):
                continue
            common = set.intersection(*pals)
            if len(common) != 1:
                continue
            bg = next(iter(common))
            others = [next(iter(p - common)) for p in pals]
            if len(set(others)) != k:
                continue
            masks = [tuple(map(tuple, (p[4] != bg).astype(int))) for p in parts]
            cnt = {}
            for m in masks:
                cnt[m] = cnt.get(m, 0) + 1
            if len(cnt) != 2:
                continue
            odd = [i for i, m in enumerate(masks) if cnt[m] == 1]
            if len(odd) != 1:
                continue
            found = parts[odd[0]]
            break
        if found:
            break
    r0, c0, th, tw, _ = found
    # Crop the canvas to the odd-one-out tile: its full rectangle
    # (background included) is exactly the answer, so a bbox is correct here.
    ops = [33, 34]
    sels = [[r0, c0, th - 1, tw - 1], [0, 0, th - 1, tw - 1]]
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
