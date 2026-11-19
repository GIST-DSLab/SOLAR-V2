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

# Reading of the task (shared by the maker and the code-golf entry): the grid is a strip
# of equal panels on one background; each panel is one colour painted on a mask, and all
# masks are identical except one. The answer is that odd panel, cut out whole.
# Both readings use the same unit (a whole panel) and the same order (find the odd one,
# then crop it), so the route is unchanged: one CropGrid on the full panel rectangle.

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, transpose=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if transpose is None:
        transpose = random.choice([False, True])
    lim = min(max_h, max_w)
    # panels stack along the "long" axis (rows before optional transpose)
    long_lim = max_w if transpose else max_h
    short_lim = max_h if transpose else max_w
    h = unifint(diff_lb, diff_ub, (2, min(5, long_lim // 3)))
    w = unifint(diff_lb, diff_ub, (2, min(30, short_lim)))
    num = unifint(diff_lb, diff_ub, (3, min(long_lim // h, 9)))
    remcols = [c for c in range(10) if c != bgc]
    ccols = random.sample(remcols, num)
    inds = [(i, j) for i in range(h) for j in range(w)]
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    origlocs = set(random.sample(inds, nc))
    canbrem = set(origlocs)
    canbeadd = set(inds) - origlocs
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
    if otherlocs == origlocs:
        raise ValueError("no odd panel")

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
    if gi.shape[0] > max_h or gi.shape[1] > max_w:
        raise ValueError("too big")
    return {"input": gi.tolist(), "output": go.tolist()}


def _check(panels):
    """Return index of the odd panel if `panels` is a genuine tiling, else None.
    Genuine: all panels share one background, each panel has exactly one other colour,
    those colours are pairwise distinct, and all masks agree except exactly one."""
    sets = [set(np.unique(p).tolist()) for p in panels]
    common = set.intersection(*sets)
    if len(common) != 1:
        return None
    bg = next(iter(common))
    fgs = []
    for s in sets:
        rest = s - {bg}
        if len(rest) != 1:
            return None
        fgs.append(next(iter(rest)))
    if len(set(fgs)) != len(fgs):
        return None
    masks = [(p != bg).tobytes() for p in panels]
    cnt = Counter(masks)
    if len(cnt) != 2:
        return None
    (common_mask, n_common), (odd_mask, n_odd) = cnt.most_common()
    if n_odd != 1 or n_common != len(panels) - 1:
        return None
    return masks.index(odd_mask)


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    candidates = []
    for num in range(3, wi + 1):
        if wi % num:
            continue
        s = wi // num
        odd = _check([I[:, k * s:(k + 1) * s] for k in range(num)])
        if odd is not None:
            candidates.append((num, 'v', s, odd))
    for num in range(3, hi + 1):
        if hi % num:
            continue
        s = hi // num
        odd = _check([I[k * s:(k + 1) * s, :] for k in range(num)])
        if odd is not None:
            candidates.append((num, 'h', s, odd))

    ops, sels = [], []
    if not candidates:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    candidates.sort(key=lambda t: (t[0], t[1] == 'v'))
    num, orient, s, odd = candidates[-1]
    # CropGrid on the whole odd panel rectangle (background included) -> that panel is the output
    if orient == 'v':
        ops.append(33); sels.append([0, odd * s, hi - 1, s - 1])
        ho, wo = hi, s
    else:
        ops.append(33); sels.append([odd * s, 0, s - 1, wi - 1])
        ho, wo = s, wi
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
