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
from collections import Counter
from maker.sel_helpers import sel_of


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, barc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = int(round(random.uniform(lo, hi)))
    return max(a, min(b, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    # canonical frame: band made of rows; transposed output swaps dims
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (7, min(30, lim_h)))
    w = _unifint(diff_lb, diff_ub, (7, min(30, lim_w)))
    loci1, loci2 = sorted(random.sample(range(2, h - 2), 2))
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci1:loci2 + 1, :] = barc
    go = gi.copy()
    num1 = _unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = _unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = random.sample(range(w), num1)
    locs2 = random.sample(range(w), num2)
    for l1 in locs1:
        k = _unifint(diff_lb, diff_ub, (1, loci1 - 1))
        for r in random.sample(range(0, loci1), k):
            gi[r, l1] = dotc
        go[loci1 - k:loci1, l1] = barc
    for l2 in locs2:
        k = _unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        for r in random.sample(range(loci2 + 1, h), k):
            gi[r, l2] = dotc
        go[loci2 + 1:loci2 + 1 + k, l2] = barc
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): a solid band of `barc` lines splits
    the grid.  For every line perpendicular to the band, on each side, the
    number of dots is counted and a bar of that length is grown out of the
    band's edge.  Only after ALL bars are grown, every dot still visible
    (the ones no bar swallowed) is wiped to background in ONE step -- the
    leftover dots are treated as one set ("all cells still of the dot
    colour"), not cleaned line by line.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: lives only in complete, contiguous lines
    barc, vertical = None, False
    for c in colors:
        rs = [r for r in range(hi) if bool((Ia[r] == c).any())]
        if rs and len(rs) == rs[-1] - rs[0] + 1 and all(bool((Ia[r] == c).all()) for r in rs):
            barc, vertical = c, False
            break
    if barc is None:
        for c in colors:
            cs = [j for j in range(wi) if bool((Ia[:, j] == c).any())]
            if cs and len(cs) == cs[-1] - cs[0] + 1 and all(bool((Ia[:, j] == c).all()) for j in cs):
                barc, vertical = c, True
                break

    rest = [c for c in colors if c != barc]
    dotc = None
    # convention from the demonstrations: the dot colour is the one that vanishes
    if examples:
        for ex in examples:
            try:
                ei, eo = ex
            except Exception:
                ei, eo = ex["input"], ex["output"]
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            gone = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            cand = [c for c in gone if c in rest]
            if len(cand) == 1:
                dotc = cand[0]
                break
    if dotc is None or dotc not in rest:
        cnt = Counter(Ia.flatten().tolist())
        dotc = min(rest, key=lambda c: cnt[c])
    others = [c for c in rest if c != dotc]
    bgc = others[0] if others else None
    if bgc is None:
        # all non-band cells are dots? impossible by generator; fall back
        bgc = 0

    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    def cells(a, b, c):
        # rows a..b of column c in canonical frame -> real-grid cells
        return [(c, r) if vertical else (r, c) for r in range(a, b + 1)]

    ops, sels = [], []
    G = Ia.copy()

    # 1) grow bars above the band, one line at a time
    for c in range(w):
        k = int(sum(1 for r in range(0, r1) if A[r, c] == dotc))
        if k == 0:
            continue
        cl = cells(r1 - k, r1 - 1, c)
        ops.append(int(barc)); sels.append(sel_of(cl))
        for (r, cc) in cl:
            G[r, cc] = barc
    # 2) grow bars below the band
    for c in range(w):
        k = int(sum(1 for r in range(r2 + 1, h) if A[r, c] == dotc))
        if k == 0:
            continue
        cl = cells(r2 + 1, r2 + k, c)
        ops.append(int(barc)); sels.append(sel_of(cl))
        for (r, cc) in cl:
            G[r, cc] = barc
    # 3) every dot still visible on the working grid vanishes at once
    left = [(r, c) for r in range(hi) for c in range(wi) if G[r, c] == dotc]
    if left:
        ops.append(int(bgc)); sels.append(sel_of(left))

    ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
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
