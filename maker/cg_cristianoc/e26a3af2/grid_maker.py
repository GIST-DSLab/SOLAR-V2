"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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
    # Band colours are re-sampled per instance by the generator and the rule does not
    # depend on them. The only discrete structural variant is the orientation
    # (horizontal bands vs. dmirror'ed vertical bands): plan it so both appear.
    variants = [{"transpose": False}, {"transpose": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    L = max_w if transpose else max_h      # extent along which bands stack
    W = max_h if transpose else max_w      # band length

    def uf(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, round(lo + d * (hi - lo) + random.uniform(-0.5, 0.5))))

    nr = uf(1, min(10, L // 2))
    w = uf(4, min(30, W))
    scols = random.sample(range(10), nr)
    sgs = [[[c] * w for _ in range(2)] for c in scols]
    numexp = uf(0, min(30, L) - 2 * nr)
    for _ in range(numexp):
        i = random.randint(0, nr - 1)
        sgs[i].append(list(sgs[i][-1]))
    sgs2 = []
    for i, col in enumerate(scols):
        sg = [list(r) for r in sgs[i]]
        a = len(sg)
        ub = (a * w) // 2 - 1
        nn = uf(0, ub)
        inds = [(r, c) for r in range(a) for c in range(w)]
        oc = [c for c in range(10) if c != col]
        for (r, c) in random.sample(inds, nn):
            sg[r][c] = random.choice(oc)
        for ix in (0, -1):
            while sum(e == col for e in sg[ix]) < w // 2:
                locs = [j for j, e in enumerate(sg[ix]) if e != col]
                sg[ix][random.choice(locs)] = col
        sgs2.append(sg)
    gi = [r for sg in sgs2 for r in sg]
    go = [r for sg in sgs for r in sg]
    if transpose:
        gi = [list(x) for x in zip(*gi)]
        go = [list(x) for x in zip(*go)]
    return {"input": gi, "output": go}


def _bands(G):
    # Partition the lines (rows of G) into consecutive bands of >= 2 lines, adjacent
    # bands differing in colour, maximising the number of cells already matching
    # their band colour. Measured from the input only.
    n, w = G.shape
    cnt = np.zeros((n + 1, 10), int)
    for r in range(n):
        cnt[r + 1] = cnt[r] + np.bincount(G[r], minlength=10)
    NEG = -10 ** 9
    dp = [[NEG] * 10 for _ in range(n + 1)]
    back = [[None] * 10 for _ in range(n + 1)]
    for j in range(2, n + 1):
        for c in range(10):
            for i in range(0, j - 1):
                if i == 0:
                    prev, pc = 0, None
                else:
                    prev, pc = max(((dp[i][d], d) for d in range(10) if d != c),
                                   default=(NEG, None))
                if prev <= NEG // 2:
                    continue
                v = prev + cnt[j][c] - cnt[i][c]
                if v > dp[j][c]:
                    dp[j][c] = v
                    back[j][c] = (i, pc)
    c = int(np.argmax(dp[n]))
    score = dp[n][c]
    bands, j = [], n
    while j > 0:
        i, pc = back[j][c]
        bands.append((i, j - 1, c))
        j, c = i, pc
    return bands[::-1], score


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Orientation from I only: the axis along which a stripe partition explains
    # the most cells (the golf entry's "is the first line majority-coloured?" test,
    # made robust to a wide band dominating the cross-direction).
    cands = []
    if hi >= 2:
        b, s = _bands(I)
        cands.append((s, False, b))
    if wi >= 2:
        b, s = _bands(I.T)
        cands.append((s, True, b))
    _, vertical, bands = max(cands, key=lambda t: (t[0], not t[1]))
    G = I.T if vertical else I

    # Unit = one line (row, or column if vertical): each noisy line is repainted
    # whole with its stripe's colour, one Color op per line, walking the stripes
    # in order. Lines already uniform in their colour need nothing.
    ops, sels = [], []
    for (a, e, col) in bands:
        for k in range(a, e + 1):
            if np.all(G[k] == col):
                continue
            ops.append(int(col))
            if vertical:
                sels.append(sel_of([(r, k) for r in range(hi)]))   # whole column k
            else:
                sels.append(sel_of([(k, c) for c in range(wi)]))   # whole row k

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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
