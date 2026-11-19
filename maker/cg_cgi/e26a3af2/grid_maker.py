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

VARIANTS = [{"transpose": False}, {"transpose": True}]

def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}

def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    v = a + d * (b - a)
    return max(a, min(b, int(round(v)) if random.random() < 0.5 else int(v)))

def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    L = min(max_h, max_w)
    cols = list(range(10))
    nr = _unifint(diff_lb, diff_ub, (1, min(10, L // 2)))
    w = _unifint(diff_lb, diff_ub, (min(4, L), L))
    scols = random.sample(cols, nr)
    sgs = [[[col] * w for _ in range(2)] for col in scols]
    numexp = _unifint(diff_lb, diff_ub, (0, L - 2 * nr))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))
    sgs2 = []
    for idx, col in enumerate(scols):
        sg = [list(r) for r in sgs[idx]]
        a, b = len(sg), w
        ub = (a * b) // 2 - 1
        nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = [(i, j) for i in range(a) for j in range(b)]
        oc = [c for c in cols if c != col]
        for (i, j) in random.sample(inds, nnoise):
            sg[i][j] = random.choice(oc)
        for ri in [0, a - 1]:
            while sum(e == col for e in sg[ri]) < w // 2:
                locs = [j for j, e in enumerate(sg[ri]) if e != col]
                sg[ri][random.choice(locs)] = col
        sgs2.append(sg)
    gi = [r for sg in sgs2 for r in sg]
    go = [r for sg in sgs for r in sg]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}

def _bands(G):
    # Split the lines (rows) of G into contiguous bands of >= 2 lines, each
    # with one colour, adjacent bands differing, minimising off-colour cells.
    n = G.shape[0]
    if n < 2:
        c = int(np.bincount(G.flatten(), minlength=10).argmax())
        return [(0, n - 1, c)], 0
    cnt = np.zeros((n + 1, 10), dtype=int)
    for i in range(n):
        cnt[i + 1] = cnt[i] + np.bincount(G[i], minlength=10)
    W = G.shape[1]
    INF = 10 ** 9
    best = [[INF] * 10 for _ in range(n + 1)]
    back = [[None] * 10 for _ in range(n + 1)]
    for e in range(2, n + 1):
        for s in range(0, e - 1):
            if s == 1:
                continue
            seg = cnt[e] - cnt[s]
            for c in range(10):
                cost = (e - s) * W - seg[c]
                if s == 0:
                    prev, pc = 0, None
                else:
                    prev, pc = min((best[s][p], p) for p in range(10) if p != c)
                    if prev >= INF:
                        continue
                if prev + cost < best[e][c]:
                    best[e][c] = prev + cost
                    back[e][c] = (s, pc)
    c = int(np.argmin(best[n])); total = best[n][c]
    out = []; e = n
    while e > 0:
        s, pc = back[e][c]
        out.append((s, e - 1, c)); e, c = s, pc
    return out[::-1], total

def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Orientation measured from I only: the axis whose stripe split fits best.
    bh, ch = _bands(I)
    bv, cv = _bands(I.T)
    vertical = cv < ch
    bands = bv if vertical else bh
    G = I.T if vertical else I
    ops, sels = [], []
    # Unit = one line (row, or column for vertical stripes): each noisy line is
    # repainted whole with the colour of the stripe it belongs to.
    for (s, e, col) in bands:
        for k in range(s, e + 1):
            if np.all(G[k] == col):
                continue                         # line already clean
            ops.append(int(col))
            if vertical:
                sels.append([0, k, hi - 1, 0])   # exact full column k
            else:
                sels.append([k, 0, 0, wi - 1])   # exact full row k
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
