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

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    # Stripe colours are irrelevant to the rule; only orientation is a structural case.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    return {"instance_plan": ex + [dict(random.choice(ex))]}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, mirror=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, round(a + (b - a) * d)))
    if mirror is None:
        mirror = random.choice([True, False])
    lim_rows = max_w if mirror else max_h
    lim_cols = max_h if mirror else max_w
    cols = list(range(10))
    nr = unifint(diff_lb, diff_ub, (1, min(10, lim_rows // 2)))
    w = unifint(diff_lb, diff_ub, (4, lim_cols))
    scols = random.sample(cols, nr)
    heights = [2] * nr
    numexp = unifint(diff_lb, diff_ub, (0, max(0, lim_rows - 2 * nr)))
    for _ in range(numexp):
        heights[random.randint(0, nr - 1)] += 1
    gi, go = [], []
    for col, a in zip(scols, heights):
        sg = [[col] * w for _ in range(a)]
        go += [list(r) for r in sg]
        ub = (a * w) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, ub))
        inds = [(i, j) for i in range(a) for j in range(w)]
        oc = [c for c in cols if c != col]
        for (i, j) in random.sample(inds, nnoise):
            sg[i][j] = random.choice(oc)
        for idx in [0, -1]:
            while sum(e == col for e in sg[idx]) < w // 2:
                locs = [j for j, e in enumerate(sg[idx]) if e != col]
                sg[idx][random.choice(locs)] = col
        gi += sg
    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if mirror:
        gi, go = gi.T, go.T
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def _bands(G, strict=True):
    """Split the rows of G into stripes (each >= 2 rows, neighbours differ in
    colour) maximising the number of cells that agree with their stripe colour."""
    n = G.shape[0]
    cnt = np.zeros((n + 1, 10), dtype=int)
    for i in range(n):
        cnt[i + 1] = cnt[i] + np.bincount(G[i], minlength=10)
    half = G.shape[1] // 2
    NEG = -10 ** 9
    dp = np.full((n + 1, 11), NEG, dtype=int)   # colour index 10 = "start"
    back = {}
    dp[0][10] = 0
    for j in range(2, n + 1):
        for i in range(0, j - 1):
            if i == 1:
                continue
            seg = cnt[j] - cnt[i]
            for c in range(10):
                # generator guarantee: a stripe's first and last line are at
                # least half its own colour
                if strict and ((G[i] == c).sum() < half or (G[j - 1] == c).sum() < half):
                    continue
                best, arg = NEG, None
                for pc in range(11):
                    if pc != c and dp[i][pc] > best:
                        best, arg = dp[i][pc], pc
                if best == NEG:
                    continue
                v = best + seg[c]
                if v > dp[j][c]:
                    dp[j][c] = v
                    back[(j, c)] = (i, arg)
    c = int(np.argmax(dp[n][:10]))
    if dp[n][c] == NEG:
        if strict:
            return _bands(G, False)
        return [(0, n - 1, int(np.bincount(G.flatten(), minlength=10).argmax()))], 0
    score = int(dp[n][c])
    out, j = [], n
    while j > 0:
        i, pc = back[(j, c)]
        out.append((i, j - 1, c))
        j, c = i, pc
    return out[::-1], score


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # orientation from I alone: whichever axis splits into cleaner stripes
    if hi < 2:
        vertical = True
    elif wi < 2:
        vertical = False
    else:
        _, sh = _bands(I)
        _, sv = _bands(I.T)
        vertical = sv > sh
    G = I.T if vertical else I
    bands, _ = _bands(G)
    # Unit of work: one LINE (row / column) at a time, stripe by stripe.
    # Each noisy line of a stripe is repainted entirely with that stripe's
    # colour; already-clean lines are skipped (painting them would be a no-op).
    for (s, e, col) in bands:
        for k in range(s, e + 1):
            if np.all(G[k] == col):
                continue
            if vertical:
                cells = [(r, k) for r in range(hi)]   # full column k
            else:
                cells = [(k, c) for c in range(wi)]   # full row k
            ops.append(int(col)); sels.append(sel_of(cells))
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
