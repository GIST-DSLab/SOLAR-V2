"""
ARC Task: 94f9d214 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, acol=1, bcol=3) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    def two_col_pair(x, y):
        px, py = set(x.flatten().tolist()), set(y.flatten().tolist())
        return len(px) == 2 and len(py) == 2 and len(px & py) == 1

    transpose = random.choice((True, False))
    if transpose:
        hmax, wmax = min(30, max_w), min(14, max_h // 2)
    else:
        hmax, wmax = min(30, max_h), min(14, max_w // 2)
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (2, max(2, hmax)))
        w = unifint(diff_lb, diff_ub, (2, max(2, wmax)))
        inds = [(i, j) for i in range(h) for j in range(w)]
        numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
        numa = random.choice((numadev, h * w - numadev))
        numb = random.choice((numadev, h * w - numbdev))
        numa = min(max(1, numa), h * w - 1)
        numb = min(max(1, numb), h * w - 1)
        aset = random.sample(inds, numa)
        bset = random.sample(inds, numb)
        A = np.full((h, w), bgc, dtype=int)
        B = np.full((h, w), bgc, dtype=int)
        for r, c in aset:
            A[r, c] = acol
        for r, c in bset:
            B[r, c] = bcol
        gi = np.hstack([A, B])
        # reject inputs that could also be read as split the other way (ambiguous halves)
        if h % 2 == 0 and two_col_pair(gi[:h // 2], gi[h // 2:]):
            continue
        break
    go = np.full((h, w), bgc, dtype=int)
    go[(A == bgc) & (B == bgc)] = 2
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- find the two halves from I alone: each half is 2 colours, sharing exactly one (bgc) ---
    cands = []
    if wi % 2 == 0:
        h2, w2 = hi, wi // 2
        cands.append((I[:, :w2], I[:, w2:], (0, w2), h2, w2))
    if hi % 2 == 0:
        h2, w2 = hi // 2, wi
        cands.append((I[:h2, :], I[h2:, :], (h2, 0), h2, w2))

    def valid_splits(G):
        G = np.asarray(G, dtype=int)
        H, W = G.shape
        out = []
        if W % 2 == 0:
            out.append((G[:, :W // 2], G[:, W // 2:], (0, W // 2), H, W // 2))
        if H % 2 == 0:
            out.append((G[:H // 2, :], G[H // 2:, :], (H // 2, 0), H // 2, W))
        good = []
        for a, b, off, h2, w2 in out:
            pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
            if len(pa) == 2 and len(pb) == 2 and len(pa & pb) == 1:
                good.append((a, b, off, h2, w2, (pa & pb).pop(), (pa - pb).pop(), (pb - pa).pop()))
        return good

    # colour roles of the episode (first half's mark colour), read from unambiguous demos
    roles = None
    for ex in (examples or []):
        g = valid_splits(ex[0])
        if len(g) == 1:
            roles = (g[0][5], g[0][6], g[0][7])
            break
    good = valid_splits(I)
    if roles is not None and len(good) > 1:
        good = [g for g in good if (g[5], g[6], g[7]) == roles] or good
    pick = good[0][:6] if good else None
    if pick is None:
        a, b, off, h2, w2 = cands[0]
        sh = set(a.flatten().tolist()) & set(b.flatten().tolist())
        bgc = sorted(sh)[0] if sh else int(np.bincount(a.flatten()).argmax())
        pick = (a, b, off, h2, w2, bgc)
    A, B, (br, bc), ho, wo, bgc = pick
    bgc = int(bgc)
    acol = [int(v) for v in set(A.flatten().tolist()) if v != bgc]
    bcol = [int(v) for v in set(B.flatten().tolist()) if v != bgc]
    acol = acol[0] if acol else bgc
    bcol = bcol[0] if bcol else bgc

    a_bg = [(r, c) for r in range(ho) for c in range(wo) if A[r, c] == bgc]
    a_mark = [(r, c) for r in range(ho) for c in range(wo) if A[r, c] != bgc]
    b_bg = [(r, c) for r in range(ho) for c in range(wo) if B[r, c] == bgc]
    b_mark = [(r, c) for r in range(ho) for c in range(wo) if B[r, c] != bgc]
    both_empty = [p for p in a_bg if B[p] == bgc]

    ops, sels = [], []
    if both_empty:
        # 1. In half B, every empty (background) cell becomes a 2 "hole" marker ...
        ops.append(2); sels.append(sel_of([(r + br, c + bc) for r, c in b_bg]))
        # ... and B's own marks become transparent (0), so only the holes travel.
        if bcol != 0 and any(A[p] == bgc for p in b_mark):
            ops.append(0); sels.append(sel_of([(r + br, c + bc) for r, c in b_mark]))
        # 2. Lay B's hole layer over half A (copy the whole B half, paste at A's origin).
        ops.append(29); sels.append([br, bc, ho - 1, wo - 1])   # whole B rectangle
        ops.append(30); sels.append([0, 0, 0, 0])
    # 3. Wherever A itself is marked the cell is not empty in both: back to background.
    ops.append(bgc); sels.append(sel_of(a_mark))
    # 4. The B half has been consumed: crop the canvas to A.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
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
                                f"for task 94f9d214"
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
                    f"Failed to build a complete episode for task 94f9d214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"94f9d214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
