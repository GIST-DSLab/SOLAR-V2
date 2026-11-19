"""
ARC Task: fafffa47 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    return int(round(random.uniform(min(a, b), max(a, b))))


def _ncols(g):
    return len(set(np.asarray(g).flatten().tolist()))


def _rule_output(I):
    I = np.asarray(I)
    hi, wi = I.shape
    top, bot = I[:hi // 2], I[hi // 2:]
    lef, rig = I[:, :wi // 2], I[:, wi // 2:]
    vertical = (_ncols(top) == 2 and _ncols(bot) == 2) and not (_ncols(lef) == 2 and _ncols(rig) == 2)
    if vertical:
        F, S = I[:hi // 2], I[hi // 2:2 * (hi // 2)]
    else:
        F, S = I[:, :wi // 2], I[:, wi // 2:2 * (wi // 2)]
    shared = sorted(set(F.flatten().tolist()) & set(S.flatten().tolist()))
    if not shared:
        return None
    b = shared[0]
    out = np.full(F.shape, b, dtype=int)
    out[(F == b) & (S == b)] = 2
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None, **kw) -> dict:
    if bgc is None or acol is None or bcol is None:
        d = sample_colors()
        bgc, acol, bcol = d["bgc"], d["acol"], d["bcol"]
    for _ in range(200):
        transpose = random.choice((True, False))
        if transpose:
            # final grid is (2w, h)
            hmax = min(30, max_w)
            wmax = min(14, max_h // 2)
        else:
            # final grid is (h, 2w)
            hmax = min(30, max_h)
            wmax = min(14, max_w // 2)
        if hmax < 2 or wmax < 2:
            continue
        h = _unifint(diff_lb, diff_ub, 2, hmax)
        w = _unifint(diff_lb, diff_ub, 2, wmax)
        n = h * w
        inds = [(r, c) for r in range(h) for c in range(w)]
        numadev = _unifint(diff_lb, diff_ub, 0, n // 2)
        numbdev = _unifint(diff_lb, diff_ub, 0, n // 2)
        numa = random.choice((numadev, n - numadev))
        numb = random.choice((numadev, n - numbdev))
        numa = min(max(1, numa), n - 1)
        numb = min(max(1, numb), n - 1)
        aset = random.sample(inds, numa)
        bset = random.sample(inds, numb)
        A = np.full((h, w), bgc, dtype=int)
        B = np.full((h, w), bgc, dtype=int)
        for r, c in aset:
            A[r, c] = acol
        for r, c in bset:
            B[r, c] = bcol
        gi = np.concatenate([A, B], axis=1)
        go = np.full((h, w), bgc, dtype=int)
        for r, c in set(inds) - (set(aset) | set(bset)):
            go[r, c] = 2
        if transpose:
            gi = gi.T.copy()
            go = go.T.copy()
        ro = _rule_output(gi)
        if ro is None or ro.shape != go.shape or not np.array_equal(ro, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def _components(cells):
    cs = set(cells)
    seen = set()
    comps = []
    for start in sorted(cs):
        if start in seen:
            continue
        comp = []
        dq = deque([start])
        seen.add(start)
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for nr, nc_ in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc_) in cs and (nr, nc_) not in seen:
                    seen.add((nr, nc_))
                    dq.append((nr, nc_))
        comps.append(sorted(comp))
    return comps


def derive_operations(I, O, examples=None):
    # The code-golf reading (per-cell pairing of the two halves: 2 exactly where
    # both halves are blank, else background) is the same reading as the maker's,
    # so the route is the same: mark both-blank cells 2, keep first half, wipe its marks.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def nc(g):
        return len(set(g.flatten().tolist()))

    top, bottom = I[:hi // 2], I[hi // 2:]
    left, right = I[:, :wi // 2], I[:, wi // 2:]
    vertical = (nc(top) == 2 and nc(bottom) == 2) and not (nc(left) == 2 and nc(right) == 2)

    if vertical:
        H1, W1 = hi // 2, wi
        F = I[:H1, :W1]
        S = I[H1:2 * H1, :W1]
    else:
        H1, W1 = hi, wi // 2
        F = I[:H1, :W1]
        S = I[:H1, W1:2 * W1]

    shared = sorted(set(F.flatten().tolist()) & set(S.flatten().tolist()))
    bgc = shared[0]

    # cells blank in BOTH halves become 2 (painted inside the first half, which is at top-left)
    both_blank = [(r, c) for r in range(H1) for c in range(W1)
                  if F[r, c] == bgc and S[r, c] == bgc]
    for comp in _components(both_blank):
        ops.append(2)
        sels.append(sel_of(comp))

    # keep only the first half: crop to its full rectangle (bbox == intended cells)
    ops.append(33)
    sels.append([0, 0, H1 - 1, W1 - 1])

    # after a transparent crop, 0-valued cells are already 0; only wipe marks that are
    # visibly not bgc now
    marks = [(r, c) for r in range(H1) for c in range(W1) if F[r, c] != bgc]
    for comp in _components(marks):
        ops.append(bgc)
        sels.append(sel_of(comp))

    ops.append(34)
    sels.append([0, 0, H1 - 1, W1 - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task fafffa47"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fafffa47"
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
                                f"for task fafffa47"
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
                    f"Failed to build a complete episode for task fafffa47 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fafffa47-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
