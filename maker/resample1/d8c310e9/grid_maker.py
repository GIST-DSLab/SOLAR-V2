"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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


def _hperiod(cells):
    # DSL hperiod on an object given as dict (r,c)->color
    if not cells:
        return 1
    mr = min(r for r, c in cells); mc = min(c for r, c in cells)
    norm = {(r - mr, c - mc): v for (r, c), v in cells.items()}
    w = max(c for r, c in norm) + 1
    for p in range(1, w):
        ok = True
        for (r, c), v in norm.items():
            if c - p >= 0 and norm.get((r, c - p)) != v:
                ok = False
                break
        if ok:
            return p
    return w


def sample_colors(num_examples=None) -> dict:
    bgc = random.randint(0, 9)
    n_ex = num_examples if num_examples else 3
    ks = [0, 1, 2, 3]
    if n_ex >= 4:
        ex = ks[:] + [random.choice(ks) for _ in range(n_ex - 4)]
        random.shuffle(ex)
    else:
        ex = random.sample(ks, n_ex)
    plan = [{"rot": k} for k in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a))); hi = a + int(round(ub * (b - a)))
    lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.randint(0, 3)
    Hmax, Wmax = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    Hmax = min(Hmax, 30); Wmax = min(Wmax, 30)
    for _ in range(200):
        h = _unif(diff_lb, diff_ub, (3, Hmax))
        w = _unif(diff_lb, diff_ub, (10, Wmax))
        p = _unif(diff_lb, diff_ub, (2, (w - 1) // 3))
        remcols = [c for c in range(10) if c != bgc]
        numc = _unif(diff_lb, diff_ub, (1, 9))
        ccols = random.sample(remcols, numc)
        obj = {}
        for j in range(p):
            numcells = _unif(diff_lb, diff_ub, (1, h - 1))
            for ii in range(h - 1, h - numcells - 1, -1):
                obj[(ii, j)] = random.choice(ccols)
        full = dict(obj)
        for (r, c), v in obj.items():
            full[(r, c + p)] = v
        addonw = random.randint(0, p)
        for (r, c), v in obj.items():
            if c < addonw:
                full[(r, c + 2 * p)] = v
        ls = random.randint(0, addonw)
        gi = np.full((h, w), bgc, dtype=int)
        vis = {}
        for (r, c), v in full.items():
            cc = c - ls
            if 0 <= cc < w:
                gi[r, cc] = v
                vis[(r, cc)] = v
        q = _hperiod(vis)
        go = gi.copy()
        for k in range(0, w + 1, q):
            for (r, c), v in vis.items():
                if c + k < w:
                    go[r, c + k] = v
        go_true = gi[:, [c % p for c in range(w)]]
        if not np.array_equal(go, go_true):
            continue
        gi = np.rot90(gi, -rot); go = np.rot90(go, -rot)
        return {"input": [list(map(int, r)) for r in gi],
                "output": [list(map(int, r)) for r in go]}
    raise ValueError("failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    idx = np.arange(H * W).reshape(H, W)
    # frame where the pattern sits bottom-left (top row & right column are pure background)
    for k in range(4):
        J = np.rot90(I, k)
        if len(set(J[0].tolist())) == 1 and len(set(J[:, -1].tolist())) == 1:
            break
    Jidx = np.rot90(idx, k)
    hJ, wJ = J.shape
    bgc = int(J[0, -1])
    vis = {(r, c): int(J[r, c]) for r in range(hJ) for c in range(wJ) if J[r, c] != bgc}
    q = _hperiod(vis)
    rmin = min(r for r, c in vis)

    def to_I(r, c):
        return divmod(int(Jidx[r, c]), W)

    def rect_I(r0, c0, r1, c1):
        pts = [to_I(r0, c0), to_I(r1, c1)]
        rr = sorted(p[0] for p in pts); cc = sorted(p[1] for p in pts)
        return [rr[0], cc[0], rr[1] - rr[0], cc[1] - cc[0]], (rr[0], cc[0])

    G = I.copy()
    ops, sels = [], []
    zero_fix = []
    cur_clip_w = None
    for off in range(q, wJ, q):
        bw = min(q, wJ - off)
        changes = False
        for r in range(rmin, hJ):
            for c in range(bw):
                sv = int(J[r, c])
                tr, tc = to_I(r, c + off)
                if sv != 0 and G[tr, tc] != sv:
                    changes = True
        if changes:
            if cur_clip_w != bw:
                # CopyI: one full period-wide rectangular block of the pattern (bg included)
                box, _ = rect_I(rmin, 0, hJ - 1, bw - 1)
                ops.append(28); sels.append(box)
                cur_clip_w = bw
            _, tl = rect_I(rmin, off, hJ - 1, off + bw - 1)
            ops.append(30); sels.append([tl[0], tl[1], 0, 0])
            for r in range(rmin, hJ):
                for c in range(bw):
                    sv = int(J[r, c])
                    tr, tc = to_I(r, c + off)
                    if sv != 0:
                        G[tr, tc] = sv
        # pattern cells of color 0 are not carried by Paste; fix them explicitly
        for r in range(rmin, hJ):
            for c in range(bw):
                if J[r, c] == 0 and bgc != 0:
                    tr, tc = to_I(r, c + off)
                    if G[tr, tc] != 0:
                        zero_fix.append((tr, tc))
    if zero_fix:
        ops.append(0); sels.append(sel_of(zero_fix))
        for t in zero_fix:
            G[t] = 0
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
