"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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
    rem = [c for c in cols if c != bgc]
    barc = random.choice(rem)
    patpool = [c for c in rem if c != barc]
    return {"bgc": bgc, "barc": barc, "patcols": patpool}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, barc=1, patcols=None, **kw) -> dict:
    if patcols is None:
        patcols = [c for c in range(10) if c not in (bgc, barc)]
    patcols = [c for c in patcols if c not in (bgc, barc)]
    vertical = random.choice((True, False))
    L = max_h if vertical else max_w
    S = max_w if vertical else max_h
    for _ in range(200):
        d = _unif(diff_lb, diff_ub, (2, 9))
        k = 4 if d < 7 else 3
        num = _unif(diff_lb, diff_ub, (2, k))
        if d <= S and num * d + num - 1 <= L:
            break
    else:
        d, num = 2, 2
    ncols = min(len(patcols), _unif(diff_lb, diff_ub, (1, 8)))
    pc = random.sample(patcols, max(1, ncols))
    c = np.full((d, d), bgc, dtype=int)
    inds = [(i, j) for i in range(d) for j in range(d)]
    ncells = _unif(diff_lb, diff_ub, (1, d * d - 1))
    for (i, j) in random.sample(inds, ncells):
        c[i, j] = random.choice(pc)
    bg = np.full((d, d), bgc, dtype=int)
    bar = np.full((d, 1), barc, dtype=int)
    fill = random.randrange(num)
    gi = c.copy() if fill == 0 else bg.copy()
    go = c.copy()
    for j in range(num - 1):
        c = np.rot90(c, -1)  # rot90 CW
        gi = np.hstack([gi, bar, c if j + 1 == fill else bg])
        go = np.hstack([go, bar, c])
    if vertical:
        gi = np.rot90(gi, -1)
        go = np.rot90(go, -1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    vertical = h > w
    d = min(h, w)
    L = max(h, w)
    num = (L + 1) // (d + 1)

    def origin(k):
        return (k * (d + 1), 0) if vertical else (0, k * (d + 1))

    panels = []
    for k in range(num):
        r, c = origin(k)
        panels.append(I[r:r + d, c:c + d])
    fi = next(k for k in range(num) if len(np.unique(panels[k])) > 1)
    bgc = int(next(panels[k][0, 0] for k in range(num) if k != fi))
    src = panels[fi]
    sr, sc = origin(fi)

    G = I.copy()
    ops, sels = [], []
    copied = False
    for k in range(num):
        if k == fi:
            continue
        r, c = origin(k)
        # paste filled panel into this empty panel
        new = G.copy()
        reg = new[r:r + d, c:c + d]
        mask = src != 0
        reg[mask] = src[mask]
        if not np.array_equal(new, G):
            if not copied:
                ops.append(28); sels.append([sr, sc, d - 1, d - 1])  # full panel rect
                copied = True
            ops.append(30); sels.append([r, c, 0, 0])
            G = new
        # 0-valued pattern cells are not pasted; paint them explicitly
        if bgc != 0:
            zc = [(r + i, c + j) for i in range(d) for j in range(d) if src[i, j] == 0]
            if zc:
                ops.append(0); sels.append(sel_of(zc))
                for (a, b) in zc:
                    G[a, b] = 0
        # rotate in place: panel k = rot90^(k-fi) CW of filled panel
        t = (k - fi) % 4
        reg = G[r:r + d, c:c + d]
        if t == 1:
            nr = np.rot90(reg, -1)
            seq = [25]
        elif t == 3:
            nr = np.rot90(reg, 1)
            seq = [24]
        else:
            nr = np.rot90(reg, 2)
            seq = [26, 27]
        if not np.array_equal(nr, reg):
            if t == 2:
                f1 = np.fliplr(reg)
                if np.array_equal(f1, reg):
                    seq = [27]
                elif np.array_equal(f1, nr):
                    seq = [26]
            for op in seq:
                ops.append(op); sels.append([r, c, d - 1, d - 1])  # full square panel
            G[r:r + d, c:c + d] = nr

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
