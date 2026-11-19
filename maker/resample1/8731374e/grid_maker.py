"""
ARC Task: 8731374e (RE-ARC) — LLM-generated grid_maker
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


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    cols = list(range(10))
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    h = _unif(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unif(diff_lb, diff_ub, (10, max(10, max_w)))
    inh = random.randint(5, h - 2)
    inw = random.randint(5, w - 2)
    num = _unif(diff_lb, diff_ub, (1, min(inh, inw)))
    mh, mw = inh - 2, inw - 2
    mat = [[bgc] * mw for _ in range(mh)]
    for k in range(num):
        if k < mh and k < mw:
            mat[k][k] = fgc
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    random.shuffle(mat)
    mat = [list(r) for r in zip(*mat)]
    sgi = [[bgc] * inw for _ in range(inh)]
    for i in range(mh):
        for j in range(mw):
            sgi[i + 1][j + 1] = mat[i][j]
    go = [row[:] for row in sgi]
    for i in range(inh):
        for j in range(inw):
            if sgi[i][j] == fgc:
                for jj in range(inw):
                    go[i][jj] = fgc
                for ii in range(inh):
                    go[ii][j] = fgc
    numci = _unif(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = random.sample(cols, numc)
    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    loci = random.randint(1, h - inh - 1)
    locj = random.randint(1, w - inw - 1)
    for i in range(inh):
        for j in range(inw):
            gi[loci + i][locj + j] = sgi[i][j]
    a, b = loci, locj
    c, d = loci + inh - 1, locj + inw - 1
    remcols = [x for x in ccols if x != bgc]
    pts = [(a - 1, random.randint(b, d)), (random.randint(a, c), b - 1),
           (c + 1, random.randint(b, d)), (random.randint(a, c), d + 1)]
    for (i, j) in pts:
        gi[i][j] = random.choice(remcols)
    return {"input": gi, "output": go}


def _largest_component(I):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    best = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = I[r, c]
            comp = []
            dq = deque([(r, c)]); seen[r, c] = True
            while dq:
                y, x = dq.popleft(); comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; dq.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    return best


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    comp = _largest_component(I)
    bgc = int(I[comp[0]])
    h, w = I.shape
    # locate the background rectangle: all-bgc border, interior uses bgc + at most one other colour
    isb = (I == bgc).astype(int)
    P = np.zeros((10, h + 1, w + 1), int)
    for k in range(10):
        P[k, 1:, 1:] = np.cumsum(np.cumsum(I == k, 0), 1)
    def cnt(k, a, b, c, d):
        if a > c or b > d:
            return 0
        return P[k, c + 1, d + 1] - P[k, a, d + 1] - P[k, c + 1, b] + P[k, a, b]
    best = None
    for r0 in range(h):
        for r1 in range(r0 + 4, h):
            colb = [cnt(bgc, r0, c, r1, c) == r1 - r0 + 1 for c in range(w)]
            both = [isb[r0, c] and isb[r1, c] for c in range(w)]
            c = 0
            while c < w:
                if not both[c]:
                    c += 1; continue
                e = c
                while e + 1 < w and both[e + 1]:
                    e += 1
                bcols = [x for x in range(c, e + 1) if colb[x]]
                for i, c0 in enumerate(bcols):
                    for c1 in reversed(bcols[i + 1:]):
                        if c1 - c0 < 4:
                            break
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if best is not None and area <= best[0]:
                            break
                        nk = sum(1 for k in range(10) if cnt(k, r0 + 1, c0 + 1, r1 - 1, c1 - 1) > 0)
                        if nk <= 2:
                            best = (area, r0, c0, r1, c1)
                            break
                c = e + 1
    _, r0, c0, r1, c1 = best
    sub = I[r0:r1 + 1, c0:c1 + 1]
    hh, ww = sub.shape
    ops, sels = [], []
    # crop canvas to the background rectangle (intended cells = the full rectangle)
    ops.append(33); sels.append([r0, c0, hh - 1, ww - 1])
    others = [int(v) for v in np.unique(sub) if v != bgc]
    if others:
        fgc = min(others, key=lambda v: int(np.sum(sub == v)))
        dots = sorted(zip(*np.where(sub == fgc)))
        for (r, c) in dots:
            cross = set((int(r), j) for j in range(ww)) | set((i, int(c)) for i in range(hh))
            ops.append(fgc); sels.append(sel_of(sorted(cross)))
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8731374e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8731374e"
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
                                f"for task 8731374e"
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
                    f"Failed to build a complete episode for task 8731374e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8731374e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
