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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(range(10), 2)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    inh = random.randint(5, h - 2)
    inw = random.randint(5, w - 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
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
    S = np.array(sgi)
    go = S.copy()
    for i, j in zip(*np.where(S == fgc)):
        go[i, :] = fgc
        go[:, j] = fgc
    numci = unifint(diff_lb, diff_ub, (3, 10))
    ccols = random.sample(cols, 13 - numci)
    gi = np.array([[random.choice(ccols) for _ in range(w)] for _ in range(h)])
    loci = random.randint(1, h - inh - 1)
    locj = random.randint(1, w - inw - 1)
    gi[loci:loci + inh, locj:locj + inw] = S
    a, b, c, d = loci, locj, loci + inh - 1, locj + inw - 1
    remcols = [x for x in ccols if x != bgc] or [x for x in cols if x != bgc]
    gi[a - 1, random.randint(b, d)] = random.choice(remcols)
    gi[random.randint(a, c), b - 1] = random.choice(remcols)
    gi[c + 1, random.randint(b, d)] = random.choice(remcols)
    gi[random.randint(a, c), d + 1] = random.choice(remcols)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Canvas colour of the rectangle: colour of the largest same-colour blob.
    seen = np.zeros_like(I, dtype=bool)
    best = (0, None, None)
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0]:
                continue
            col = I[r0, c0]
            st = [(r0, c0)]; seen[r0, c0] = True; comp = []
            while st:
                r, c = st.pop(); comp.append((r, c))
                for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True; st.append((nr, nc))
            if len(comp) > best[0]:
                best = (len(comp), col, comp)
    _, bgc, comp = best
    R0 = min(p[0] for p in comp); R1 = max(p[0] for p in comp)
    C0 = min(p[1] for p in comp); C1 = max(p[1] for p in comp)

    # The rectangle: largest box inside that blob's bbox whose border lines are
    # pure canvas colour and whose inside holds a single dot colour, one dot per
    # row and per column (noisy rows/columns get rejected -- line-purity reading).
    B = (I == bgc).astype(int)
    P = np.zeros((hi + 1, wi + 1), dtype=int)
    P[1:, 1:] = B.cumsum(0).cumsum(1)

    def cnt(r0, c0, r1, c1):
        return P[r1+1, c1+1] - P[r0, c1+1] - P[r1+1, c0] + P[r0, c0]

    found = None
    for top in range(R0, R1 + 1):
        for bot in range(top + 4, R1 + 1):
            for left in range(C0, C1 + 1):
                if cnt(top, left, bot, left) != bot - top + 1:
                    continue
                for right in range(left + 4, C1 + 1):
                    if cnt(top, left, top, right) != right - left + 1:
                        break
                    if cnt(bot, left, bot, right) != right - left + 1:
                        break
                    if cnt(top, right, bot, right) != bot - top + 1:
                        continue
                    area = (bot - top + 1) * (right - left + 1)
                    if found and area <= found[0]:
                        continue
                    sub = I[top:bot+1, left:right+1]
                    rr, cc = np.where(sub != bgc)
                    if len(rr) == 0 or len(set(sub[rr, cc].tolist())) != 1:
                        continue
                    if len(set(rr.tolist())) != len(rr) or len(set(cc.tolist())) != len(cc):
                        continue
                    found = (area, top, left, bot, right)
    _, top, left, bot, right = found
    ho, wo = bot - top + 1, right - left + 1
    rect = I[top:bot+1, left:right+1]
    rr, cc = np.where(rect != bgc)
    fgc = int(rect[rr[0], cc[0]])

    ops, sels = [], []
    # 1. Cut the clean rectangle out of the noise (the whole box is the intended region).
    ops.append(33); sels.append([top, left, ho - 1, wo - 1])
    # 2. Every row holding a dot becomes a full line of the dot colour.
    for r in sorted(set(rr.tolist())):
        ops.append(fgc); sels.append(sel_of([(r, c) for c in range(wo)]))
    # 3. Every column holding a dot becomes a full line of the dot colour.
    for c in sorted(set(cc.tolist())):
        ops.append(fgc); sels.append(sel_of([(r, c) for r in range(ho)]))
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
