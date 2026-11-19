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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of

FILL = 2
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    # 2 is the answer mark, 0 is ARCLE's transparency value (needed for the stencil paste)
    cols = [1, 3, 4, 5, 6, 7, 8, 9]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = choice((True, False))
    if mirrored:
        # after dmirror the grid is (2w) x h
        hcap = max(2, min(30, max_w))
        wcap = max(2, min(14, max_h // 2))
    else:
        hcap = max(2, min(30, max_h))
        wcap = max(2, min(14, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hcap))
    w = unifint(diff_lb, diff_ub, (2, wcap))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = set(sample(inds, numa))
    bset = set(sample(inds, numb))
    allinds = set(inds)
    # keep the task non-degenerate: at least one cell empty in BOTH panels,
    # and at least one mark in each panel
    if len(allinds - aset - bset) == 0:
        spare = choice(inds)
        aset.discard(spare)
        bset.discard(spare)
        others = [i for i in inds if i != spare]
        if len(aset) == 0:
            aset.add(choice(others))
        if len(bset) == 0:
            bset.add(choice(others))
    A = fill(c, acol, frozenset(aset))
    B = fill(c, bcol, frozenset(bset))
    gi = hconcat(A, B)
    res = allinds - aset - bset
    go = fill(c, 2, frozenset(res))
    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Two panels are laid side by side (or stacked).  Panel A is turned into the
    answer sheet: every cell it left empty becomes the mark, its own marks drop out
    to background.  Panel B is turned into a stencil: its empty cells transparent,
    its marks background-coloured.  Stamping the stencil over panel A extinguishes
    every mark B covered, leaving marks only where BOTH panels were empty."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def panels(vert):
        if vert:
            return I[: hi // 2, :], I[hi // 2:, :]
        return I[:, : wi // 2], I[:, wi // 2:]

    def two_panel_ok(P, Q):
        pp = set(int(v) for v in np.unique(P))
        qq = set(int(v) for v in np.unique(Q))
        return len(pp) == 2 and len(qq) == 2 and len(pp & qq) == 1

    # split direction: the one whose halves are each a 2-colour panel sharing one colour
    cands = []
    if wi % 2 == 0 and two_panel_ok(*panels(False)):
        cands.append(False)
    if hi % 2 == 0 and two_panel_ok(*panels(True)):
        cands.append(True)
    vert = cands[0] if len(cands) == 1 else (hi == 2 * ho and wi == wo)

    A, B = panels(vert)
    ph, pw = A.shape

    # background = the colour both panels share
    pa = set(int(v) for v in np.unique(A))
    pb = set(int(v) for v in np.unique(B))
    shared = sorted(pa & pb)
    if len(shared) == 1:
        bgc = shared[0]
    else:
        cnt = Counter(I.flatten().tolist())
        bgc = max(shared, key=lambda c: cnt[c]) if shared else cnt.most_common(1)[0][0]

    # the mark colour is a convention read off the demonstrations
    fill_c = FILL
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            extra = set(int(v) for v in np.unique(eo)) - set(int(v) for v in np.unique(ei))
            if len(extra) == 1:
                fill_c = extra.pop()
                break

    br0, bc0 = (ph, 0) if vert else (0, pw)

    a_empty = [(int(r), int(c)) for r, c in zip(*np.nonzero(A == bgc))]
    a_mark = [(int(r), int(c)) for r, c in zip(*np.nonzero(A != bgc))]
    b_empty = [(int(r) + br0, int(c) + bc0) for r, c in zip(*np.nonzero(B == bgc))]
    b_mark = [(int(r) + br0, int(c) + bc0) for r, c in zip(*np.nonzero(B != bgc))]

    # panel A -> answer sheet
    if a_empty:
        ops.append(int(fill_c)); sels.append(sel_of(a_empty))
    if a_mark:
        ops.append(int(bgc)); sels.append(sel_of(a_mark))

    # panel B -> stencil (empty cells transparent, marks background-coloured)
    if b_empty:
        ops.append(0); sels.append(sel_of(b_empty))
    if b_mark:
        ops.append(int(bgc)); sels.append(sel_of(b_mark))

    # stamp the stencil onto panel A
    ops.append(29); sels.append([br0, bc0, ph - 1, pw - 1])  # whole panel-B rectangle
    ops.append(30); sels.append([0, 0, 0, 0])                # paste origin = panel A's corner

    # keep panel A
    ops.append(33); sels.append([0, 0, ph - 1, pw - 1])      # whole panel-A rectangle
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
