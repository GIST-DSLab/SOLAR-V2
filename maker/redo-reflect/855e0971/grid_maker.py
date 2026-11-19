"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ----
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


# the only discrete structural variant: the whole picture is transposed or not
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    # dotc is the one color role the rule depends on (the marks that grow into
    # frontiers).  Bar colors are pure decoration and stay free, as in the
    # original generator.
    dotc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"dotc": dotc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             dotc: int, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])

    # limits expressed in the CANONICAL frame (bars are horizontal there);
    # if the instance is mirrored the final grid is the transpose, so swap.
    lim_h = max_w if mirrored else max_h
    lim_w = max_h if mirrored else max_w

    cols = list(range(10))
    nbarsd = _unifint(diff_lb, diff_ub, (1, 4))
    nbars = random.choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    nbars = max(3, min(nbars, max(3, lim_h // 2)))

    h = _unifint(diff_lb, diff_ub, (2 * nbars, max(2 * nbars, lim_h)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, lim_w)))

    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[random.randint(0, nbars - 1)] += 1

    remcols = [c for c in cols if c != dotc]
    lastcol = -1
    nloclbs = [random.choice((0, 1)) for _ in range(nbars)]
    if sum(nloclbs) < 2:
        i1, i2 = random.sample(range(nbars), 2)
        nloclbs[i1] = 1
        nloclbs[i2] = 1

    gi, go = [], []
    for bs, nloclb in zip(barsizes, nloclbs):
        col = random.choice([c for c in remcols if c != lastcol])
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = _unifint(diff_lb, diff_ub, (nloclb, max(nloclb, w // 2)))
        for jj in random.sample(range(w), nl):
            rr = random.randint(0, bs - 1)
            gim[rr][jj] = dotc
            for r2 in range(bs):          # vfrontier inside the bar
                gom[r2][jj] = dotc
        lastcol = col
        gi += gim
        go += gom

    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


# ---------------------------------------------------------- derivation -------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- orientation ---------------------------------------------------------
    # In the canonical picture every row lies inside ONE bar, so a row holds at
    # most {bar colour, dot colour} = 2 colours.  If some row shows 3+ colours
    # the picture is the transposed one: the bars run vertically.
    mirrored = any(len(set(I[r].tolist())) >= 3 for r in range(H))

    def emit_transpose(h, w):
        """Perform the reflection across the main diagonal on the whole grid:
        rot90 CCW followed by an up/down flip IS the transpose.  Rotation needs
        a square canvas, so widen to sq x sq first and cut back after."""
        s = max(h, w)
        if h != w:
            # full-rectangle selections: whole canvas, background included
            ops.append(33); sels.append([0, 0, s - 1, s - 1])   # square canvas
        ops.append(24); sels.append([0, 0, s - 1, s - 1])       # rotate CCW
        ops.append(27); sels.append([0, 0, s - 1, s - 1])       # flip up/down
        if h != w:
            ops.append(33); sels.append([0, 0, w - 1, h - 1])   # back to w x h

    if mirrored:
        emit_transpose(H, W)
        C = I.T.copy()          # working grid now literally equals C
    else:
        C = I.copy()
    ch, cw = C.shape

    # --- which colour are the marks? ----------------------------------------
    # The dot colour is the colour whose removal leaves every row single
    # coloured (every row is one bar colour plus scattered dots).
    cands = []
    for c in sorted(set(C.flatten().tolist())):
        if all(len(set(C[r].tolist()) - {c}) == 1 for r in range(ch)):
            cands.append(c)
    if cands:
        dotc = min(cands, key=lambda c: int((C == c).sum()))
    else:
        vals, cnts = np.unique(C, return_counts=True)
        dotc = int(vals[int(np.argmin(cnts))])

    # --- the bars ------------------------------------------------------------
    rowcol = []
    for r in range(ch):
        rest = set(C[r].tolist()) - {dotc}
        rowcol.append(rest.pop() if rest else dotc)
    bars, start = [], 0
    for r in range(1, ch + 1):
        if r == ch or rowcol[r] != rowcol[r - 1]:
            bars.append((start, r - 1))
            start = r

    # --- grow each mark into the vertical frontier of its own bar ------------
    for (r0, r1) in bars:                       # bar by bar, top to bottom
        for c in range(cw):                     # mark by mark, left to right
            if not any(C[r, c] == dotc for r in range(r0, r1 + 1)):
                continue
            line = [(r, c) for r in range(r0, r1 + 1)]
            if all(C[r, c] == dotc for r, c in line):
                continue                        # already the whole frontier
            ops.append(int(dotc)); sels.append(sel_of(line))
            for r, cc in line:
                C[r, cc] = dotc

    # --- put the picture back the way it was ---------------------------------
    if mirrored:
        emit_transpose(ch, cw)

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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
