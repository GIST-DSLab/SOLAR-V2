"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ca, cb = sample(cols, 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, ca: int, cb: int) -> dict:
    hcap = min(8, max_h // 3)
    wcap = min(8, max_w // 3)
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, hcap))
        wi = unifint(diff_lb, diff_ub, (2, wcap))
        h = unifint(diff_lb, diff_ub, (3 * hi, max_h))
        w = unifint(diff_lb, diff_ub, (3 * wi, max_w))
        if w < hi:
            continue
        c = canvas(-1, (hi, hi))
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, frozenset(A))
                bcands = set(reminds) & mapply(dneighbors, frozenset(B))
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(frozenset(A)) * width(frozenset(A))
            condb = len(B) == height(frozenset(B)) * width(frozenset(B))
        A0 = frozenset(A)
        B0 = frozenset(B)
        go = fill(c, ca, A0)
        go = fill(go, cb, B0)
        ula = ulcorner(A0)
        ulb = ulcorner(B0)
        fullocs = totuple(asindices(gi))
        An = normalize(A0)
        Bn = normalize(B0)
        ha, wa = shape(An)
        hb, wb = shape(Bn)

        def square_ok(sq, foreign):
            r, cc = sq
            if r < 0 or cc < 0 or r + hi > h or cc + hi > w:
                return False
            for (fr, fc) in foreign:
                if r <= fr < r + hi and cc <= fc < cc + hi:
                    return False
            return True

        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locaj = randint(0, w - wa)
                if locaj > h - ha:
                    continue
                plcda = shift(An, (locaj, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(totuple(remlocs2))
                plcdb = shift(Bn, ch)
                if not set(plcdb).issubset(set(remlocs2)):
                    continue
                sqa = (locaj - ula[0], locaj - ula[1])
                sqb = (ch[0] - ulb[0], ch[1] - ulb[1])
                if square_ok(sqa, plcdb) or square_ok(sqb, plcda):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Background: the canvas colour the generator paints before dropping the two pieces.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    piece_cols = [int(v) for v in np.unique(I) if int(v) != bgc]

    # Each piece is one solid-coloured jigsaw part. In O they interlock into a square.
    # Find the piece that already sits at its final place: translating O's cells of that
    # colour by a single (R,C) must land exactly on that piece in I, the square must lie
    # inside the canvas, and the other piece must not intrude on it.
    anchor = None
    for col in piece_cols:
        other = [v for v in piece_cols if v != col][0]
        src = np.argwhere(I == col)
        tgt = np.argwhere(O == col)
        if len(tgt) == 0:
            continue
        R = int(src[:, 0].min() - tgt[:, 0].min())
        C = int(src[:, 1].min() - tgt[:, 1].min())
        if set(map(tuple, (tgt + np.array([R, C])).tolist())) != set(map(tuple, src.tolist())):
            continue
        if R < 0 or C < 0 or R + ho > hi or C + wo > wi:
            continue
        foreign = np.argwhere(I == other)
        if any(R <= r < R + ho and C <= c < C + wo for r, c in foreign.tolist()):
            continue
        anchor = (col, other, R, C)
        break

    col, other, R, C = anchor

    ops, sels = [], []

    # 1. Keep only the square the anchor piece belongs to; the rest of the canvas is scenery.
    ops.append(33)
    sels.append([R, C, ho - 1, wo - 1])

    # 2. Inside that square the anchor piece leaves one connected hole, exactly the shape of
    #    the second piece. Fill that hole with the second piece's colour in one flood fill.
    seed = None
    for r in range(ho):
        for c in range(wo):
            if I[R + r, C + c] == bgc:
                seed = (r, c)
                break
        if seed is not None:
            break
    ops.append(10 + int(other))
    sels.append([seed[0], seed[1], 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
