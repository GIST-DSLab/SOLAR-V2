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
import numpy as np
from maker.sel_helpers import sel_of

# ---------------------------------------------------------------------------
# Task e26a3af2 : the grid is a stack of noisy stripes (bands of rows, or -
# after the generator's dmirror - bands of columns).  Every stripe collapses
# to its own solid colour: one clean line per stripe, replicated across the
# whole stripe.  The verifier's rot90 calls are only orientation bookkeeping
# (turning a 1xN row of colours into an Nx1 column); the transformation the
# grid actually undergoes is the replication (repeat / hupscale / vupscale),
# performed here with CopyO + Paste.
# ---------------------------------------------------------------------------

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    # Stripe colours are sampled per instance and the rule does not depend on
    # which colours they are, so no colour role has to be pinned.  The one
    # discrete structural variant is the orientation (horizontal bands vs
    # vertical bands), which must be covered by the examples.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             transpose=None) -> dict:
    if transpose is None:
        transpose = choice((True, False))

    # the grid is built with horizontal bands and dmirrored afterwards,
    # so the pre-mirror limits swap when transpose is on
    if transpose:
        limH, limW = max_w, max_h
    else:
        limH, limW = max_h, max_w
    limH = max(2, min(30, limH))
    limW = max(4, min(30, limW))

    cols = interval(0, 10, 1)
    nr = unifint(diff_lb, diff_ub, (1, max(1, min(10, limH // 2))))
    w = unifint(diff_lb, diff_ub, (4, limW))
    scols = sample(cols, nr)
    sgs = [canvas(col, (2, w)) for col in scols]

    numexp_ub = max(0, min(30 - nr, limH - 2 * nr))
    numexp = unifint(diff_lb, diff_ub, (0, numexp_ub))
    for k in range(numexp):
        idx = randint(0, nr - 1)
        sgs[idx] = sgs[idx] + sgs[idx][-1:]

    sgs2 = []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = shape(sg)
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
        inds = totuple(asindices(sg))
        noise = sample(inds, nnoise)
        oc = remove(col, cols)
        noise = frozenset({(choice(oc), ij) for ij in noise})
        sg2 = paint(sg, noise)
        for idxx in [0, -1]:
            while sum([e == col for e in sg2[idxx]]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                ch = choice(locs)
                if idxx == 0:
                    sg2 = (sg2[0][:ch] + (col,) + sg2[0][ch + 1:],) + sg2[1:]
                else:
                    sg2 = sg2[:-1] + (sg2[-1][:ch] + (col,) + sg2[-1][ch + 1:],)
        sgs2.append(sg2)

    gi = tuple(row for sg in sgs2 for row in sg)
    go = tuple(row for sg in sgs for row in sg)
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # orientation: bands of rows -> every row of O is uniform,
    #              bands of cols -> every column of O is uniform
    horizontal = all(len(set(O[r].tolist())) == 1 for r in range(h))

    n_band = h if horizontal else w      # index along the stacking axis
    n_rep = w if horizontal else h       # index along the replication axis

    def rc(a, b):
        """(band index, repetition index) -> real (row, col)"""
        return (a, b) if horizontal else (b, a)

    # stripe colours, grouped into bands of consecutive equal-coloured lines
    line_col = [int(O[rc(a, 0)]) for a in range(n_band)]
    bands = []
    a = 0
    while a < n_band:
        a2 = a
        while a2 + 1 < n_band and line_col[a2 + 1] == line_col[a]:
            a2 += 1
        bands.append((a, a2, line_col[a]))
        a = a2 + 1

    sim = I.copy()
    ops, sels = [], []

    # 1. Stripes whose colour is 0 cannot be replicated by Paste (Paste never
    #    writes 0 cells), so they are filled outright, whole stripe at once.
    for (a0, a1, c) in bands:
        if c != 0:
            continue
        region = [rc(x, b) for x in range(a0, a1 + 1) for b in range(n_rep)]
        if any(sim[p] != 0 for p in region):
            ops.append(0)
            sels.append(sel_of(region))
            for p in region:
                sim[p] = 0

    # 2. Build the single clean line (first column / first row): each stripe's
    #    segment of it painted with that stripe's colour.
    for (a0, a1, c) in bands:
        if c == 0:
            continue
        seg = [rc(x, 0) for x in range(a0, a1 + 1)]
        if any(sim[p] != c for p in seg):
            ops.append(int(c))
            sels.append(sel_of(seg))
            for p in seg:
                sim[p] = c

    # 3. Replicate that line across the whole grid: copy it once, paste it at
    #    every other position along the repetition axis.
    clip = [int(sim[rc(x, 0)]) for x in range(n_band)]
    strip = [rc(x, 0) for x in range(n_band)]
    targets = []
    probe = list(clip)
    for b in range(1, n_rep):
        if any(probe[x] != 0 and sim[rc(x, b)] != probe[x] for x in range(n_band)):
            targets.append(b)
    if targets:
        ops.append(29)                     # CopyO: the clean line
        sels.append(sel_of(strip))
        for b in targets:
            ops.append(30)                 # Paste it at this line's origin
            sels.append(sel_of([rc(0, b)]))
            for x in range(n_band):
                if clip[x] != 0:
                    sim[rc(x, b)] = clip[x]

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])      # bbox == whole grid, intended
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
