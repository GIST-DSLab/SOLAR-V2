"""
ARC Task: 82819916 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

# The generator applies one of four whole-grid rotations at the end, which is the
# only DISCRETE structural variant: it decides whether the master line runs
# horizontally (identity / rot180) or vertically (rot90 / rot270), and on which
# side of each partial line the visible stub sits.  The rule itself is
# orientation-agnostic, but the trajectory is not: a vertical instance is solved
# by first mirroring the grid along its diagonal (exactly the RE-ARC verifier's
# `dmirror` branch), so both orientations must be covered by the examples.
_ROTS = ["identity", "rot90", "rot180", "rot270"]
_VERTICAL = ["rot90", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ass, bss = random.sample(rem, 2)          # the master line's two colors
    n_ex = num_examples if num_examples else 3

    if n_ex >= len(_ROTS):
        examples = [{"rot": r} for r in _ROTS]
        examples += [{"rot": random.choice(_ROTS)} for _ in range(n_ex - len(_ROTS))]
    elif n_ex >= 2:
        # always show both orientations of the master line
        examples = [{"rot": random.choice(_VERTICAL)},
                    {"rot": random.choice(["identity", "rot180"])}]
        examples += [{"rot": random.choice(_ROTS)} for _ in range(n_ex - 2)]
    else:
        examples = [{"rot": random.choice(_ROTS)} for _ in range(n_ex)]
    random.shuffle(examples)
    test = dict(random.choice(examples))      # test variant was shown
    return {"bgc": bgc, "ass": ass, "bss": bss, "instance_plan": examples + [test]}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ass, bss, rot=None) -> dict:
    if rot is None:
        rot = random.choice(_ROTS)
    swaps = rot in _VERTICAL                   # rot90/rot270 transpose the shape
    hb = max(5, min(30, max_w if swaps else max_h))
    wb = max(5, min(30, max_h if swaps else max_w))
    h = unifint(diff_lb, diff_ub, (5, hb))
    w = unifint(diff_lb, diff_ub, (5, wb))
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    itv = interval(0, w, 1)
    na = randint(2, w - 2)
    alocs = sample(itv, na)
    blocs = difference(itv, alocs)
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = randint(0, h - 1)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    numl = unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = remove(llocs, interval(0, h, 1))
    for k in range(numl):
        lloc = choice(remlocs)
        remlocs = remove(lloc, remlocs)
        a, b = sample(remcols, 2)
        gi = fill(gi, a, {(lloc, j) for j in alocs})
        gi = fill(gi, b, {(lloc, j) for j in blocs})
    cutoff = min(blocs) + 1
    go = tuple(e for e in gi)
    gi = fill(gi, bgc, backdrop(frozenset({(0, cutoff), (h - 1, w - 1)})))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    return {"input": rotf(gi), "output": rotf(go)}


def _diag_mirror(sq, ops, sels):
    """Mirror the whole square canvas along its main diagonal.

    transpose == rot90CCW(fliplr(.)); both ARCLE ops need a SQUARE selection.
    Every selection here is a full rectangle of the canvas -- the region really
    is the whole grid, background included -- so the bbox form is the honest one.
    """
    ops.append(26); sels.append([0, 0, sq - 1, sq - 1])   # FlipH  (left<->right)
    ops.append(24); sels.append([0, 0, sq - 1, sq - 1])   # Rotate90 CCW -> diagonal mirror


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Every cell that changes was canvas background in I, so the changed cells
    # name the background color directly.
    ch = np.argwhere(I != O)
    bgc = Counter(int(I[r, c]) for r, c in ch).most_common(1)[0][0]

    # The master line is the one full line of the grid that holds no background.
    # If it runs vertically, mirror the canvas along its diagonal so that every
    # line becomes a row; mirror back at the end.
    horizontal = any(not (I[r] == bgc).any() for r in range(hi))
    sq = max(hi, wi)
    if horizontal:
        G = I.copy()
    else:
        if hi != wi:
            # the mirror needs a square canvas to turn on
            ops.append(33); sels.append([0, 0, sq - 1, sq - 1])
        _diag_mirror(sq, ops, sels)
        G = I.T.copy()                      # canvas is now the diagonal mirror
    H, W = G.shape

    p = [r for r in range(H) if not (G[r] == bgc).any()][0]
    K = [int(v) for v in G[p]]
    classes = {}
    for j, c in enumerate(K):
        classes.setdefault(c, []).append(j)

    for q in range(H):
        if q == p:
            continue
        stub = [j for j in range(W) if int(G[q, j]) != bgc]
        if not stub:
            continue
        # the stub says which color this line uses for each class of the master
        order, mapping = [], {}
        for j in stub:
            kc = K[j]
            if kc not in mapping:
                order.append(kc)
                mapping[kc] = int(G[q, j])
        # continue each class of the master line across this whole line
        for kc in order:
            tgt = mapping[kc]
            cells = [(q, j) for j in classes[kc] if int(G[q, j]) != tgt]
            if not cells:
                continue
            ops.append(tgt)
            sels.append(sel_of(cells))
            for (r, c) in cells:
                G[r, c] = tgt

    if not horizontal:
        _diag_mirror(sq, ops, sels)         # mirror back (canvas is still square)
        if hi != wi:
            ops.append(33); sels.append([0, 0, ho - 1, wo - 1])   # shrink canvas back

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
                        f"num_examples+1 ({num_examples + 1}) for task 82819916"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 82819916"
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
                                f"for task 82819916"
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
                    f"Failed to build a complete episode for task 82819916 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"82819916-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
