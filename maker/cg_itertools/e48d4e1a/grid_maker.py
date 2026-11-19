"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # frontier colour kept non-zero so the cross can be carried by ARCLE Move
    # (Move/object mode ignores 0-valued cells)
    fgc = random.choice([c for c in cols if c != bgc and c != 0])
    dotc = random.choice([c for c in cols if c not in (bgc, fgc)])
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or fgc is None or dotc is None:
        bgc, fgc, dotc = sample(cols, 3)
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = sample(totuple(inds), nn)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # The cross: one full row and one full column of the same colour.
    hrow, fgc = None, None
    for r in range(hi):
        if np.all(I[r, :] == I[r, 0]):
            hrow, fgc = r, int(I[r, 0])
            break
    vcol = None
    for c in range(wi):
        if np.all(I[:, c] == fgc):
            vcol = c
            break

    # The other two colours: background is the more frequent one; the rest are dots.
    cnt = Counter(int(v) for v in I.flatten() if v != fgc)
    others = [c for c, _ in cnt.most_common()]
    bgc = others[0]
    dotc = others[1]
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    n = len(dots)

    # The cross moves n cells directly away from the dots.
    mr = sum(r for r, _ in dots) / n
    mc = sum(c for _, c in dots) / n
    di = sign(hrow - mr)
    dj = sign(vcol - mc)

    ops, sels = [], []

    # 1. The dots get used up: erase them to the background.
    ops.append(bgc); sels.append(sel_of(dots))

    new_vcol = vcol + dj * n
    new_hrow = hrow + di * n

    if fgc == 0:
        # A 0-coloured line cannot be grabbed by Move (object mode drops 0s): repaint.
        ops.append(bgc); sels.append(sel_of([(hrow, c) for c in range(wi)]))
        ops.append(bgc); sels.append(sel_of([(r, vcol) for r in range(hi)]))
        ops.append(fgc); sels.append(sel_of([(new_hrow, c) for c in range(wi)]))
        ops.append(fgc); sels.append(sel_of([(r, new_vcol) for r in range(hi)]))
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    def slide(cells, dr, dc, steps):
        mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        cur = list(cells)
        ops.append(mop); sels.append(sel_of(cur))           # grab
        cur = [(r + dr, c + dc) for r, c in cur]
        for _ in range(steps - 1):
            ops.append(mop); sels.append(sel_of([]))        # continue same grab
            cur = [(r + dr, c + dc) for r, c in cur]
        hole = sorted(set(cells) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    # 2. Slide the vertical line (its arms, i.e. the column minus the crossing cell,
    #    so the horizontal line stays intact in the snapshot) sideways by n.
    if dj != 0 and n > 0:
        arms = [(r, vcol) for r in range(hi) if r != hrow]
        slide(arms, 0, dj, n)

    # 3. Slide the horizontal line (its arms, i.e. the row minus the cell where the
    #    vertical line now crosses it) up/down by n.
    if di != 0 and n > 0:
        arms = [(hrow, c) for c in range(wi) if c != new_vcol]
        slide(arms, di, 0, n)

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
