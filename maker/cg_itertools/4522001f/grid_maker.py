"""
ARC Task: 4522001f (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": k} for k in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, dotc=2, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # pre-rotation bounds so the rotated 3h x 3w output fits max_h x max_w
    hb = (max_h if rot % 2 == 0 else max_w) // 3
    wb = (max_w if rot % 2 == 0 else max_h) // 3
    h = _unif(diff_lb, diff_ub, (3, max(3, min(10, hb))))
    w = _unif(diff_lb, diff_ub, (3, max(3, min(10, wb))))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((3 * h, 3 * w), bgc, dtype=int)

    def place(i, j):
        gi[i, j] = gi[i, j + 1] = gi[i + 1, j] = sqc
        gi[i + 1, j + 1] = dotc
        for k in (0, 4):
            go[i + k:i + k + 4, j + k:j + k + 4] = sqc

    def nb(cells):
        s = set()
        for r, c in cells:
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                s.add((r + dr, c + dc))
        return s

    loci = random.randint(0, min(h - 2, 3 * h - 8))
    locj = random.randint(0, min(w - 2, 3 * w - 8))
    place(loci, locj)
    blk = {(loci + a, locj + b) for a in (0, 1) for b in (0, 1)}
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc} - nb(blk)
    noccs = _unif(diff_lb, diff_ub, (0, (h * w) // 9))
    succ = tr = 0
    while tr < 10 * noccs and succ < noccs:
        tr += 1
        cands = [p for p in iinds if p[0] <= h - 2 and p[1] <= w - 2]
        if not cands:
            break
        i, j = random.choice(sorted(cands))
        b = {(i + a, j + c) for a in (0, 1) for c in (0, 1)}
        if b <= iinds:
            succ += 1
            iinds = (iinds - b) - nb(b)
            place(i, j)
    gi = np.rot90(gi, -rot)   # rot x 90deg clockwise
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading: there is ONE canonical picture -- each seed (3 square cells + a dot in the
    bottom-right corner of its 2x2 block) grows into a 4x4 square from the seed's top-left
    plus a second 4x4 square 4 cells further down-right, on a 3h x 3w canvas anchored at
    the top-left. Every other instance is that picture mirrored. So: mirror the input into
    canonical orientation, build the canonical picture, mirror the whole canvas back.
    Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                   # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None      # 1 cell per seed

    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    drs, dcs = [], []
    for rd, cd in dots:
        drs.append(1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0)
        dcs.append(1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0)
    DR = Counter(drs).most_common(1)[0][0] if drs else 1
    DC = Counter(dcs).most_common(1)[0][0] if dcs else 1

    ops, sels = [], []
    whole_in = [0, 0, hi - 1, wi - 1]          # whole input grid (full rectangle)
    G = I.copy()
    flips = []
    if DC == 0:                                 # dot on the left -> mirror left<->right
        ops.append(26); sels.append(whole_in); G = G[:, ::-1]; flips.append(26)
    if DR == 0:                                 # dot on top -> mirror up<->down
        ops.append(27); sels.append(whole_in); G = G[::-1, :]; flips.append(27)

    # grow canvas to 3h x 3w (canonical picture is anchored top-left)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    C = np.zeros((ho, wo), dtype=int)
    C[:hi, :wi] = G
    if bgc != 0:
        # the rows / columns the resize just added become background (full rectangles)
        ops.append(bgc); sels.append([hi, 0, ho - 1 - hi, wo - 1]); C[hi:, :] = bgc
        ops.append(bgc); sels.append([0, wi, hi - 1, wo - 1 - wi]); C[:hi, wi:] = bgc

    # canonical seeds: dot at the bottom-right of its 2x2 block
    seeds = [(r - 1, c - 1) for r in range(hi) for c in range(wi) if G[r, c] == dotc]
    for i, j in seeds:
        for k in (0, 4):                        # first square, then the one 4 further along
            cells = [(r, c) for r in range(i + k, i + k + 4) for c in range(j + k, j + k + 4)
                     if 0 <= r < ho and 0 <= c < wo]
            if any(C[r, c] != sqc for r, c in cells):
                ops.append(sqc); sels.append(sel_of(cells))
                for r, c in cells:
                    C[r, c] = sqc

    # mirror the whole canvas back to this instance's orientation (full rectangle)
    whole_out = [0, 0, ho - 1, wo - 1]
    for f in reversed(flips):
        ops.append(f); sels.append(whole_out)
    ops.append(34); sels.append(whole_out)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4522001f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4522001f"
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
                                f"for task 4522001f"
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
                    f"Failed to build a complete episode for task 4522001f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4522001f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
