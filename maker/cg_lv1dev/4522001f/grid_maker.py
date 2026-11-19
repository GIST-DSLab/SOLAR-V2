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


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, rot=None) -> dict:
    def unifint(lb, ub):
        lo = int(round(lb + (ub - lb) * diff_lb))
        hi_ = int(round(lb + (ub - lb) * diff_ub))
        return random.randint(min(lo, hi_), max(lo, hi_))
    if rot is None:
        rot = random.choice(ROTS)
    ub_h = max(3, min(10, max_h // 3))
    ub_w = max(3, min(10, max_w // 3))
    h = unifint(3, ub_h)
    w = unifint(3, ub_w)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((3 * h, 3 * w), bgc, dtype=int)

    def place(i, j):
        gi[i, j] = sqc; gi[i, j + 1] = sqc; gi[i + 1, j] = sqc; gi[i + 1, j + 1] = dotc
        go[i:i + 4, j:j + 4] = sqc
        go[i + 4:i + 8, j + 4:j + 8] = sqc

    def dn(cells):
        s = set()
        for r, c in cells:
            s |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return s

    loci = random.randint(0, min(h - 2, 3 * h - 8))
    locj = random.randint(0, min(w - 2, 3 * w - 8))
    place(loci, locj)
    seed = {(loci + a, locj + b) for a in (0, 1) for b in (0, 1)}
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc} - dn(seed)
    noccs = unifint(0, (h * w) // 9)
    succ = tr = 0
    while tr < 10 * noccs and succ < noccs:
        tr += 1
        cands = [p for p in iinds if p[0] <= h - 2 and p[1] <= w - 2]
        if not cands:
            break
        i, j = random.choice(sorted(cands))
        cells = {(i + a, j + b) for a in (0, 1) for b in (0, 1)}
        if cells <= iinds:
            succ += 1
            iinds = (iinds - cells) - dn(cells)
            place(i, j)
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading (the golf entry's): the output is ONE canonical picture -- every seed with its
    dot at bottom-right, the picture anchored at the canvas top-left, each seed growing two
    4x4 squares down-right -- seen through a mirror chosen by the dot's corner.
    This differs from the per-seed / shifted-anchor reading, so the ops are:
      mirror the input into canonical orientation -> grow canvas to 3h x 3w ->
      background on the added area -> paint each seed's two 4x4 squares ->
      mirror the whole canvas back.
    Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                  # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None     # 1 cell per seed

    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    drs, dcs = [], []
    for rd, cd in dots:
        drs.append(1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0)
        dcs.append(1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0)
    DR = Counter(drs).most_common(1)[0][0] if drs else 1
    DC = Counter(dcs).most_common(1)[0][0] if dcs else 1

    ops, sels = [], []
    G = I.copy()
    whole_in = [0, 0, hi - 1, wi - 1]        # whole input grid: a full rectangle
    # 1. mirror the input so every dot sits at the bottom-right of its seed
    if DR == 0:
        ops.append(27); sels.append(whole_in); G = G[::-1, :].copy()
    if DC == 0:
        ops.append(26); sels.append(whole_in); G = G[:, ::-1].copy()

    # 2. grow the canvas to 3h x 3w (picture anchored top-left in canonical frame)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    C = np.zeros((ho, wo), dtype=int)
    C[:hi, :wi] = G

    # 3. the rows / columns the resize added become background (full rectangles)
    if bgc != 0:
        ops.append(bgc); sels.append([hi, 0, ho - 1 - hi, wo - 1]); C[hi:, :] = bgc
        ops.append(bgc); sels.append([0, wi, hi - 1, wo - 1 - wi]); C[:hi, wi:] = bgc

    # 4. each canonical seed (top-left (i,j), dot at (i+1,j+1)) grows two 4x4 squares
    seeds = [(r - 1, c - 1) for r in range(hi) for c in range(wi)
             if G[r, c] == dotc and r >= 1 and c >= 1
             and G[r - 1, c] == sqc and G[r, c - 1] == sqc]
    for i, j in seeds:
        for k in (0, 1):
            r0, c0 = i + 4 * k, j + 4 * k
            r1, c1 = min(r0 + 3, ho - 1), min(c0 + 3, wo - 1)
            if r0 > r1 or c0 > c1:
                continue
            if np.all(C[r0:r1 + 1, c0:c1 + 1] == sqc):
                continue                      # already that colour: op would be a no-op
            # the 4x4 square is exactly a full rectangle
            ops.append(sqc); sels.append([r0, c0, r1 - r0, c1 - c0])
            C[r0:r1 + 1, c0:c1 + 1] = sqc

    # 5. mirror the whole canvas back to the dot's orientation (full rectangle)
    whole_out = [0, 0, ho - 1, wo - 1]
    if DR == 0:
        ops.append(27); sels.append(whole_out)
    if DC == 0:
        ops.append(26); sels.append(whole_out)

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
