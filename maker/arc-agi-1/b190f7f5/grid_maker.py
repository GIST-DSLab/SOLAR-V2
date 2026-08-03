"""
ARC Task: b190f7f5 (RE-ARC) — LLM-generated grid_maker
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
import math
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

VARIANTS = [{"direction": "vertical"}, {"direction": "horizontal"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test drawn from shown variants
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, direction=None) -> dict:
    if direction is None:
        direction = random.choice(["vertical", "horizontal"])

    cols = [c for c in range(10) if c != bgc]

    # half dims (h, w): output is (h^2, w^2); input is (2h, w) or (h, 2w)
    hmax = min(5, int(math.isqrt(max_h)))
    wmax = min(5, int(math.isqrt(max_w)))
    if direction == "vertical":
        hmax = min(hmax, max_h // 2)
        wmax = min(wmax, max_w)
    else:
        hmax = min(hmax, max_h)
        wmax = min(wmax, max_w // 2)
    hmax = max(2, hmax)
    wmax = max(2, wmax)

    h, w = 2, 2
    for _ in range(200):
        ch = random.randint(2, hmax)
        cw = random.randint(2, wmax)
        # enforce is_portrait consistency: vertical -> tall I, horizontal -> wide I
        if direction == "vertical" and 2 * ch > cw:
            h, w = ch, cw
            break
        if direction == "horizontal" and 2 * cw > ch:
            h, w = ch, cw
            break

    inds = [(i, j) for i in range(h) for j in range(w)]

    numcd = random.randint(0, (h * w) // 2)
    numc = random.choice([numcd, h * w - numcd])
    numc = min(max(1, numc), h * w - 1)

    numcd2 = random.randint(0, (h * w) // 2)
    numc2 = random.choice([numcd2, h * w - numcd2])
    numc2 = min(max(2, numc2), h * w - 1)

    srclocs = random.sample(inds, numc)
    srccol = random.choice(cols)
    remcols = [x for x in cols if x != srccol]

    numcols = random.randint(2, min(8, len(remcols)))
    trglocs = random.sample(inds, numc2)
    ccols = random.sample(remcols, numcols)
    fixc1 = random.choice(ccols)
    trgobj = {}
    trgobj[trglocs[0]] = fixc1
    trgobj[trglocs[1]] = random.choice([x for x in ccols if x != fixc1])
    for ij in trglocs[2:]:
        trgobj[ij] = random.choice(ccols)

    # source half: bgc canvas with srccol at srclocs
    gisrc = [[bgc] * w for _ in range(h)]
    for (i, j) in srclocs:
        gisrc[i][j] = srccol
    # target half: bgc canvas with colored cells
    gitrg = [[bgc] * w for _ in range(h)]
    for (i, j), col in trgobj.items():
        gitrg[i][j] = col

    if direction == "vertical":
        top, bot = random.choice([[gisrc, gitrg], [gitrg, gisrc]])
        gi = [row[:] for row in top] + [row[:] for row in bot]
    else:
        left, right = random.choice([[gisrc, gitrg], [gitrg, gisrc]])
        gi = [lr[:] + rr[:] for lr, rr in zip(left, right)]

    H, W = h * h, w * w
    go = [[bgc] * W for _ in range(H)]
    for (i, j) in trglocs:
        col = gitrg[i][j]
        for (sr, sc) in srclocs:
            go[i * h + sr][j * w + sc] = col

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # 1) split axis from I: portrait (tall) -> vsplit, else hsplit
    if hi > wi:
        halves = [I[:hi // 2, :], I[hi // 2:, :]]
    else:
        halves = [I[:, :wi // 2], I[:, wi // 2:]]

    def ncolors(g):
        return len(set(g.flatten().tolist()))

    # 2) source = fewer colors, target = more colors
    if ncolors(halves[0]) <= ncolors(halves[1]):
        source, target = halves[0], halves[1]
    else:
        source, target = halves[1], halves[0]

    # 3) common color (palette intersection) = background
    ps = set(source.flatten().tolist())
    pt = set(target.flatten().tolist())
    common = ps & pt
    bgc = int(sorted(common)[0])

    hs, ws = source.shape
    Hs, Ws = hs * hs, ws * ws

    # source stamp = non-bgc cells of source (its own coords)
    stamp = [(r, c) for r in range(hs) for c in range(ws) if int(source[r, c]) != bgc]

    # resize canvas to output size (transparent copy leaves input junk at top-left)
    ops.append(33); sels.append([0, 0, Hs - 1, Ws - 1])
    # lay uniform background base (clears leftover input, sets bgc everywhere)
    ops.append(bgc); sels.append([0, 0, Hs - 1, Ws - 1])

    # for each non-bgc target cell, stamp source pattern into its block, colored by that cell
    for r in range(hs):
        for c in range(ws):
            col = int(target[r, c])
            if col == bgc:
                continue
            base_r, base_c = r * hs, c * ws
            cells = [(base_r + sr, base_c + sc) for (sr, sc) in stamp]
            if cells:
                ops.append(col); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, Hs - 1, Ws - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b190f7f5"
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
                                f"for task b190f7f5"
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
                    f"Failed to build a complete episode for task b190f7f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b190f7f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
