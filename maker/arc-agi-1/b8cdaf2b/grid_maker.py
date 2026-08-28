"""
ARC Task: b8cdaf2b (RE-ARC) — LLM-generated grid_maker
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
    # The four input orientations (rot 0/1/2/3) are discrete structural variants:
    # plan them up front so every case an episode can test has been shown.
    VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def rotate(g, k):
        for _ in range(k % 4):
            g = [list(row) for row in zip(*g[::-1])]  # CW
        return g

    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    # rot 1/3 transpose the grid -> swap the dimension budgets so the final
    # instance still fits inside (max_h, max_w)
    hb = max_w if rot % 2 == 1 else max_h
    wb = max_h if rot % 2 == 1 else max_w
    hb = max(3, min(30, hb))
    wb = max(3, min(30, wb))

    h = unifint(diff_lb, diff_ub, (3, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))

    winv = unifint(diff_lb, diff_ub, (2, w - 1))
    w2 = w - winv
    w2 = min(max(w2, 1), w - 2)
    locj = random.randint(1, w - w2 - 1)

    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = linc
    for j in range(locj, locj + w2):
        gi[0][j] = dotc
        gi[1][j] = linc

    go = [row[:] for row in gi]
    r, c = 2, locj - 1
    while 0 <= r < h and 0 <= c < w:
        go[r][c] = dotc
        r += 1
        c -= 1
    r, c = 2, locj + w2
    while 0 <= r < h and 0 <= c < w:
        go[r][c] = dotc
        r += 1
        c += 1

    gi = rotate(gi, rot)
    go = rotate(go, rot)
    return {"input": [list(map(int, row)) for row in gi],
            "output": [list(map(int, row)) for row in go]}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # dotc = least frequent color (bar segment); guaranteed unique minimum here
    cnt = Counter(I.flatten().tolist())
    dotc = int(min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0])

    pts = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]
    rs = sorted({r for r, _ in pts})
    cs = sorted({c for _, c in pts})

    # The bar lies on one edge line, never touching a corner -> orientation is unambiguous.
    if len(cs) == 1 and cs[0] in (0, w - 1) and rs[0] >= 1 and rs[-1] <= h - 2:
        axis = 'col'          # edge line is a column; rays mirror across a horizontal axis
        L, D, Lat = cs[0], w, h
        a, b = rs[0], rs[-1]
    else:
        axis = 'row'          # edge line is a row; rays mirror across a vertical axis
        L, D, Lat = rs[0], h, w
        a, b = cs[0], cs[-1]

    sign = 1 if L == 0 else -1

    def pos(d, lat):
        return (d, lat) if axis == 'row' else (lat, d)

    def lat_of(cell):
        return cell[1] if axis == 'row' else cell[0]

    def dep_of(cell):
        return cell[0] if axis == 'row' else cell[1]

    def ray(lat_start, step):
        out, k = [], 0
        while True:
            d = L + sign * (2 + k)
            lat = lat_start + step * k
            if not (0 <= d < D) or not (0 <= lat < Lat):
                break
            out.append(pos(d, lat))
            k += 1
        return out

    ray_a = ray(a - 1, -1)          # diagonal leaving the low end of the bar
    ray_b = ray(b + 1, +1)          # diagonal leaving the high end of the bar

    # The two diagonals are exact mirror images across the bar's centre line:
    # lateral x  <->  S - x
    S = a + b
    lat0 = max(0, S - (Lat - 1))
    lat1 = S - lat0                 # symmetric lateral span that fits on the grid
    if sign == 1:
        d0, d1 = L + 2, D - 1       # interior, excluding the edge line and its inner bar
    else:
        d0, d1 = 0, L - 2

    if axis == 'row':
        # full rectangle (background included): rows d0..d1, cols lat0..lat1
        rect = [d0, lat0, d1 - d0, lat1 - lat0]
        flip_op = 26                # FlipH: mirrors columns about (lat0+lat1)/2 = S/2
    else:
        # full rectangle (background included): rows lat0..lat1, cols d0..d1
        rect = [lat0, d0, lat1 - lat0, d1 - d0]
        flip_op = 27                # FlipV: mirrors rows about (lat0+lat1)/2 = S/2

    # draw the longer diagonal first, then let the reflection produce its twin
    src, dst = (ray_a, ray_b) if len(ray_a) >= len(ray_b) else (ray_b, ray_a)

    mirrored = set()
    for cell in src:
        lt = lat_of(cell)
        if lat0 <= lt <= lat1:
            mirrored.add(pos(dep_of(cell), S - lt))

    ops, sels = [], []

    # 1. draw one diagonal ray out of the bar's end
    ops.append(dotc)
    sels.append(sel_of(src))

    # 2. reflect the interior region: the ray becomes the ray on the other side
    ops.append(flip_op)
    sels.append(rect)

    # 3. re-draw the original ray (the reflection carried it over to the far side)
    ops.append(dotc)
    sels.append(sel_of(src))

    # 4. whatever the reflection could not reach (the mirror ran off the grid edge)
    rest = [c for c in dst if c not in mirrored]
    if rest:
        ops.append(dotc)
        sels.append(sel_of(rest))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b8cdaf2b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8cdaf2b"
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
                                f"for task b8cdaf2b"
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
                    f"Failed to build a complete episode for task b8cdaf2b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8cdaf2b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
