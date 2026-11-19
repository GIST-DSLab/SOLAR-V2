"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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

H_OUT = 10                      # the rule always answers with a ten-row grid

# the only discrete structural variant: which way the repetition drifts
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 9)
    ccols = random.sample(remcols, ncols)          # palette of the motif, fixed per episode

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


# ----------------------------------------------------------------------------- generator
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(int(a + (b - a) * diff_lb), int(a + (b - a) * diff_ub))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, ccols=None, mirrored=None) -> dict:
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
    if mirrored is None:
        mirrored = random.choice([True, False])
    w_ub = max(8, min(30, int(max_w)))
    h_ub = max(2, min(6, int(max_h)))

    while True:
        h = _unifint(diff_lb, diff_ub, (2, h_ub))
        w = _unifint(diff_lb, diff_ub, (8, w_ub))
        oh = _unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
        ow = _unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = _unifint(diff_lb, diff_ub, (1, oh * ow))
        picked = random.sample(bounds, ncells)
        obj = [(random.choice(ccols), ij) for ij in picked]
        mi = min(ij[0] for _, ij in obj)
        mj = min(ij[1] for _, ij in obj)
        obj = [(col, (i - mi, j - mj)) for col, (i, j) in obj]          # normalize
        oh = max(i for _, (i, j) in obj) + 1
        ow = max(j for _, (i, j) in obj) + 1
        locj = random.randint(0, w // 2)
        hoffs = random.randint(0, ow // 2 + 1)

        go = [[bgc] * w for _ in range(H_OUT)]
        for k in range(H_OUT // oh + 1):
            for col, (i, j) in obj:
                rr, cc = i + k * oh, j + locj + k * hoffs
                if 0 <= rr < H_OUT and 0 <= cc < w:
                    go[rr][cc] = col
        if len({v for row in go[h:] for v in row}) > 1:
            break

    gi = [row[:] for row in go[:h]]
    if mirrored:
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]
    return {'input': tuple(tuple(r) for r in gi), 'output': tuple(tuple(r) for r in go)}


# ----------------------------------------------------------------------------- helpers
def _infer_bgc(I, examples):
    cnt = Counter(np.asarray(I, dtype=int).flatten().tolist())
    if examples:
        for pair in examples:
            if isinstance(pair, dict):
                grids = [pair.get('input'), pair.get('output')]
            else:
                grids = list(pair)
            for g in grids:
                if g is None:
                    continue
                cnt.update(np.asarray(g, dtype=int).flatten().tolist())
    return int(cnt.most_common(1)[0][0])


def _covers(motif, a, b, h, w, fg):
    """does repeating the top band with step (a, b) rebuild exactly what the input shows?"""
    recon = {}
    k = 0
    while k * a < h:
        for (r, c), v in motif.items():
            rr, cc = r + k * a, c + k * b
            if 0 <= rr < h and 0 <= cc < w:
                recon[(rr, cc)] = v
        k += 1
    return recon == fg


def _detect_period(fg, h, w):
    """the step (dr, dc) by which the picture repeats itself, read off the input alone"""
    best_key, best = None, None
    for a in range(1, h):
        motif = {p: v for p, v in fg.items() if p[0] < a}
        if not motif:
            continue
        for b in range(-(w - 1), w):
            ov, ok = 0, True
            for (r, c), v in fg.items():
                p = (r + a, c + b)
                if p in fg:
                    if fg[p] != v:
                        ok = False
                        break
                    ov += 1
            if not ok or ov == 0:
                continue
            key = (ov, 1 if _covers(motif, a, b, h, w, fg) else 0, a * b, -a)
            if best_key is None or key > best_key:
                best_key, best = key, (a, b)
    return best


def _bands(motif, dr, dc, h, w):
    """copy k of the motif, keeping only the cells that are genuinely new (below the input)"""
    out = {}
    for k in range(1, (H_OUT - 1) // dr + 1):
        cells = {}
        for (r, c), v in motif.items():
            rr, cc = r + k * dr, c + k * dc
            if h <= rr < H_OUT and 0 <= cc < w:
                cells[(rr, cc)] = v
        out[k] = cells
    return out


def _grouped(cells, leftfirst):
    """cells of one copy, split per colour, taken in the order the drift runs"""
    groups = {}
    for p, v in cells.items():
        groups.setdefault(v, []).append(p)
    order = sorted(groups,
                   key=lambda v: (min(c for _, c in groups[v]), min(groups[v])),
                   reverse=not leftfirst)
    return [(v, sorted(groups[v])) for v in order]


# ----------------------------------------------------------------------------- solver
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = _infer_bgc(I, examples)
    fg = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if int(I[r, c]) != bgc}

    ops, sels = [], []

    # 1. the answer is always ten rows tall -> grow the canvas (exact full rectangle)
    ops.append(33)
    sels.append([0, 0, H_OUT - 1, w - 1])

    per = _detect_period(fg, h, w) if fg else None
    if per is None:
        if bgc != 0:
            ops.append(int(bgc))
            sels.append([h, 0, H_OUT - 1 - h, w - 1])       # exact full rectangle
        ops.append(34)
        sels.append([0, 0, H_OUT - 1, w - 1])
        return ops, sels

    dr, dc = per
    motif = {p: v for p, v in fg.items() if p[0] < dr}
    bands = _bands(motif, dr, dc, h, w)
    draw = {k: cs for k, cs in bands.items() if cs}
    fill_bands = sorted({r // dr for r in range(h, H_OUT)})
    rr_ = [p[0] for p in motif]
    cc_ = [p[1] for p in motif]
    r0, r1, c0, c1 = min(rr_), max(rr_), min(cc_), max(cc_)

    # --- how to go about it: decided by the picture itself, never by the answer ---
    if dc == 0:
        route = "stripes"                       # the copies stack into plain vertical runs
    else:
        stamp_ok = bool(draw) and all(
            h <= r0 + k * dr and r1 + k * dr < H_OUT
            and 0 <= c0 + k * dc and c1 + k * dc < w
            for k in draw
        )
        # a busy motif is worth copying once; a plain one is quicker drawn
        route = "stamp" if (stamp_ok and len(set(motif.values())) >= 3) else "bands"

    if route == "stripes":
        if bgc != 0:
            ops.append(int(bgc))
            sels.append([h, 0, H_OUT - 1 - h, w - 1])       # lay the background, draw on top
        allcells = {}
        for k in sorted(draw):
            allcells.update(draw[k])
        # start from the edge the motif already sits against
        leftfirst = (c0 + c1) / 2.0 <= (w - 1) / 2.0
        for v, cells in _grouped(allcells, leftfirst):
            ops.append(int(v))
            sels.append(sel_of(cells))
    else:
        downward = dc > 0                       # follow the way the repetition marches
        per_band_fill = (bgc != 0 and len(fill_bands) > 2)
        if bgc != 0 and not per_band_fill:
            ops.append(int(bgc))
            sels.append([h, 0, H_OUT - 1 - h, w - 1])       # exact full rectangle
        if route == "stamp":
            ops.append(28)                                   # the motif, read from the input
            sels.append([r0, c0, r1 - r0, c1 - c0])
        for k in sorted(fill_bands, reverse=not downward):
            if per_band_fill:
                lo, hi = max(k * dr, h), min(k * dr + dr - 1, H_OUT - 1)
                if lo <= hi:
                    ops.append(int(bgc))
                    sels.append([lo, 0, hi - lo, w - 1])     # this copy's strip, full width
            cells = draw.get(k)
            if not cells:
                continue
            if route == "stamp":
                ops.append(30)
                sels.append([r0 + k * dr, c0 + k * dc, 0, 0])
                # a paste carries no 0s: put those cells of the motif down by hand
                zeroes = sorted(p for p, v in cells.items() if v == 0)
                if zeroes:
                    ops.append(0)
                    sels.append(sel_of(zeroes))
            else:
                for v, grp in _grouped(cells, downward):
                    ops.append(int(v))
                    sels.append(sel_of(grp))

    ops.append(34)
    sels.append([0, 0, H_OUT - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
