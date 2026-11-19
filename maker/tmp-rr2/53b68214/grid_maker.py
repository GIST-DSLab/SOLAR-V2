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

H_OUT = 10


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(list(range(10)))}


def _shift_candidates(I):
    """Verifier's criterion: shifts (i in 1..5, j in -10..9) under which the
    pattern overlaps itself consistently; keep those with maximal overlap and,
    among them, maximal i*j.  Returns the list of tied best shifts."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    cells = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if I[r, c] != bgc}
    scored = []
    for i in range(1, 6):
        for j in range(-10, 10):
            inter = [(r, c) for (r, c) in cells if (r - i, c - j) in cells]
            if not inter:
                continue
            if any(cells[(r, c)] != cells[(r - i, c - j)] for r, c in inter):
                continue
            scored.append((len(inter), i * j, (i, j)))
    if not scored:
        return bgc, []
    m = max(s[0] for s in scored)
    top = [s for s in scored if s[0] == m]
    p = max(s[1] for s in top)
    return bgc, [s[2] for s in top if s[1] == p]


def _render(I, shift):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    out = {}
    for k in range(H_OUT):
        for r in range(h):
            for c in range(w):
                if I[r, c] == bgc:
                    continue
                rr, cc = r + k * shift[0], c + k * shift[1]
                if 0 <= rr < H_OUT and 0 <= cc < w:
                    if (rr, cc) in out and out[(rr, cc)] != I[r, c]:
                        return None
                    out[(rr, cc)] = int(I[r, c])
    G = np.full((H_OUT, w), bgc, dtype=int)
    for (rr, cc), v in out.items():
        G[rr, cc] = v
    return G


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if max_h < H_OUT:
        raise ValueError("output always has 10 rows")
    hcap = max(2, min(6, max_h))
    wcap = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, hcap))
        w = unifint(diff_lb, diff_ub, (8, wcap))
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w // 2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (H_OUT, w))
        hoffs = randint(0, ow // 2 + 1)
        for k in range(H_OUT // oh + 1):
            go = paint(go, shift(plcd, (k * oh, k * hoffs)))
        if len(palette(go[h:])) <= 1:
            continue
        gi = go[:h]
        if choice((True, False)):
            gi = vmirror(gi)
            go = vmirror(go)
        # keep only instances whose rule is unambiguous: background is the
        # majority colour, and every tied best shift extends to this output
        I = np.array(gi)
        if Counter(I.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        _, cands = _shift_candidates(I)
        if not cands:
            continue
        O = np.array(go)
        ok = True
        for s in cands:
            R = _render(I, s)
            if R is None or not np.array_equal(R, O):
                ok = False
                break
        if ok:
            return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: the input is the top of a pattern repeating every (dr, dc); extend it
    to 10 rows.  Unit of work: ONE NEW ROW, built from the finished row one
    period above it.  Each new row starts empty (the resize pads with 0): the
    earlier row's on-grid part is copied (CopyO) and pasted dc columns over,
    and then only the strip the paste could not reach (the columns the shift
    slid in from outside the grid, or the whole row if no earlier row exists
    one period up) is laid with background.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, cands = _shift_candidates(I)
    dr, dc = cands[0] if cands else (1, 0)

    ops, sels = [], []
    G = np.zeros((H_OUT, w), dtype=int)
    G[:h] = I

    # 1. grow the canvas to 10 rows (whole new canvas rectangle)
    ops.append(33)
    sels.append([0, 0, H_OUT - 1, w - 1])

    clip = None                       # what the clipboard currently holds
    sc0 = max(0, -dc)                 # source columns that stay on the grid
    sc1 = w - 1 - max(0, dc)
    for r in range(h, H_OUT):
        s = r - dr
        landed = set()
        if s >= 0 and sc0 <= sc1:
            tc0 = sc0 + dc
            seg = G[s, sc0:sc1 + 1]
            if np.any(seg != 0):
                if clip != tuple(seg.tolist()):    # clipboard doesn't hold it yet
                    ops.append(29)                 # CopyO the earlier row's segment
                    sels.append([s, sc0, 0, sc1 - sc0])
                    clip = tuple(seg.tolist())
                ops.append(30)                     # Paste it dc columns over on row r
                sels.append([r, tc0, 0, 0])
                for k, v in enumerate(seg):
                    if v != 0:
                        G[r, tc0 + k] = v
            landed = {tc0 + k for k in range(sc1 - sc0 + 1)}
        if bgc != 0:
            strip = [(r, c) for c in range(w) if c not in landed]
            if strip:                              # columns no copy reaches
                ops.append(int(bgc))
                sels.append(sel_of(strip))
                for rc in strip:
                    G[rc] = bgc

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
