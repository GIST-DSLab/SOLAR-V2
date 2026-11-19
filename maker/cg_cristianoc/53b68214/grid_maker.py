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
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _bgc_of(I, examples=None):
    cnt = Counter(np.asarray(I).flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i).flatten().tolist())
    return cnt.most_common(1)[0][0]


def _shift_row(row, dc, bgc):
    w = len(row)
    out = np.full(w, bgc, dtype=int)
    if dc >= 0:
        out[dc:] = row[:w - dc]
    else:
        out[:w + dc] = row[-dc:]
    return out


def _rule(I, bgc):
    """Repetition step read from I: smallest row period p and sideways step dc
    such that every row of I from p on is the row p above it slid dc columns."""
    h, w = I.shape
    for p in range(1, h):
        for dc in sorted(range(-(w - 1), w), key=lambda d: (abs(d), d)):
            if all(np.array_equal(I[r], _shift_row(I[r - p], dc, bgc)) for r in range(p, h)):
                return p, dc
    return None


def _stamp_all(I, bgc):
    """The whole object of I, re-stamped as one unit at multiples of its block step."""
    h, w = I.shape
    rl = _rule(I, bgc)
    if rl is None:
        return None
    p, dc = rl
    m = max(1, h // p)
    G = np.full((H_OUT, w), bgc, dtype=int)
    G[:h] = I
    obj = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if I[r, c] != bgc}
    k = 1
    while k * m * p < H_OUT:
        sr, sc = k * m * p, k * m * dc
        for (r, c), col in obj.items():
            if 0 <= r + sr < H_OUT and 0 <= c + sc < w:
                G[r + sr, c + sc] = col
        k += 1
    return G


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, mirrored=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if bgc is None:
        bgc = random.choice(range(10))
    if mirrored is None:
        mirrored = random.choice((True, False))
    max_w = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (8, max_w))
        remcols = [c for c in range(10) if c != bgc]
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = random.sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        cells = random.sample(bounds, ncells)
        mi = min(i for i, _ in cells); mj = min(j for _, j in cells)
        obj = [(random.choice(ccols), (i - mi, j - mj)) for i, j in cells]
        oh = max(i for _, (i, _) in obj) + 1
        ow = max(j for _, (_, j) in obj) + 1
        locj = random.randint(0, w // 2)
        go = [[bgc] * w for _ in range(H_OUT)]
        hoffs = random.randint(0, ow // 2 + 1)
        for k in range(H_OUT // oh + 1):
            for c, (i, j) in obj:
                r, cc = i + k * oh, j + locj + k * hoffs
                if 0 <= r < H_OUT and 0 <= cc < w:
                    go[r][cc] = c
        if len(set(v for row in go[h:] for v in row)) <= 1:
            continue
        if mirrored:
            go = [row[::-1] for row in go]
        gi = [row[:] for row in go[:h]]
        I = np.array(gi, dtype=int)
        if _bgc_of(I) != bgc:
            continue
        # keep only instances whose continuation is fixed by the input alone
        G = _stamp_all(I, bgc)
        if G is not None and np.array_equal(G, np.array(go, dtype=int)):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading (code-golf reading): every non-background cell of I together is ONE
    unit -- a block of m whole periods of the repeating motif.  The unit is copied
    from the input and stamped whole further down the grid, block after block,
    each stamp shifted by the block step (m*p rows, m*dc columns) measured from I.
    Agrees with the maker on the canvas growth + background base; differs in the
    unit (whole object stamp vs. one band per step, colour by colour)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = _bgc_of(I, examples)
    ops, sels = [], []

    # 1. grow the canvas to the rule's 10 rows (full rectangle -> bbox is exact)
    ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
    # 2. lay the background over the rows the resize added (full rectangle)
    if bgc != 0 and h < H_OUT:
        ops.append(int(bgc)); sels.append([h, 0, H_OUT - 1 - h, w - 1])
    G0 = np.zeros((H_OUT, w), dtype=int)
    G0[:h] = I
    if bgc != 0:
        G0[h:] = bgc

    # 3. the whole object and its block step, both read from I
    obj = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if I[r, c] != bgc}
    rl = _rule(I, bgc)
    steps = []
    if rl is not None and obj:
        p, dc = rl
        m = max(1, h // p)
        k = 1
        while k * m * p < H_OUT:
            sr, sc = k * m * p, k * m * dc
            # cells pushed past the left edge cannot be pasted (origin must be >= 0)
            on = [(r, c) for (r, c) in obj if r + sr < H_OUT and 0 <= c + sc < w]
            nz = sorted((r, c) for r, c in on if obj[(r, c)] != 0)
            zs = sorted((r + sr, c + sc) for r, c in on if obj[(r, c)] == 0)
            if nz:
                r0 = min(r for r, _ in nz); c0 = min(c for _, c in nz)
                steps.append(["stamp", nz, (r0 + sr, c0 + sc), k])
            if zs and bgc != 0:
                # Copy/Paste cannot carry colour 0: paint where this stamp's 0-cells land
                steps.append(["zero", zs, k])
            k += 1

    def _emit(stp):
        o, s, clip = [], [], None
        for st in stp:
            if st[0] == "stamp":
                if clip != st[1]:
                    o.append(28); s.append(sel_of(st[1]))      # CopyI: the object's cells
                    clip = st[1]
                o.append(30); s.append([st[2][0], st[2][1], 0, 0])  # paste the stamp
            else:
                o.append(0); s.append(sel_of(st[1]))
        return o, s

    def _sim(o, s):
        G = G0.copy()
        clip = None
        for op, sel in zip(o, s):
            if op == 28:
                cells = [tuple(x) for x in sel["cells"]]
                r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
                clip = {(r - r0, c - c0): int(I[r, c]) for r, c in cells if I[r, c] != 0}
            elif op == 30:
                if clip:
                    for (dr, dcc), col in clip.items():
                        rr, cc = sel[0] + dr, sel[1] + dcc
                        if rr < H_OUT and cc < w:   # beyond the canvas: invisible
                            G[rr, cc] = col
            else:
                for r, c in sel["cells"]:
                    G[r, c] = op
        return G

    o, s = _emit(steps)
    target = _sim(o, s)
    # drop any op the remaining ones make unnecessary (a stamp wholly re-covered
    # by a later stamp, a re-copy whose old clipboard already does the job) --
    # judged purely against the I-derived stamping, never against O
    changed = True
    while changed:
        changed = False
        for i in range(len(o)):
            to, ts = o[:i] + o[i + 1:], s[:i] + s[i + 1:]
            if np.array_equal(_sim(to, ts), target):
                o, s = to, ts
                changed = True
                break
    ops += o; sels += s

    ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
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
