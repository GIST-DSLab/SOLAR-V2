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
    """Row recurrence read from I: the smallest period p (rows) and sideways step dc
    such that every row of I from p on is the row p above it slid dc columns."""
    h, w = I.shape
    for p in range(1, h):
        for dc in sorted(range(-(w - 1), w), key=lambda d: (abs(d), d)):
            if all(np.array_equal(I[r], _shift_row(I[r - p], dc, bgc)) for r in range(p, h)):
                return p, dc
    return None


def _continue(I, bgc):
    h, w = I.shape
    G = np.full((H_OUT, w), bgc, dtype=int)
    G[:h] = I
    rl = _rule(I, bgc)
    if rl is None:
        return None
    p, dc = rl
    for r in range(h, H_OUT):
        G[r] = _shift_row(G[r - p], dc, bgc)
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
        # keep only instances whose continuation is fixed by the input's row recurrence
        G = _continue(I, bgc)
        if G is not None and np.array_equal(G, np.array(go, dtype=int)):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = _bgc_of(I, examples)
    ops, sels = [], []

    # 1. extend the canvas down to the rule's 10 rows (full rectangle)
    ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
    G = np.zeros((H_OUT, w), dtype=int)
    G[:h] = I

    rl = _rule(I, bgc)
    if rl is not None and h < H_OUT:
        p, dc = rl
        # source columns that stay on the grid after sliding dc, and where they land
        if dc >= 0:
            c0, c1, dst = 0, w - 1 - dc, dc
        else:
            c0, c1, dst = -dc, w - 1, 0
        clip = None  # what the clipboard currently holds
        # 2. the unit is a whole ROW: each new row is the row p above it slid dc
        #    columns. Copy that row segment and paste it one period further down.
        for r in range(h, H_OUT):
            s = r - p
            seg = G[s, c0:c1 + 1].copy()
            tgt = G[r, dst:dst + len(seg)]
            nz = seg != 0
            if np.any(tgt[nz] != seg[nz]):
                key = tuple(seg.tolist())
                if key != clip:   # the clipboard does not already hold this row
                    ops.append(28 if s < h else 29); sels.append([s, c0, 0, c1 - c0])
                    clip = key
                ops.append(30); sels.append([r, dst, 0, 0])
                G[r, dst:dst + len(seg)][nz] = seg[nz]
            # 3. the strip this slid row could not reach is background; finish the
            #    row now, since it will itself be copied p rows further down
            if bgc != 0 and dc != 0:
                sc = 0 if dc > 0 else w + dc
                ops.append(int(bgc)); sels.append([r, sc, 0, abs(dc) - 1])
                G[r, sc:sc + abs(dc)] = bgc

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
