"""
ARC Task: f9012d9b (RE-ARC) — LLM-generated grid_maker
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


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return random.randint(lo, max(lo, hi))


def sample_colors(num_examples=None) -> dict:
    # The rule (restore the blanked rectangle of a periodic wallpaper and output it)
    # does not depend on any colour role: 0 is always the hole, and the wallpaper
    # colours are drawn fresh per instance, as in the original generator.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    hp = _unif(diff_lb, diff_ub, (2, min(10, max_h // 3)))
    wp = _unif(diff_lb, diff_ub, (2, min(10, max_w // 3)))
    nc = _unif(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(range(1, 10), nc)
    tile = np.array([[random.choice(ccols) for _ in range(wp)] for _ in range(hp)])
    numhp = _unif(diff_lb, diff_ub, (3, max_h // hp))
    numwp = _unif(diff_lb, diff_ub, (3, max_w // wp))
    gi = np.tile(tile, (numhp, numwp))
    hc = random.randint(0, hp)
    wc = random.randint(0, wp)
    gi = gi[:gi.shape[0] - hc, :gi.shape[1] - wc]
    h, w = gi.shape
    sgh = _unif(diff_lb, diff_ub, (1, h - hp - 1))
    sgw = _unif(diff_lb, diff_ub, (1, w - wp - 1))
    li = random.randint(0, h - sgh)
    lj = random.randint(0, w - sgw)
    go = gi[li:li + sgh, lj:lj + sgw].copy()
    gi = gi.copy()
    gi[li:li + sgh, lj:lj + sgw] = 0
    mf = random.choice([
        lambda g: g,
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
        lambda g: g.T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 2).T,
    ])
    return {"input": mf(gi).tolist(), "output": mf(go).tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf entry): the input is a wallpaper with one rectangle
    # blanked to 0. The maker fills that hole bit by bit with greedy patches. Here the
    # hole is handled as ONE piece instead: find an intact window exactly the size of
    # the hole, a whole number of periods away, copy it, lay it over the hole, and
    # crop to the hole. If no such window exists, fall back to the period-shifted
    # patches. Everything is measured from I; O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    nz = I != 0

    # --- the hole: the only 0 cells in the input ---
    zs = np.argwhere(I == 0)
    r0, c0 = int(zs[:, 0].min()), int(zs[:, 1].min())
    r1, c1 = int(zs[:, 0].max()), int(zs[:, 1].max())
    hh, hw = r1 - r0 + 1, c1 - c0 + 1

    # --- the wallpaper's periods, measured on the visible cells ---
    def period(axis):
        n = hi if axis == 0 else wi
        for p in range(1, n):
            if axis == 0:
                a, b = I[:n - p, :], I[p:, :]
                m = nz[:n - p, :] & nz[p:, :]
            else:
                a, b = I[:, :n - p], I[:, p:]
                m = nz[:, :n - p] & nz[:, p:]
            if bool(np.all(a[m] == b[m])):
                return p
        return n

    pv, ph = period(0), period(1)

    ops, sels = [], []
    drs = [k * pv for k in range(-(hi // pv + 1), hi // pv + 2)]
    dcs = [m * ph for m in range(-(wi // ph + 1), wi // ph + 2)]

    # --- look for an intact window the size of the hole, nearest period shift first ---
    best = None
    for dr in drs:
        for dc in dcs:
            if dr == 0 and dc == 0:
                continue
            sr, sc = r0 + dr, c0 + dc
            if sr < 0 or sc < 0 or sr + hh > hi or sc + hw > wi:
                continue
            if not nz[sr:sr + hh, sc:sc + hw].all():
                continue
            key = (abs(dr) // pv + abs(dc) // ph, abs(dr) + abs(dc), dr, dc)
            if best is None or key < best[0]:
                best = (key, sr, sc)

    if best is not None:
        _, sr, sc = best
        # full rectangle: the intact window, same size as the hole
        ops.append(28)
        sels.append([sr, sc, hh - 1, hw - 1])
        # paste it over the hole (only its top-left corner is needed)
        ops.append(30)
        sels.append([r0, c0, 0, 0])
    else:
        # no single intact window: rebuild the hole from period-shifted patches
        remaining = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        while remaining:
            cand = None
            for dr in drs:
                for dc in dcs:
                    if dr == 0 and dc == 0:
                        continue
                    sr0, sr1 = max(0, r0 - dr), min(hi - 1, r1 - dr)
                    sc0, sc1 = max(0, c0 - dc), min(wi - 1, c1 - dc)
                    if sr0 > sr1 or sc0 > sc1:
                        continue
                    gain = [(r, c) for (r, c) in remaining
                            if sr0 <= r - dr <= sr1 and sc0 <= c - dc <= sc1
                            and I[r - dr, c - dc] != 0]
                    if not gain:
                        continue
                    key = (-len(gain), abs(dr) + abs(dc), abs(dr), abs(dc))
                    if cand is None or key < cand[0]:
                        cand = (key, dr, dc, gain, (sr0, sc0, sr1, sc1))
            if cand is None:
                break
            _, dr, dc, gain, (sr0, sc0, sr1, sc1) = cand
            # full rectangle: a patch of pattern a whole number of periods away
            ops.append(28)
            sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
            ops.append(30)
            sels.append([sr0 + dr, sc0 + dc, 0, 0])
            remaining -= set(gain)

    # full rectangle: the answer is exactly the restored hole
    ops.append(33)
    sels.append([r0, c0, hh - 1, hw - 1])
    ops.append(34)
    sels.append([0, 0, hh - 1, hw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
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
                                f"for task f9012d9b"
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
                    f"Failed to build a complete episode for task f9012d9b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f9012d9b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
