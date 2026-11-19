"""
ARC Task: f25ffba3 (RE-ARC) — LLM-generated grid_maker
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

SIDES = [{"mirror_lr": False}, {"mirror_lr": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SIDES):
        examples = [dict(v) for v in SIDES]
        examples += [dict(random.choice(SIDES)) for _ in range(n_ex - len(SIDES))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(SIDES, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    a = max(lo, min(hi, a))
    b = max(lo, min(hi, b))
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, mirror_lr=None,
             mirror_ud=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.randint(1, 9)
    if mirror_lr is None:
        mirror_lr = random.choice([True, False])
    if mirror_ud is None:
        mirror_ud = random.choice([True, False])
    remcols_all = [c for c in range(1, 10) if c != bgc]
    hmax = max(1, min(14, (max_h - 1) // 2))
    h = _unifint(diff_lb, diff_ub, 1, hmax) * 2 + 1
    wmax = max(3, min(15, max_w // 2))
    w = _unifint(diff_lb, diff_ub, 3, wmax)
    numcols = _unifint(diff_lb, diff_ub, 1, 8)
    remcols = random.sample(remcols_all, numcols)
    canv = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, 2, h * w - 2)
    bx = {(r, c) for r in range(h) for c in range(w)}

    def nbrs(cells):
        out = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        out.add((r + dr, c + dc))
        return out

    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    bx.discard(start)
    for _ in range(nc - 1):
        cand = sorted(bx & nbrs(obj.keys()))
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    while min(r for r, _ in obj) > h // 2 - 1 or max(r for r, _ in obj) < h // 2 + 1:
        cand = sorted(bx & nbrs(obj.keys()))
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    gix = canv.copy()
    for (r, c), v in obj.items():
        gix[r, c] = v
    # push non-bg cells to the left in each row (stable)
    rows = []
    for r in range(h):
        row = list(gix[r])
        rows.append([x for x in row if x != bgc] + [x for x in row if x == bgc])
    gix = np.array(rows, dtype=int)
    gi = np.hstack([gix, canv])
    go = np.hstack([gix, gix[:, ::-1]])
    if mirror_lr:
        gi = gi[:, ::-1]
        go = go[:, ::-1]
    if mirror_ud:
        gi = gi[::-1, :]
        go = go[::-1, :]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    half = wi // 2
    left = I[:, :half]
    right = I[:, half:]
    if len(np.unique(right)) == 1 and len(np.unique(left)) > 1:
        src_c, dst_c = 0, half
    else:
        src_c, dst_c = half, 0
    ops, sels = [], []
    # Copy the whole object half (full rectangle; all colours are non-zero)
    ops.append(28); sels.append([0, src_c, hi - 1, half - 1])
    # Paste it onto the empty half
    ops.append(30); sels.append([0, dst_c, 0, 0])
    # Mirror the pasted half left<->right in place (full rectangle intended)
    ops.append(26); sels.append([0, dst_c, hi - 1, half - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f25ffba3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25ffba3"
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
                                f"for task f25ffba3"
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
                    f"Failed to build a complete episode for task f25ffba3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25ffba3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
