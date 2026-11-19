"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# ---------------------------------------------------------------------------
# helpers shared by generate() / derive_operations()
# ---------------------------------------------------------------------------
_R1 = ((0, 0), (0, 4), (4, 0), (4, 4))
_R2 = ((2, 0), (0, 2), (4, 2), (2, 4))
_R3 = ((1, 1), (3, 1), (1, 3), (3, 3))
_R4 = ((2, 2),)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 9)))
    ccols = list(ccols)

    rings = [_R4, _R3, _R2, _R1]
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(1, h - 1) for j in range(1, w - 1)}
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ, tr, maxtr = 0, 0, 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - 5 and ij[0] <= w - 5]
        if not cands:
            break
        li, lj = random.choice(sorted(cands))
        plcd = {(li + a, lj + b) for a in range(5) for b in range(5)}
        if not plcd.issubset(inds):
            continue
        outb = {(li + a, lj + b) for a in range(-1, 6) for b in range(-1, 6)} - plcd
        ringcols = [random.choice(ccols) for _ in range(4)]
        plcdrings = [[(li + a, lj + b) for a, b in r] for r in rings]
        idx = random.randint(1, 3)
        remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
        remringcols = ringcols[1:idx] + ringcols[idx + 1:]
        numrs = unifint(diff_lb, diff_ub, (1, 2))
        locs = random.sample((0, 1), numrs)
        remrings = [rr for j, rr in enumerate(remrings) if j in locs]
        remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
        fgi, fgo = [], []
        for rr, col in zip(remrings, remringcols):
            k = 4 - unifint(diff_lb, diff_ub, (0, 3))
            for cell in random.sample(rr, k):
                fgi.append((cell, col))
            for cell in rr:
                fgo.append((cell, col))
        if not fgi:
            continue
        rs = [c[0] for c, _ in fgi]
        cs = [c[1] for c, _ in fgi]
        if min(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1) != 5:
            continue
        inds = (inds - plcd) - outb
        for (r, c) in plcdrings[0]:
            gi[r][c] = ringcols[0]; go[r][c] = ringcols[0]
        for (r, c) in plcdrings[idx]:
            gi[r][c] = ringcols[idx]; go[r][c] = ringcols[idx]
        for (r, c), col in fgi:
            gi[r][c] = col
        for (r, c), col in fgo:
            go[r][c] = col
        succ += 1
    return {"input": gi, "output": go}


def _find_blocks(I, bgc):
    """5x5 ring blocks: centre filled, odd-parity cells empty, full 5x5 extent,
    and the surrounding 7x7 frame entirely background (generator guarantees it)."""
    h, w = I.shape
    out = []
    for r0 in range(1, h - 5):
        for c0 in range(1, w - 5):
            win = I[r0 - 1:r0 + 6, c0 - 1:c0 + 6]
            if win.shape != (7, 7):
                continue
            frame = np.concatenate([win[0], win[6], win[1:6, 0], win[1:6, 6]])
            if (frame != bgc).any():
                continue
            b = win[1:6, 1:6]
            if b[2, 2] == bgc:
                continue
            if any(b[i, j] != bgc for i in range(5) for j in range(5) if (i + j) % 2):
                continue
            if not ((b[0] != bgc).any() and (b[4] != bgc).any()
                    and (b[:, 0] != bgc).any() and (b[:, 4] != bgc).any()):
                continue
            out.append((r0, c0))
    return out


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])

    ops, sels = [], []
    # Each block is a set of concentric monochrome rings; a ring that is only
    # partly drawn in I gets completed: paint the whole ring (its 4 symmetric
    # positions, named structurally from the block) with the ring's own colour.
    for (r0, c0) in _find_blocks(I, bgc):
        for ring in (_R1, _R2, _R3):
            cells = [(r0 + i, c0 + j) for i, j in ring]
            present = [(r, c) for r, c in cells if I[r, c] != bgc]
            colset = {int(I[r, c]) for r, c in present}
            if 0 < len(present) < 4 and len(colset) == 1:
                ops.append(colset.pop())
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
