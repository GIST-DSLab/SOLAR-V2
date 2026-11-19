"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    return {"bgc": bgc, "linc": linc, "objc": objc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return max(a, min(b, random.randint(min(lo, hi), max(lo, hi))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_h - 1) // 2))))
    w = _unifint(diff_lb, diff_ub, (2, max(2, min(14, (max_w - 1) // 2))))
    nc = _unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start}
    bx.discard(start)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj.add(ch)
        bx.discard(ch)
    q = np.full((h, w), bgc, dtype=int)
    for (i, j) in obj:
        q[i, j] = objc
    gi = np.full((2 * h + 1, 2 * w + 1), bgc, dtype=int)
    gi[h, :] = linc
    gi[:, w] = linc
    gi[:h, :w] = q
    top = np.hstack([q, np.fliplr(q)])
    go = np.vstack([top, np.flipud(top)])
    go[go == objc] = linc
    scf = random.choice([lambda g: g, np.flipud, np.fliplr,
                         lambda g: np.flipud(np.fliplr(g))])
    gi = scf(gi)
    go = scf(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    h, w = H // 2, W // 2
    linc = int(I[h, 0])  # colour of the full divider row/column
    origins = {(0, 0): (0, 0), (0, 1): (0, w + 1),
               (1, 0): (h + 1, 0), (1, 1): (h + 1, w + 1)}
    quads = {k: I[r:r + h, c:c + w] for k, (r, c) in origins.items()}
    bgc = None
    for k, q in quads.items():
        if len(np.unique(q)) == 1:
            bgc = int(q[0, 0])
            break
    qkey = [k for k, q in quads.items() if len(np.unique(q)) > 1][0]
    qr0, qc0 = origins[qkey]
    G = I.copy()
    ops, sels = [], []

    # 1. recolour the object's cells to the divider colour
    cells = [(qr0 + r, qc0 + c) for r in range(h) for c in range(w)
             if I[qr0 + r, qc0 + c] not in (bgc, linc)]
    ops.append(linc); sels.append(sel_of(cells))
    for (r, c) in cells:
        G[r, c] = linc

    # 2. copy the whole recoloured object quadrant (full rectangle; all cells nonzero)
    clip = G[qr0:qr0 + h, qc0:qc0 + w].copy()
    ops.append(29); sels.append([qr0, qc0, h - 1, w - 1])

    # 3. shrink canvas to 2h x 2w (full rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
    G = G[:2 * h, :2 * w].copy()

    # 4. paste the quadrant into each output quadrant, mirroring away from the source
    qi, qj = qkey
    for (i, j) in [(qi, qj), (qi, 1 - qj), (1 - qi, qj), (1 - qi, 1 - qj)]:
        r0, c0 = i * h, j * w
        if not np.array_equal(G[r0:r0 + h, c0:c0 + w], clip):
            ops.append(30); sels.append([r0, c0, 0, 0])
            G[r0:r0 + h, c0:c0 + w] = clip
        if j != qj:
            reg = G[r0:r0 + h, c0:c0 + w]
            if not np.array_equal(reg, np.fliplr(reg)):
                ops.append(26); sels.append([r0, c0, h - 1, w - 1])  # whole quadrant rectangle
                G[r0:r0 + h, c0:c0 + w] = np.fliplr(reg)
        if i != qi:
            reg = G[r0:r0 + h, c0:c0 + w]
            if not np.array_equal(reg, np.flipud(reg)):
                ops.append(27); sels.append([r0, c0, h - 1, w - 1])  # whole quadrant rectangle
                G[r0:r0 + h, c0:c0 + w] = np.flipud(reg)

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
