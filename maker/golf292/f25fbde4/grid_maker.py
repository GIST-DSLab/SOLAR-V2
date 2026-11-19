"""
ARC Task: f25fbde4 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    bh = max(1, min(15, h - 1, max_h // 2))
    bw = max(1, min(15, w - 1, max_w // 2))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (bh * bw) // 2)))
    nc = min(max(1, ncd), max(0, bh * bw - 1))
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (bh, bw)))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        cands = totuple((bounds - shp) & mapply(neighbors, shp))
        if len(cands) == 0:
            break
        shp.add(choice(cands))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(shp, (loci, locj))
    gi = fill(c, fgc, plcd)
    go = compress(gi)
    go = upscale(go, 2)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour. The shape's bbox is strictly smaller than
    # the grid, so the border is dominated by bgc.
    border = list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1])
    bgc = Counter(int(v) for v in border).most_common(1)[0][0]

    rows = [r for r in range(hi) if any(I[r, c] != bgc for c in range(wi))]
    cols = [c for c in range(wi) if any(I[r, c] != bgc for r in range(hi))]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    H, W = 2 * h, 2 * w

    ops, sels = [], []
    G = np.zeros((H, W), dtype=int)
    G[:h, :w] = I[r0:r1 + 1, c0:c1 + 1]

    # (1) compress: crop the canvas to the shape's bounding box (full rectangle).
    #     Unneeded only when the bbox already sits at the origin and the rest is 0.
    if not (r0 == 0 and c0 == 0 and bgc == 0):
        ops.append(33); sels.append([r0, c0, h - 1, w - 1])
    # (2) enlarge the canvas to the upscaled size 2h x 2w (full rectangle).
    ops.append(33); sels.append([0, 0, H - 1, W - 1])

    # (3) spread columns: original column j slides right by j, leaving an empty
    #     column after each one. Step k moves the whole block of columns from
    #     original index k onward (a full rectangle) one cell right.
    for k in range(1, w):
        start = 2 * k - 1            # current position of original column k
        end = start + (w - 1 - k)    # current position of the last column
        block = G[:h, start:end + 1]
        if not block.any():
            continue                 # nothing visible to move
        ops.append(22); sels.append([0, start, h - 1, end - start])
        G[:h, start + 1:end + 2] = block.copy()
        G[:h, start] = 0
    # (4) fill each empty odd column from its left neighbour: copy the spread
    #     block, paste it one column right (0s are transparent and the odd
    #     columns are empty, so each column lands exactly in the next one).
    src = G[:h, 0:W - 1].copy()
    if src.any():
        ops.append(29); sels.append([0, 0, h - 1, W - 2])
        ops.append(30); sels.append([0, 1, 0, 0])
        G[:h, 1:W][src != 0] = src[src != 0]

    # (5) same for rows: original row i slides down by i (full-width row blocks).
    for k in range(1, h):
        start = 2 * k - 1
        end = start + (h - 1 - k)
        block = G[start:end + 1, :W]
        if not block.any():
            continue
        ops.append(21); sels.append([start, 0, end - start, W - 1])
        G[start + 1:end + 2, :W] = block.copy()
        G[start, :W] = 0
    # (6) fill each empty odd row from the row above via one shifted paste.
    src = G[0:H - 1, :W].copy()
    if src.any():
        ops.append(29); sels.append([0, 0, H - 2, W - 1])
        ops.append(30); sels.append([1, 0, 0, 0])
        G[1:H, :W][src != 0] = src[src != 0]

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
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
                                f"for task f25fbde4"
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
                    f"Failed to build a complete episode for task f25fbde4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25fbde4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
