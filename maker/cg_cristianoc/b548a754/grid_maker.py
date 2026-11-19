"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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
    bgc, boxc, inc, dotc = random.sample(range(10), 4)
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc}


def _box_cells(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c)); cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0)); cells.add((r, c1))
    return cells


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxc=None, inc=None, dotc=None) -> dict:
    if None in (bgc, boxc, inc, dotc):
        bgc, boxc, inc, dotc = random.sample(range(10), 4)

    def unifint(lb, ub):
        if ub <= lb:
            return lb
        a = lb + (ub - lb) * diff_lb
        b = lb + (ub - lb) * diff_ub
        return random.randint(int(round(a)), int(round(b)))

    k = random.choice([0, 1, 2, 3])          # number of CCW quarter turns
    if k % 2 == 1:
        H, W = max_w, max_h                   # dims swap under rotation
    else:
        H, W = max_h, max_w
    H = max(5, min(30, H)); W = max(4, min(30, W))

    h = unifint(5, H)
    w = unifint(4, W)
    hi = unifint(4, h - 1)
    wi = unifint(3, w - 1)
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)

    go = np.full((h, w), bgc, dtype=int)
    for (r, c) in _box_cells(loci, locj, loci + hi - 1, locj + wi - 1):
        go[r, c] = boxc
    go[loci + 1:loci + hi - 1, locj + 1:locj + wi - 1] = inc

    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = np.full((h, w), bgc, dtype=int)
    for (r, c) in _box_cells(loci, locj, cutoff, locj + wi - 1):
        gi[r, c] = boxc
    gi[loci + 1:cutoff, locj + 1:locj + wi - 1] = inc
    lc = random.randint(locj, locj + wi - 1)
    gi[loci + hi - 1, lc] = dotc

    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # --- read the scene out of I: rectangular frame (outline + uniform inside) and a lone dot ---
    boxc = inc = dotc = None
    for cand in np.unique(I).tolist():
        cells = np.argwhere(I == cand)
        a0, b0 = int(cells[:, 0].min()), int(cells[:, 1].min())
        a1, b1 = int(cells[:, 0].max()), int(cells[:, 1].max())
        bh, bw = a1 - a0 + 1, b1 - b0 + 1
        if bh < 3 or bw < 3 or len(cells) != 2 * bh + 2 * bw - 4:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        if not (np.all(sub[0, :] == cand) and np.all(sub[-1, :] == cand)
                and np.all(sub[:, 0] == cand) and np.all(sub[:, -1] == cand)):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or len(np.unique(inner)) != 1 or int(inner[0, 0]) == cand:
            continue
        cand_inc = int(inner[0, 0])
        rest = [x for x in np.unique(I).tolist() if x != cand and x != cand_inc]
        if len(rest) != 2:
            continue
        singles = [x for x in rest if int((I == x).sum()) == 1]
        if len(singles) != 1:
            continue
        boxc, inc, dotc = cand, cand_inc, singles[0]
        break

    # The enlarged box is the bounding rectangle of {frame, dot} -- the unit this reading works on.
    pts = np.argwhere((I == boxc) | (I == dotc))
    R0, C0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    R1, C1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    ops, sels = [], []
    # 1. Flood the whole rectangle spanned by box + dot with the interior colour
    #    (full rectangle intended -> bbox selection is exact).
    ops.append(int(inc)); sels.append([R0, C0, R1 - R0, C1 - C0])
    # 2. Draw the frame back as the outline of that rectangle.
    ops.append(int(boxc)); sels.append(sel_of(sorted(_box_cells(R0, C0, R1, C1))))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
