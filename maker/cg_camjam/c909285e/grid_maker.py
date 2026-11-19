"""
ARC Task: c909285e (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, max(lo_i, hi_i)))


def _find_box(I):
    """Verifier reading: among single-colour cell sets whose extent is at least
    2x2, keep those that are exactly the outline of their own bounding box,
    and take the smallest-area one."""
    I = np.asarray(I, dtype=int)
    best = None
    for col in sorted(set(I.flatten().tolist())):
        rs, cs = np.nonzero(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        cells = set(zip(rs.tolist(), cs.tolist()))
        outline = set()
        for c in range(c0, c1 + 1):
            outline.add((r0, c)); outline.add((r1, c))
        for r in range(r0, r1 + 1):
            outline.add((r, c0)); outline.add((r, c1))
        if cells != outline:
            continue
        area = h * w
        if best is None or area < best[0]:
            best = (area, int(r0), int(c0), int(h), int(w))
    return best


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "boxcol": boxcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxcol, **kwargs) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc, boxcol)]
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
        w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
        nfronts = _unifint(diff_lb, diff_ub, (1, (h + w) // 2))
        gi = np.full((h, w), bgc, dtype=int)
        inds = [(i, j) for i in range(h) for j in range(w)]
        for _k in range(nfronts):
            horiz = random.random() < 0.5
            loc = random.choice(inds)
            inds.remove(loc)
            col = random.choice(remcols)
            if horiz:
                gi[loc[0], :] = col
            else:
                gi[:, loc[1]] = col
        oh = _unifint(diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
        ow = _unifint(diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
        if h - oh - 1 < 1 or w - ow - 1 < 1:
            continue
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        r1, c1 = loci + oh - 1, locj + ow - 1
        gi[loci, locj:c1 + 1] = boxcol
        gi[r1, locj:c1 + 1] = boxcol
        gi[loci:r1 + 1, locj] = boxcol
        gi[loci:r1 + 1, c1] = boxcol
        go = gi[loci:r1 + 1, locj:c1 + 1].copy()
        # make sure the frame is the unambiguous (smallest) outline-shaped colour
        fb = _find_box(gi)
        if fb is None or fb[1:] != (loci, locj, oh, ow):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O):
    # Both readings (maker: the outline-shaped colour object; golf: rows/cols
    # holding the frame colour) agree: the answer is the frame's full bounding
    # rectangle, cropped out in one step. The frame is identified from I alone.
    I = np.asarray(I, dtype=int)
    box = _find_box(I)
    _, r0, c0, h, w = box
    ops, sels = [], []
    # CropGrid to the frame's full bounding rectangle (frame + interior intended)
    ops.append(33)
    sels.append([r0, c0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c909285e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c909285e"
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
                                f"for task c909285e"
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
                    f"Failed to build a complete episode for task c909285e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c909285e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
