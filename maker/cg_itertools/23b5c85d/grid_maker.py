"""
ARC Task: 23b5c85d (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + int(np.floor(diff_lb * (b - a)))
    hi = a + int(np.ceil(diff_ub * (b - a)))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _solid_regions(G):
    """Every colour whose cells exactly fill their bounding box -> (area, colour, r0, c0, h, w)."""
    G = np.asarray(G, dtype=int)
    out = []
    for c in sorted(set(G.flatten().tolist())):
        rs, cs = np.nonzero(G == c)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if len(rs) == bh * bw:
            out.append((bh * bw, c, int(r0), int(c0), int(bh), int(bw)))
    return out


def _smallest_solid_colour(G):
    regs = _solid_regions(G)
    if not regs:
        return None
    best = min(regs, key=lambda t: t[0])
    return best[1]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    max_h = max(5, min(30, max_h))
    max_w = max(5, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (5, max_h))
    w = _unifint(diff_lb, diff_ub, (5, max_w))
    colopts = [c for c in range(10) if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = None
    oh = _unifint(diff_lb, diff_ub, (2, h - 1))
    ow = _unifint(diff_lb, diff_ub, (2, w - 1))
    num = _unifint(diff_lb, diff_ub, (1, 8))
    cnt = 0
    while cnt < num and colopts:
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        col = random.choice(colopts)
        colopts = [c for c in colopts if c != col]
        gi2 = gi.copy()
        gi2[loci:loci + oh, locj:locj + ow] = col
        if _smallest_solid_colour(gi2) != col:
            break
        gi = gi2
        go = np.full((oh, ow), col, dtype=int)
        oh = _unifint(diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
        ow = _unifint(diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
        if oh < 1 or ow < 1:
            break
        cnt += 1
    if go is None:
        raise ValueError("no valid rectangle placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Reading: the unit is "all cells of one colour". Among colours whose cells
    # form a solid rectangle (count == bbox area), the target is the one with the
    # fewest cells (the code-golf reading: rarest colour). The verifier only
    # considers solid colours, so we keep that filter. When a colour is solid,
    # its cells and its bounding box are the same set, so both readings give the
    # same single op: CropGrid on that colour's cells. Measured from I only.
    regs = _solid_regions(I)
    _, col, r0, c0, bh, bw = min(regs, key=lambda t: t[0])
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(I == col))]

    ops, sels = [], []
    # CropGrid to the bbox of the selected colour's cells. They fill the bbox
    # exactly, so a 0-coloured target is fine too: the crop's zero padding is
    # that colour.
    ops.append(33); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23b5c85d"
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
                                f"for task 23b5c85d"
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
                    f"Failed to build a complete episode for task 23b5c85d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23b5c85d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
