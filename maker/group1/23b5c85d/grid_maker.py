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


def sample_colors(num_examples=None) -> dict:
    # The rule only looks at shapes: find the smallest single-colour region that
    # completely fills its bounding box. Object colours don't change the rule,
    # so only the background colour is fixed for the episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _solid_min_color(g):
    # Same rule as the verifier: split by colour, keep the colours whose cells
    # exactly fill their bounding box, and take the one with the smallest box.
    best = None
    for c in sorted(set(g.flatten().tolist())):
        rs, cs = np.where(g == c)
        bh = rs.max() - rs.min() + 1
        bw = cs.max() - cs.min() + 1
        if len(rs) != bh * bw:
            continue
        if best is None or bh * bw < best[0]:
            best = (bh * bw, c)
    return None if best is None else best[1]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, **kwargs) -> dict:
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
        w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
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
            colopts.remove(col)
            gi2 = gi.copy()
            gi2[loci:loci + oh, locj:locj + ow] = col
            if _solid_min_color(gi2) != col:
                break
            gi = gi2
            go = np.full((oh, ow), col, dtype=int)
            oh = _unifint(diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
            ow = _unifint(diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
            if oh < 1 or ow < 1:
                break
            cnt += 1
        if go is None:
            continue
        # sanity: the verifier's crop must equal go
        c = _solid_min_color(gi)
        rs, cs = np.where(gi == c)
        crop = gi[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
        if crop.shape != go.shape or not np.array_equal(crop, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Rule, measured from I only: split I by colour, keep the colours that are
    # solid rectangles, and pick the one with the smallest area.
    best = None  # (area, r0, c0, bh, bw)
    for c in sorted(set(I.flatten().tolist())):
        rs, cs = np.where(I == c)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if len(rs) != bh * bw:
            continue
        if best is None or bh * bw < best[0]:
            best = (bh * bw, r0, c0, bh, bw)
    _, r0, c0, bh, bw = best

    ops, sels = [], []
    cur_h, cur_w = hi, wi

    # Step 1: keep only the band of rows the target rectangle spans (all columns).
    # The selection is a full rectangle: rows r0..r0+bh-1 across the whole width.
    if not (r0 == 0 and bh == hi):
        ops.append(33)
        sels.append([r0, 0, bh - 1, cur_w - 1])
        cur_h = bh

    # Step 2: from that band, keep only the columns the rectangle spans.
    # The band now starts at row 0, so the selection is rows 0..bh-1,
    # columns c0..c0+bw-1 (a full rectangle).
    if not (c0 == 0 and bw == wi):
        ops.append(33)
        sels.append([0, c0, cur_h - 1, bw - 1])
        cur_w = bw

    ops.append(34)
    sels.append([0, 0, bh - 1, bw - 1])
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
