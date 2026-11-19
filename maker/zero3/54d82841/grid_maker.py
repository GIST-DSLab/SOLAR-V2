"""
ARC Task: 54d82841 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)

    def unif(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))

    h = unif(5, max(5, max_h))
    w = unif(5, max(5, max_w))
    remcols = [c for c in cols if c != bgc]
    nshps = unif(1, max(1, w // 3))
    gi = np.full((h, w), bgc, dtype=int)
    go = gi.copy()
    locs = list(range(1, w - 1))
    for _ in range(nshps):
        if not locs:
            break
        loc = random.choice(locs)
        locs = [l for l in locs if abs(l - loc) > 2]
        loci = random.randint(1, h - 1)
        col = random.choice(remcols)
        shp = [(loci - 1, loc - 1), (loci - 1, loc), (loci - 1, loc + 1),
               (loci, loc - 1), (loci, loc + 1)]
        for r, c in shp:
            gi[r, c] = col
            go[r, c] = col
        go[h - 1, loc] = 4
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Each U-shape is recognised by its mouth cell (a background cell) plus the
    # five wall cells around it, all of one non-background colour. Matching the
    # template directly (instead of connected components) keeps two adjacent
    # same-coloured U's from merging into one blob and being skipped.
    TEMPL = {
        "down":  [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)],
        "up":    [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1)],
        "right": [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0)],
        "left":  [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0)],
    }
    found = {k: [] for k in TEMPL}
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc:
                continue
            for k, offs in TEMPL.items():
                cells = [(r + dr, c + dc) for dr, dc in offs]
                if not all(0 <= y < h and 0 <= x < w for y, x in cells):
                    continue
                vals = {int(I[y, x]) for y, x in cells}
                if len(vals) == 1 and bgc not in vals:
                    found[k].append((r, c))

    # the whole grid shares one orientation: take the one the objects agree on
    d = max(found, key=lambda k: len(found[k]))
    targets = []
    for (r, c) in found[d]:
        if d == "down":
            targets.append((h - 1 - r, (h - 1, c)))
        elif d == "up":
            targets.append((r, (0, c)))
        elif d == "right":
            targets.append((w - 1 - c, (r, w - 1)))
        else:
            targets.append((c, (r, 0)))

    # project each object's mouth onto the border it faces, nearest object first
    targets.sort(key=lambda t: t[0])

    ops, sels = [], []
    for _, (mr, mc) in targets:
        ops.append(4)
        sels.append([int(mr), int(mc), 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 54d82841"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 54d82841"
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
                                f"for task 54d82841"
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
                    f"Failed to build a complete episode for task 54d82841 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"54d82841-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
