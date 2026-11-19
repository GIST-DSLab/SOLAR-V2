"""
ARC Task: 6cf79266 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque


def sample_colors(num_examples=None) -> dict:
    # The rule depends only on where the background (0) forms 3x3 blocks,
    # not on the noise colours; still, keep one palette for the whole episode.
    cols = [c for c in range(10) if c not in (0, 1)]
    nfgcs = random.randint(1, 8)
    return {"ccols": random.sample(cols, nfgcs)}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, ccols=None) -> dict:
    if ccols is None:
        cols = difference(interval(0, 10, 1), (0, 1))
        nfgcs = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(cols, nfgcs)
    ccols = list(ccols)

    hub = max(6, min(30, int(max_h)))
    wub = max(6, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (6, hub))
    w = unifint(diff_lb, diff_ub, (6, wub))

    gi = canvas(-1, (h, w))
    fgcobj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, fgcobj)

    num = unifint(diff_lb, diff_ub, (int(0.25 * h * w), int(0.6 * h * w)))
    inds = asindices(gi)
    locs = sample(totuple(inds), num)
    gi = fill(gi, 0, locs)

    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
    cands = asindices(canvas(-1, (h - 2, w - 2)))
    locs = sample(totuple(cands), noccs)
    mini = asindices(canvas(-1, (3, 3)))
    for ij in locs:
        gi = fill(gi, 0, shift(mini, ij))

    trg = recolor(0, mini)
    occs = occurrences(gi, trg)
    go = tuple(e for e in gi)
    for occ in occs:
        go = fill(go, 1, shift(mini, occ))

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (measured from I alone): every 3x3 window of I that is entirely
    background (0) is a 'hole'; stamp that 3x3 unit with colour 1."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def unit_cells(o):
        r, c = o
        return frozenset((r + dr, c + dc) for dr in range(3) for dc in range(3))

    # --- 1. measure the occurrences of the 3x3 all-background unit in I ---
    occs = []
    for r in range(hi - 2):
        for c in range(wi - 2):
            if not I[r:r + 3, c:c + 3].any():      # window is all background
                occs.append((r, c))

    # --- 2. drop occurrences that are fully covered by the other ones, so
    #        every stamp we emit paints at least one cell nothing else paints ---
    keep = list(occs)
    cover = Counter()
    for o in keep:
        cover.update(unit_cells(o))
    shrinking = True
    while shrinking:
        shrinking = False
        for o in list(keep):
            cs = unit_cells(o)
            if all(cover[x] >= 2 for x in cs):
                keep.remove(o)
                for x in cs:
                    cover[x] -= 1
                shrinking = True

    # --- 3. group the surviving stamps into the connected blobs they form,
    #        so painting proceeds region by region (not in raster order) ---
    painted = set()
    for o in keep:
        painted |= unit_cells(o)

    comp_id = {}
    comps = []
    for cell in sorted(painted):
        if cell in comp_id:
            continue
        idx = len(comps)
        comp = []
        dq = deque([cell])
        comp_id[cell] = idx
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in painted and (nr, nc) not in comp_id:
                    comp_id[(nr, nc)] = idx
                    dq.append((nr, nc))
        comps.append(sorted(comp))

    groups = [[] for _ in comps]
    for o in keep:
        groups[comp_id[o]].append(o)

    order = sorted(range(len(comps)), key=lambda i: comps[i][0])

    ops, sels = [], []
    for i in order:
        for (r, c) in sorted(groups[i]):
            # selection is exactly the full 3x3 rectangle of this occurrence
            ops.append(1)                    # Color1
            sels.append([r, c, 2, 2])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6cf79266"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6cf79266"
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
                                f"for task 6cf79266"
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
                    f"Failed to build a complete episode for task 6cf79266 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6cf79266-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
