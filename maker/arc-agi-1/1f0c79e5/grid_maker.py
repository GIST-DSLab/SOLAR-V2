"""
ARC Task: 1f0c79e5 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, **kwargs) -> dict:
    from utils import unifint, canvas, asindices, sfilter, shift, totuple, fill, mapply, rbind, shoot, dneighbors, decrement, multiply, add, invert
    if bgc is None or objc is None:
        cols = [c for c in range(10) if c != 2]
        bgc, objc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 24)))
    inds = asindices(gi)
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for k in range(nobjs):
        cands = sfilter(inds, lambda ij: shift(set(obj), ij).issubset(inds))
        if len(cands) == 0:
            break
        loc = random.choice(totuple(cands))
        plcd = shift(obj, loc)
        nred = unifint(diff_lb, diff_ub, (1, 3))
        reds = random.sample(totuple(plcd), nred)
        gi = fill(gi, objc, plcd)
        gi = fill(gi, 2, reds)
        for idx in reds:
            direc = decrement(multiply(2, add(idx, invert(loc))))
            go = fill(go, objc, mapply(rbind(shoot, direc), frozenset(plcd)))
        inds = (inds - plcd) - mapply(dneighbors, set(plcd))
    return {"input": gi, "output": go}


def _block_stamps(I, bgc, objc):
    """Rule read off I: each 2x2 block's red corners name diagonal directions; the block
    slides that way, stamping itself in objc at each step until it leaves the grid.
    Stamps are grouped per block, per direction, marching outward."""
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    blocks = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0 = min(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            reds = sorted((p[0] - r0, p[1] - c0) for p in comp if I[p[0], p[1]] == 2)
            if reds:
                blocks.append((r0, c0, reds))
    blocks.sort()
    G = I.copy()
    stamps = []
    for r0, c0, reds in blocks:
        for dr, dc in reds:
            di, dj = 2 * dr - 1, 2 * dc - 1
            k = 0
            while True:
                br, bc = r0 + k * di, c0 + k * dj
                rr0, rr1 = max(br, 0), min(br + 1, hi - 1)
                cc0, cc1 = max(bc, 0), min(bc + 1, wi - 1)
                if rr0 > rr1 or cc0 > cc1:
                    break
                if not np.all(G[rr0:rr1 + 1, cc0:cc1 + 1] == objc):
                    stamps.append([rr0, cc0, rr1 - rr0, cc1 - cc0])
                    G[rr0:rr1 + 1, cc0:cc1 + 1] = objc
                k += 1
    return stamps


def _replay(I, stamps, objc):
    G = I.copy()
    for r, c, h, w in stamps:
        G[r:r + h + 1, c:c + w + 1] = objc
    return G


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: canvas colour, guaranteed majority (blob cells <= h*w/6)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # object colour: the non-bg, non-red colour (from I, else from the demonstrations)
    objc = None
    for v in sorted(set(I.flatten().tolist())):
        if v != bgc and v != 2:
            objc = v
    if objc is None and examples:
        for ei, _eo in examples:
            ei = np.asarray(ei, dtype=int)
            eb = Counter(ei.flatten().tolist()).most_common(1)[0][0]
            for v in sorted(set(ei.flatten().tolist())):
                if v != eb and v != 2:
                    objc = v
            if objc is not None:
                break
    if objc is None:
        return [34], [[0, 0, hi - 1, wi - 1]]

    stamps = _block_stamps(I, bgc, objc)
    # the rule's own full picture (derived from I alone)
    F = _replay(I, stamps, objc)
    # a stamp whose every changed cell is re-stamped by a later step of the rule adds
    # nothing: drop it (same order, same grouping otherwise)
    i = 0
    while i < len(stamps):
        trial = stamps[:i] + stamps[i + 1:]
        if np.array_equal(_replay(I, trial, objc), F):
            stamps = trial
        else:
            i += 1

    ops, sels = [], []
    for s in stamps:
        # full (clipped) 2x2 stamp rectangle — the block's footprint at this step
        ops.append(int(objc))
        sels.append(list(s))
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f0c79e5"
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
                                f"for task 1f0c79e5"
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
                    f"Failed to build a complete episode for task 1f0c79e5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f0c79e5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
