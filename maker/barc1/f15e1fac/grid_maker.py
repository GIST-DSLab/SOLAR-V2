"""
ARC Task: f15e1fac (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # colour 2 is hard-coded for the markers; bgc and the line colour are sampled
    cols = [c for c in range(10) if c != 2]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c != 2]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)

    H = max(4, min(30, max_h))
    W = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, H))
    w = unifint(diff_lb, diff_ub, (4, W))
    nsps = max(1, unifint(diff_lb, diff_ub, (1, (w - 1) // 2)))
    ngps = max(1, unifint(diff_lb, diff_ub, (1, (h - 1) // 2)))
    spsj = sorted(random.sample(range(1, w - 1), min(nsps, w - 2)))
    gpsi = sorted(random.sample(range(1, h - 1), min(ngps, h - 2)))

    gi = np.full((h, w), bgc, dtype=int)
    for jj in spsj:
        gi[0, jj] = linc
    for ii in gpsi:
        gi[ii, 0] = 2
    go = gi.copy()
    ofs = 0
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            c = jj + ofs
            if 0 <= c < w:
                go[a:b + 1, c] = linc
        ofs += 1

    mfs = [
        lambda g: g,                         # identity
        lambda g: g.T,                       # dmirror
        lambda g: np.rot90(g, 2).T,          # cmirror
        lambda g: np.fliplr(g),              # vmirror
        lambda g: np.flipud(g),              # hmirror
        lambda g: np.rot90(g, 3),            # rot90
        lambda g: np.rot90(g, 2),            # rot180
        lambda g: np.rot90(g, 1),            # rot270
    ]
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    # The reference solution and the maker break the rule up the same way:
    # each seed sends out its own flow line toward the far edge, and every
    # time that line reaches a row/column holding a 2-marker it moves one cell
    # away from the marker edge. So each seed is drawn as its own object, one
    # straight segment at a time (one segment for each gap between markers).
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Background = the colour the canvas was painted with. It is always the
    # majority colour: seeds and markers together take fewer than half of
    # one edge.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    linc = None
    for c in np.unique(I).tolist():
        if c != bgc and c != 2:
            linc = c
    if linc is None:
        return [34], [[0, 0, ho - 1, wo - 1]]

    # Find the view where the markers sit on the left edge and the seeds on
    # the top edge. That decides the flow and push directions from I alone.
    idx = np.arange(hi * wi).reshape(hi, wi)
    C = IX = None
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(I, k)
            ix = np.rot90(idx, k)
            if flip:
                g = np.fliplr(g)
                ix = np.fliplr(ix)
            r2, c2 = np.where(g == 2)
            rl, cl = np.where(g == linc)
            if r2.size and rl.size and np.all(c2 == 0) and np.all(rl == 0):
                C, IX = g, ix
                break
        if C is not None:
            break
    if C is None:
        return [34], [[0, 0, ho - 1, wo - 1]]

    hc, wc = C.shape
    gps = sorted(set(np.where(C == 2)[0].tolist()))     # rows where the flow gets pushed
    sps = sorted(np.where(C[0] == linc)[0].tolist())    # seed columns
    starts = [0] + gps
    ends = [x - 1 for x in gps] + [hc - 1]

    ops, sels = [], []
    for j in sps:                       # one flow line per seed
        for ofs, (a, b) in enumerate(zip(starts, ends)):
            col = j + ofs               # pushed one step for each marker passed
            if col >= wc:               # flow pushed off the grid -> stops
                break
            cells = []
            for r in range(a, b + 1):
                if C[r, col] == linc:   # the seed pixel itself is already drawn
                    continue
                rr, cc = divmod(int(IX[r, col]), wi)
                cells.append((rr, cc))
            if not cells:
                continue
            # straight run of the flow between two marker rows (a line segment)
            ops.append(int(linc))
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
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
                                f"for task f15e1fac"
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
                    f"Failed to build a complete episode for task f15e1fac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f15e1fac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
