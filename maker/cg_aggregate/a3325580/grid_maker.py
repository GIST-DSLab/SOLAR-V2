"""
ARC Task: a3325580 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    random.shuffle(remcols)
    return {"bgc": bgc, "ccols": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    if bgc is None:
        bgc = random.randint(0, 9)
    if ccols is None:
        ccols = [c for c in range(10) if c != bgc]
        random.shuffle(ccols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, 9))
    gi = [[bgc] * w for _ in range(h)]
    lmocc = set()
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = random.randint(nobjs + 1, 30)

    def dnb(cell):
        a, b = cell
        return {(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)}

    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(1, 6)
        ow = random.randint(1, 6)
        while oh * ow < mxncells:
            oh = random.randint(1, 6)
            ow = random.randint(1, 6)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = random.choice(sorted(bounds))
        obj = {sp}
        for _ in range(ncells - 1):
            nb = set()
            for c in obj:
                nb |= dnb(c)
            obj.add(random.choice(sorted((bounds - obj) & nb)))
        fo = frozenset(obj)
        if fo in seenobjs:
            continue
        seenobjs.add(fo)
        mr = min(a for a, _ in obj)
        mc = min(b for _, b in obj)
        obj = {(a - mr, b - mc) for a, b in obj}
        oh = max(a for a, _ in obj) + 1
        ow = max(b for _, b in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = {(a + loc[0], b + loc[1]) for a, b in obj}
        if plcd.issubset(inds):
            nb = set()
            for c in plcd:
                nb |= dnb(c)
            inds = (inds - plcd) - nb
            for a, b in plcd:
                gi[a][b] = ccols[succ]
            succ += 1
            lmocc.add(loc[1])

    # colours present with their cell count and leftmost column
    cnt = Counter()
    left = {}
    for r in range(h):
        for c in range(w):
            v = gi[r][c]
            if v != bgc:
                cnt[v] += 1
                left[v] = min(left.get(v, c), c)
    if not cnt:
        raise ValueError("no objects")
    m = max(cnt.values())
    big = sorted([v for v in cnt if cnt[v] == m], key=lambda v: left[v])
    go = [[v for v in big] for _ in range(m)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared by the golf entry): the unit is a COLOUR — every
    foreground colour is one object, so its size is simply how many cells
    carry it. Keep the colours whose count is the maximum m, ordered by the
    first column they appear in (a column-major scan). Each such colour
    becomes a stripe of m cells; stripes are laid out as rows, then the block
    is diagonally mirrored (transpose) so each stripe stands as a column.
    The golf entry also builds rows and transposes them, so the layout agrees
    with the maker; only the unit (colour count instead of connected
    component) and the order (column-major first appearance) are read
    differently, and both are taken from I alone."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        # every demonstration is drawn on the same background as I
        cand = votes.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand

    # column-major scan: first appearance order and counts per colour
    order = []
    cnt = Counter()
    for c in range(wi):
        for r in range(hi):
            v = int(I[r, c])
            if v == bgc:
                continue
            cnt[v] += 1
            if v not in order:
                order.append(v)
    m = max(cnt.values())
    stripe_colors = [v for v in order if cnt[v] == m]
    n = len(stripe_colors)

    ops, sels = [], []
    if m == 1 and n == 1:
        col = stripe_colors[0]
        pos = next((r, c) for c in range(wi) for r in range(hi) if I[r, c] == col)
        ops.append(33); sels.append([pos[0], pos[1], 0, 0])  # crop to that single cell
        ops.append(34); sels.append([0, 0, 0, 0])
        return ops, sels

    s = max(n, m)
    if s > hi or s > wi:
        # whole s x s square canvas the mirror needs
        ops.append(33); sels.append([0, 0, s - 1, s - 1])

    # one horizontal stripe (full row segment of length m) per kept colour
    for i, col in enumerate(stripe_colors):
        ops.append(int(col)); sels.append([i, 0, 0, m - 1])

    # diagonal mirror part 1: quarter turn CW of the full square region
    ops.append(25); sels.append([0, 0, s - 1, s - 1])
    # keep the turned block: rows 0..m-1, cols s-n..s-1 (full rectangle)
    ops.append(33); sels.append([0, s - n, m - 1, n - 1])
    # diagonal mirror part 2: reflect the full block left<->right
    if n > 1:
        ops.append(26); sels.append([0, 0, m - 1, n - 1])

    ops.append(34); sels.append([0, 0, m - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3325580"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3325580"
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
                                f"for task a3325580"
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
                    f"Failed to build a complete episode for task a3325580 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3325580-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
