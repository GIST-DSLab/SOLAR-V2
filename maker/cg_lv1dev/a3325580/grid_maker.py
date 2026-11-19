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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role; object colours are
    # sampled per instance (the rule never depends on which colours they are).
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    while True:
        h = _unifint(diff_lb, diff_ub, (10, max_h))
        w = _unifint(diff_lb, diff_ub, (10, max_w))
        nobjs = _unifint(diff_lb, diff_ub, (1, 9))
        remcols = [c for c in range(10) if c != bgc]
        ccols = random.sample(remcols, nobjs)
        gi = [[bgc] * w for _ in range(h)]
        lmocc = set()
        inds = {(i, j) for i in range(h) for j in range(w)}
        succ = 0
        tr = 0
        maxtr = 4 * nobjs
        seenobjs = set()
        mxncells = random.randint(nobjs + 1, 30)

        def dnb(cell):
            i, j = cell
            return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(1, 6)
            ow = random.randint(1, 6)
            while oh * ow < mxncells:
                oh = random.randint(1, 6)
                ow = random.randint(1, 6)
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            ncells = _unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
            ncells = _unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
            sp = random.choice(sorted(bounds))
            obj = {sp}
            for _ in range(ncells - 1):
                nb = set()
                for cell in obj:
                    nb |= dnb(cell)
                cand = (bounds - obj) & nb
                obj.add(random.choice(sorted(cand)))
            fobj = frozenset(obj)
            if fobj in seenobjs:
                continue
            seenobjs.add(fobj)
            mi = min(i for i, _ in obj)
            mj = min(j for _, j in obj)
            obj = {(i - mi, j - mj) for i, j in obj}
            oh = max(i for i, _ in obj) + 1
            ow = max(j for _, j in obj) + 1
            cands = [(i, j) for (i, j) in inds
                     if i <= h - oh and j <= w - ow and j not in lmocc]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            plcd = {(i + loc[0], j + loc[1]) for i, j in obj}
            if plcd.issubset(inds):
                nbs = set()
                for cell in plcd:
                    nbs |= dnb(cell)
                inds = (inds - plcd) - nbs
                for i, j in plcd:
                    gi[i][j] = ccols[succ]
                succ += 1
                lmocc.add(loc[1])
        if succ == 0:
            continue
        # objects = per-colour components (each colour is used by one object)
        G = np.array(gi)
        sizes = {}
        left = {}
        for c in set(G.flatten().tolist()) - {bgc}:
            pos = np.argwhere(G == c)
            sizes[c] = len(pos)
            left[c] = int(pos[:, 1].min())
        mx = max(sizes.values())
        cols = sorted([c for c in sizes if sizes[c] == mx], key=lambda c: left[c])
        go = [[c for c in cols] for _ in range(mx)]
        return {"input": gi, "output": go}


def _background(I, examples):
    # The background is the colour every input of the episode is painted on.
    grids = [np.asarray(I, dtype=int)]
    if examples:
        grids += [np.asarray(ex[0], dtype=int) for ex in examples]
    common = set(range(10))
    total = Counter()
    for g in grids:
        common &= set(g.flatten().tolist())
        total.update(g.flatten().tolist())
    pool = common if common else set(total)
    return max(pool, key=lambda c: total[c])


def derive_operations(I, O, examples=None):
    """Reading (agrees with the golf entry, not with the maker's stripes-then-mirror):
    count the cells of every non-background colour; the colours reaching the
    maximum count m are the answer columns, in the order they are first met when
    the grid is read column by column (= by leftmost column). The answer is
    built directly as m rows by n columns. Each chosen colour gets its own full
    output column. Nothing is laid out sideways and then turned or mirrored.
    Every quantity comes from I and the background agreed by the examples."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = _background(I, examples)

    # colour counts and column-major first appearance
    counts = Counter(I.flatten().tolist())
    order = []
    for c in range(wi):
        for r in range(hi):
            v = int(I[r, c])
            if v != bgc and v not in order:
                order.append(v)
    m = max(counts[v] for v in order)
    stripe_colors = [v for v in order if counts[v] == m]
    n = len(stripe_colors)

    ops, sels = [], []
    # Resize the canvas to the answer frame: m tall (largest object size),
    # n wide (number of largest objects). Full rectangle intended.
    ops.append(33); sels.append([0, 0, m - 1, n - 1])

    # Mirror ARCLE's transparent crop so later no-op checks use the current grid.
    cur = np.zeros((m, n), dtype=int)
    for r in range(min(m, hi)):
        for c in range(min(n, wi)):
            cur[r, c] = I[r, c]

    # One full column per largest object, in leftmost order.
    for j, col in enumerate(stripe_colors):
        cells = [(r, j) for r in range(m)]
        if all(cur[r, j] == col for r in range(m)):
            continue  # column already holds this colour; painting would be a no-op
        ops.append(int(col)); sels.append(sel_of(cells))
        for r in range(m):
            cur[r, j] = col

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
