"""
ARC Task: 73251a56 (RE-ARC) — LLM-generated grid_maker
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
    # Every colour (stripes, noise) is sampled per instance by the generator; the rule
    # (restore diagonal symmetry, noise colour = rarest colour explaining every asymmetry)
    # does not depend on any fixed colour role.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        return random.randint(int(a + (b - a) * lb), int(a + (b - a) * ub))

    cols = list(range(10))
    dmax = max(10, min(30, max_h, max_w))
    while True:
        d = unifint(diff_lb, diff_ub, (10, dmax))
        h = w = d
        noisec = random.choice(cols)
        nsl = unifint(diff_lb, diff_ub, (2, min(9, h // 2)))
        slopes = [0] + sorted(random.sample(range(1, h - 1), nsl - 1))
        ccols = random.sample(cols, nsl)
        gi = [[-1] * w for _ in range(h)]
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            for i in range(h):
                for j in range(w):
                    if slope * j <= i:
                        gi[i][j] = col
        for k in range(d):
            gi[k][k] = ccols[-2]
        for i in range(h):
            for j in range(w):
                if j >= i:
                    gi[j][i] = gi[i][j]
        ln = {(k, k) for k in range(d)}

        def ok(g):
            if not any(g[k][k] == ccols[-2] for k in range(d)):
                return False
            nz = {(i, j) for i in range(h) for j in range(w) if g[i][j] == noisec}
            return len({p for p in nz if (p[1], p[0]) in nz} - ln) == 0

        ndist = unifint(diff_lb, diff_ub, (1, (h * w) // 15))
        go = [row[:] for row in gi]
        tr = succ = 0
        maxtr = 10 * ndist
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = random.randint(1, 5)
            ow = random.randint(1, 5)
            if h - oh - 1 < 1 or w - ow - 1 < 1:
                continue
            loci = random.randint(1, h - oh - 1)
            locj = random.randint(1, w - ow - 1)
            g2 = [row[:] for row in gi]
            for i in range(loci, loci + oh):
                for j in range(locj, locj + ow):
                    g2[i][j] = noisec
            if ok(g2):
                succ += 1
                gi = g2
        if gi != go:
            break
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(np.array(gi), -k).tolist()
    go = np.rot90(np.array(go), -k).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Rule: the clean grid is symmetric about one of its diagonals (main or anti,
    # depending on orientation) and that diagonal is one solid colour (its corner colour).
    # Rectangular blobs of a single noise colour were stamped on it, never covering a
    # cell together with its mirror partner. Repair: each noise cell takes the colour of
    # its mirror partner across the axis (or the diagonal colour when on the axis).
    # Axis and noise colour are found from I alone: the (axis, colour) pair explaining
    # every asymmetry, preferring the rarest colour.  O is never consulted.
    I = np.asarray(I, dtype=int)
    n, m = I.shape
    mirrors = {
        "main": (lambda r, c: (c, r), I[0, 0]),
        "anti": (lambda r, c: (n - 1 - c, n - 1 - r), I[0, n - 1]),
    }
    cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
    best = None
    for name, (f, dcol) in mirrors.items():
        for col in sorted(cnt, key=lambda v: cnt[v]):
            good = True
            for r in range(n):
                for c in range(n):
                    rr, cc = f(r, c)
                    a, b = I[r, c], I[rr, cc]
                    if (rr, cc) == (r, c):
                        if a != col and a != dcol:
                            good = False
                            break
                    elif a != col and b != col and a != b:
                        good = False
                        break
                if not good:
                    break
            if good:
                if best is None or cnt[col] < cnt[best[1]]:
                    best = (name, col)
                break
    name, noisec = best
    f, dcol = mirrors[name]

    # noise cells needing repair, with the colour read from their mirror partner in I
    fix = {}
    for r in range(n):
        for c in range(n):
            if I[r, c] != noisec:
                continue
            rr, cc = f(r, c)
            tgt = int(dcol) if (rr, cc) == (r, c) else int(I[rr, cc])
            if tgt != noisec:
                fix[(r, c)] = tgt

    # group into noise blobs (4-connected components of cells to repair)
    seen, blobs = set(), []
    for p in sorted(fix):
        if p in seen:
            continue
        comp, stack = [], [p]
        seen.add(p)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in fix and q not in seen:
                    seen.add(q)
                    stack.append(q)
        blobs.append(sorted(comp))

    ops, sels = [], []
    for comp in blobs:
        bycol = {}
        for p in comp:
            bycol.setdefault(fix[p], []).append(p)
        for tc in sorted(bycol, key=lambda k: min(bycol[k])):
            ops.append(tc)
            sels.append(sel_of(bycol[tc]))
    ops.append(34)
    sels.append([0, 0, n - 1, m - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 73251a56"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 73251a56"
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
                                f"for task 73251a56"
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
                    f"Failed to build a complete episode for task 73251a56 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"73251a56-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
