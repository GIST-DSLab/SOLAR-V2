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

ROTS = [0, 1, 2, 3]  # number of CW quarter-turns applied to the clean grid


def sample_colors(num_examples=None) -> dict:
    noisec = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": k} for k in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        # make sure both diagonal orientations (main: 0/2, anti: 1/3) appear
        ex = [{"rot": random.choice([0, 2])}, {"rot": random.choice([1, 3])}]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - 2)]
        ex = ex[:n_ex]
        random.shuffle(ex)
    plan = ex + [dict(random.choice(ex))]
    return {"noisec": noisec, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, noisec=None, rot=None, **kw) -> dict:
    if noisec is None:
        noisec = random.randrange(10)
    if rot is None:
        rot = random.choice(ROTS)
    dmax = max(10, min(30, max_h, max_w))
    while True:
        d = _unifint(diff_lb, diff_ub, (10, dmax))
        h = w = d
        nsl = _unifint(diff_lb, diff_ub, (2, min(9, h // 2)))
        slopes = [0] + sorted(random.sample(range(1, h - 1), nsl - 1))
        ccols = random.sample(range(10), nsl)
        gi = [[-1] * w for _ in range(h)]
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            for i in range(h):
                for j in range(w):
                    if slope * j <= i:
                        gi[i][j] = col
        for i in range(d):
            gi[i][i] = ccols[-2]
        for i in range(h):
            for j in range(i, w):
                gi[j][i] = gi[i][j]
        go = [row[:] for row in gi]
        ln = {(i, i) for i in range(d)}

        def ok(g):
            if not any(g[i][i] == ccols[-2] for i in range(d)):
                return False
            nz = {(i, j) for i in range(h) for j in range(w) if g[i][j] == noisec}
            return len((nz & {(j, i) for i, j in nz}) - ln) == 0

        ndist = _unifint(diff_lb, diff_ub, (1, (h * w) // 15))
        tr = succ = 0
        while tr < 10 * ndist and succ < ndist:
            tr += 1
            oh, ow = random.randint(1, 5), random.randint(1, 5)
            if h - oh - 1 < 1 or w - ow - 1 < 1:
                continue
            li = random.randint(1, h - oh - 1)
            lj = random.randint(1, w - ow - 1)
            g2 = [row[:] for row in gi]
            for a in range(li, li + oh):
                for b in range(lj, lj + ow):
                    g2[a][b] = noisec
            if ok(g2):
                succ += 1
                gi = g2
        if gi != go:
            break
    gi = np.rot90(np.array(gi), -rot).tolist()
    go = np.rot90(np.array(go), -rot).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): the clean picture is symmetric about a
    # diagonal, so every stamped noise cell is restored from its mirror partner —
    # row i is repaired from column i (main diagonal) or its anti-diagonal
    # counterpart.  A noise cell lying ON the axis has no distinct partner, so it
    # takes the colour of its predecessor along the axis in the grid already
    # repaired.  The repair runs ROW BY ROW, top to bottom: a row is the unit, and
    # within a row one Color op per partner colour.  Axis and noise colour come
    # from I alone (the (axis, colour) pair that explains every asymmetry,
    # preferring the rarest colour); O is never read.
    I = np.asarray(I, dtype=int)
    n, m = I.shape
    mirrors = {
        "main": (lambda r, c: (c, r), lambda r, c: (r - 1, c - 1)),
        "anti": (lambda r, c: (n - 1 - c, n - 1 - r), lambda r, c: (r - 1, c + 1)),
    }
    cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
    best = None
    for name, (f, _) in mirrors.items():
        dcol = I[0, 0] if name == "main" else I[0, n - 1]
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
    f, pred = mirrors[name]

    G = I.copy()
    ops, sels = [], []
    for r in range(n):
        bycol = {}
        for c in range(n):
            if I[r, c] != noisec:
                continue
            rr, cc = f(r, c)
            if (rr, cc) == (r, c):
                pr, pc = pred(r, c)
                tgt = int(G[pr, pc])   # predecessor along the axis, already repaired
            else:
                tgt = int(I[rr, cc])   # mirror partner, read from the input
            if tgt != noisec:
                bycol.setdefault(tgt, []).append((r, c))
        for tc in sorted(bycol, key=lambda k: min(bycol[k])):
            ops.append(tc)
            sels.append(sel_of(bycol[tc]))
            for p in bycol[tc]:
                G[p] = tc
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
