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

# The golf entry and the maker read the task the same way. The clean picture is
# symmetric about a diagonal (it is the max of the grid and its transpose). One
# noise colour was stamped on top as blobs. Each blob is repaired from its mirror
# image, and the diagonal is then painted in its corner colour. The only real
# difference is grouping. The golf entry handles the diagonal as a separate
# whole-line step at the end, so we do the same: first the off-axis blobs take
# their mirror colours, then one Color op paints the noise cells on the axis.
# Everything else agrees, so the rest is unchanged.

ROTS = [0, 1, 2, 3]  # 0/2 -> main-diagonal symmetry, 1/3 -> anti-diagonal symmetry


def sample_colors(num_examples=None) -> dict:
    noisec = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    main = random.choice([0, 2])
    anti = random.choice([1, 3])
    ex = [{"rot": main}, {"rot": anti}][:n_ex]
    ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ex))]
    random.shuffle(ex)
    plan = ex + [dict(random.choice(ex))]
    return {"noisec": noisec, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    t = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * t))))


def generate(diff_lb, diff_ub, max_h, max_w, noisec, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    top = max(10, min(30, max_h, max_w))
    cols = [c for c in range(10) if c != noisec]
    while True:
        d = _unifint(diff_lb, diff_ub, (10, top))
        h = w = d
        nsl = _unifint(diff_lb, diff_ub, (2, min(9, h // 2)))
        slopes = [0] + sorted(random.sample(range(1, h - 1), nsl - 1))
        ccols = random.sample(cols, nsl)
        gi = np.full((h, w), -1, dtype=int)
        for col, hd in zip(ccols, slopes):
            s = hd / w
            for i in range(h):
                for j in range(w):
                    if s * j <= i:
                        gi[i, j] = col
        dcol = ccols[-2]
        for k in range(d):
            gi[k, k] = dcol
        for i in range(h):
            for j in range(i, w):
                gi[j, i] = gi[i, j]
        go = gi.copy()
        ndist = _unifint(diff_lb, diff_ub, (1, (h * w) // 15))
        tr = succ = 0
        while tr < 10 * ndist and succ < ndist:
            tr += 1
            oh, ow = random.randint(1, 5), random.randint(1, 5)
            if h - oh - 1 < 1 or w - ow - 1 < 1:
                continue
            li = random.randint(1, h - oh - 1)
            lj = random.randint(1, w - ow - 1)
            g2 = gi.copy()
            g2[li:li + oh, lj:lj + ow] = noisec
            if dcol not in [g2[k, k] for k in range(d)]:
                continue
            m = (g2 == noisec)
            off = m & m.T
            np.fill_diagonal(off, False)
            if off.any():
                continue
            succ += 1
            gi = g2
        if (gi != go).any():
            break
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Rule: the clean picture is symmetric about a diagonal (main or anti). Every
    # cell on that diagonal carries the corner colour. One noise colour was stamped
    # in blobs that never cover a cell together with its mirror partner. Repair is
    # "overlay the picture with its own reflection":
    #   1) each off-axis noise blob takes its mirror partner's colours;
    #   2) the noise cells on the axis are repainted as one line in the axis colour.
    # The noise colour comes from the demonstrations (the colour they repair), and
    # the axis from I alone. O is never consulted.
    I = np.asarray(I, dtype=int)
    n, m = I.shape
    mirrors = {
        "main": (lambda r, c: (c, r), int(I[0, 0])),
        "anti": (lambda r, c: (n - 1 - c, n - 1 - r), int(I[0, n - 1])),
    }

    def consistent(f, dcol, col):
        for r in range(n):
            for c in range(n):
                rr, cc = f(r, c)
                a, b = I[r, c], I[rr, cc]
                if (rr, cc) == (r, c):
                    if a != col and a != dcol:
                        return False
                elif a != col and b != col and a != b:
                    return False
        return True

    # noise colour: the colour that the demonstrations repair
    cand = None
    if examples:
        seen = {}
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            eo = np.asarray(ex[1], dtype=int)
            if ei.shape != eo.shape:
                continue
            for v in ei[ei != eo].tolist():
                seen[v] = seen.get(v, 0) + 1
        if seen:
            cand = [max(seen, key=seen.get)]
    cnt = {int(v): int((I == v).sum()) for v in np.unique(I)}
    order = (cand or []) + sorted(cnt, key=lambda v: cnt[v])
    best = None
    for col in order:
        for name, (f, dcol) in mirrors.items():
            if consistent(f, dcol, col):
                best = (name, col)
                break
        if best:
            break
    if best is None:
        return [34], [[0, 0, n - 1, m - 1]]
    name, noisec = best
    f, dcol = mirrors[name]

    off_fix, axis = {}, []
    for r in range(n):
        for c in range(n):
            if I[r, c] != noisec:
                continue
            rr, cc = f(r, c)
            if (rr, cc) == (r, c):
                if dcol != noisec:
                    axis.append((r, c))
            elif I[rr, cc] != noisec:
                off_fix[(r, c)] = int(I[rr, cc])

    # 1) reflect: each off-axis noise blob (4-connected) takes its mirror image's colours
    seen, blobs = set(), []
    for p in sorted(off_fix):
        if p in seen:
            continue
        comp, stack = [], [p]
        seen.add(p)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in off_fix and q not in seen:
                    seen.add(q)
                    stack.append(q)
        blobs.append(sorted(comp))
    ops, sels = [], []
    for comp in blobs:
        bycol = {}
        for p in comp:
            bycol.setdefault(off_fix[p], []).append(p)
        for tc in sorted(bycol, key=lambda k: min(bycol[k])):
            ops.append(tc)
            sels.append(sel_of(bycol[tc]))
    # 2) restore the symmetry axis as one line in the axis (corner) colour
    if axis:
        ops.append(dcol)
        sels.append(sel_of(sorted(axis)))
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
