"""
ARC Task: fafffa47 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]          # 2 is the rule's own colour
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, mirrored=None, **kw) -> dict:
    def uni(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = max(a, min(a + int((b - a) * lb), b))
        hi = max(lo, min(a + int((b - a) * ub), b))
        return random.randint(lo, hi)

    if mirrored is None:
        mirrored = random.choice(list(range(2))) == 1

    # input is hconcat(A, B) with A, B of shape (h, w); dmirrored when mirrored
    if mirrored:
        hcap, wcap = max_w, max_h // 2
    else:
        hcap, wcap = max_h, max_w // 2
    hcap = max(2, min(30, hcap))
    wcap = max(2, min(14, wcap))

    h = uni(diff_lb, diff_ub, (2, hcap))
    w = uni(diff_lb, diff_ub, (2, wcap))
    n = h * w

    numadev = uni(diff_lb, diff_ub, (0, n // 2))
    numbdev = uni(diff_lb, diff_ub, (0, n // 2))
    numa = random.choice((numadev, n - numadev))
    numb = random.choice((numadev, n - numbdev))
    numa = min(max(1, numa), n - 1)
    numb = min(max(1, numb), n - 1)

    inds = [(r, c) for r in range(h) for c in range(w)]
    aset = set(random.sample(inds, numa))
    bset = set(random.sample(inds, numb))

    A = [[acol if (r, c) in aset else bgc for c in range(w)] for r in range(h)]
    B = [[bcol if (r, c) in bset else bgc for c in range(w)] for r in range(h)]
    gi = [A[r] + B[r] for r in range(h)]
    go = [[2 if ((r, c) not in aset and (r, c) not in bset) else bgc
           for c in range(w)] for r in range(h)]

    if mirrored:
        gi = [list(x) for x in zip(*gi)]
        go = [list(x) for x in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Two half-grids lie side by side (or stacked). Keep one half's frame:
    a cell blank in BOTH halves becomes 2, everything else background.

    The rule is fixed; the route through it is not. Which half to work in,
    whether to mark only the doubly-blank cells or fill this half's blanks and
    knock the covered ones back out, whether to take a set in one sweep / line
    by line / blob by blob, which end to start from, brush or paint bucket, and
    whether to shrink the canvas before or after the tidying up -- each of those
    is read off the input grid, so one input always gives one route, and the
    variety comes from the inputs differing.
    """
    I = np.asarray(I, dtype=int)          # O is never read: the route comes from I
    hi, wi = I.shape

    def ncol(g):
        return len(set(g.flatten().tolist()))

    # ── the rule's own split test, measured from I alone ──────────────────────
    vertical = (ncol(I[:hi // 2]) == 2 and ncol(I[hi // 2:]) == 2) and not (
        ncol(I[:, :wi // 2]) == 2 and ncol(I[:, wi // 2:]) == 2)
    if vertical:
        H1, W1 = hi // 2, wi
        origins = [(0, 0), (H1, 0)]
    else:
        H1, W1 = hi, wi // 2
        origins = [(0, 0), (0, W1)]
    halves = [I[r:r + H1, c:c + W1] for (r, c) in origins]

    # the colour the two palettes share is the background
    bgc = sorted(set(halves[0].flatten().tolist())
                 & set(halves[1].flatten().tolist()))[0]
    marks = [[(r, c) for r in range(H1) for c in range(W1) if halves[k][r, c] != bgc]
             for k in (0, 1)]

    # ── work in the half with less to rub out ────────────────────────────────
    keep = 1 if len(marks[1]) < len(marks[0]) else 0
    K, X = halves[keep], halves[1 - keep]
    blanks = [(r, c) for r in range(H1) for c in range(W1) if K[r, c] == bgc]
    doubly = [rc for rc in blanks if X[rc[0], rc[1]] == bgc]
    covered = [rc for rc in blanks if X[rc[0], rc[1]] != bgc]

    # ── sweep geometry, also read off the input ──────────────────────────────
    along_rows = W1 >= H1                      # sweep across the half's long lines
    span = (H1 - 1) if along_rows else (W1 - 1)

    def okey(rc):
        return (rc[0], rc[1]) if along_rows else (rc[1], rc[0])

    def backwards(cells):
        # start where the cells pile up: past the middle means sweep from the far end
        idx = [(rc[0] if along_rows else rc[1]) for rc in cells]
        return bool(idx) and sum(idx) * 2 > span * len(idx)

    def blobs(cells, diagonal):
        rest = set(cells)
        out = []
        while rest:
            seed = min(rest, key=okey)
            rest.discard(seed)
            stack, comp = [seed], []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (dr or dc) and (diagonal or dr == 0 or dc == 0):
                            p = (r + dr, c + dc)
                            if p in rest:
                                rest.discard(p)
                                stack.append(p)
            out.append(sorted(comp, key=okey))
        out.sort(key=lambda comp: okey(comp[0]))
        return out

    def grouping(cells):
        """One go, line by line, or one shape at a time -- by how thick they lie."""
        if not cells:
            return "none", []
        dens = len(cells) / float(H1 * W1)
        if dens >= 0.55:                       # they are everywhere: one selection
            return "sweep", [sorted(cells, key=okey)]
        if dens >= 0.2:                        # thick enough to go line by line
            lines = {}
            for rc in cells:
                lines.setdefault(rc[0] if along_rows else rc[1], []).append(rc)
            return "lines", [sorted(lines[k], key=okey)
                             for k in sorted(lines, reverse=backwards(cells))]
        cs = blobs(cells, True)                # sparse: one shape at a time
        if backwards(cells):
            cs.reverse()
        return "blobs", cs

    ops, sels = [], []
    off = [origins[keep][0], origins[keep][1]]     # where the kept half sits now

    def paint(cells, colour):
        for g in grouping(cells)[1]:
            ops.append(int(colour))
            sels.append(sel_of([(r + off[0], c + off[1]) for (r, c) in g]))

    def erase(cells):
        """Send this half's own marks back to background."""
        mode, gs = grouping(cells)
        if mode == "blobs":
            chunks = blobs(cells, False)           # the bucket walks 4-connected
            biggest = max(len(c) for c in chunks) if chunks else 0
            if 3 <= biggest <= 120:                # real blobs: use the paint bucket
                if backwards(cells):
                    chunks.reverse()
                for comp in chunks:
                    r, c = comp[0]
                    ops.append(10 + int(bgc))      # FloodFill<bgc> from one seed cell
                    sels.append([r + off[0], c + off[1], 0, 0])
                return
        for g in gs:
            ops.append(int(bgc))
            sels.append(sel_of([(r + off[0], c + off[1]) for (r, c) in g]))

    def crop():
        ops.append(33)              # bbox IS exactly the kept half's full rectangle
        sels.append([off[0], off[1], H1 - 1, W1 - 1])
        off[0] = off[1] = 0

    if doubly and len(covered) < len(doubly):
        # the other half barely covers anything: fill this half's blanks with 2,
        # then knock back out the few the other half does cover
        paint(blanks, 2)
        if len(covered) <= len(marks[keep]):
            paint(covered, bgc)                    # the knock-outs, then shrink
            crop()
            erase(marks[keep])
        else:
            erase(marks[keep])                     # the marks first, then knock-outs
            paint(covered, bgc)
            crop()
    else:
        # straight reading: only the cells blank in both halves become 2
        paint(doubly, 2)
        if len(marks[keep]) * 2 >= H1 * W1:
            crop()                                 # lots to rub out: shrink first
            erase(marks[keep])
        else:
            erase(marks[keep])
            crop()

    ops.append(34)
    sels.append([0, 0, H1 - 1, W1 - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task fafffa47"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fafffa47"
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
                                f"for task fafffa47"
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
                    f"Failed to build a complete episode for task fafffa47 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fafffa47-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
