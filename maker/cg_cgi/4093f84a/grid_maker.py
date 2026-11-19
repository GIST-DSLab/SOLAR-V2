"""
ARC Task: 4093f84a (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, barc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, dotc=None,
             mirror=None, **kw) -> dict:
    _r = random
    if bgc is None or barc is None or dotc is None:
        bgc, barc, dotc = _r.sample(range(10), 3)
    if mirror is None:
        mirror = _r.choice((True, False))
    hb, wb = (max_w, max_h) if mirror else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (7, max(7, min(30, hb))))
    w = unifint(diff_lb, diff_ub, (7, max(7, min(30, wb))))
    loci1, loci2 = sorted(_r.sample(interval(2, h - 2, 1), 2))
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = _r.sample(opts, num1)
    locs2 = _r.sample(opts, num2)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = _r.sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = _r.sample(interval(loci2 + 1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Gravity reading: the solid band attracts the scattered dots.  In every line
    perpendicular to the band, each run of dots (nearest first) turns into band
    colour and slides (Move) toward the band until it rests against the band or
    the runs that already landed, so the dots stack into one bar.  Measured
    from I only; O is used only for the submit bbox.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: the colour living only in complete, contiguous lines
    barc, vertical = None, False
    for c in colors:
        rs = [r for r in range(hi) if bool((Ia[r] == c).any())]
        if rs and len(rs) == rs[-1] - rs[0] + 1 and all(bool((Ia[r] == c).all()) for r in rs):
            barc, vertical = c, False
            break
    if barc is None:
        for c in colors:
            cs = [j for j in range(wi) if bool((Ia[:, j] == c).any())]
            if cs and len(cs) == cs[-1] - cs[0] + 1 and all(bool((Ia[:, j] == c).all()) for j in cs):
                barc, vertical = c, True
                break

    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    cnt = Counter(Ia.flatten().tolist())
    rest = [c for c in colors if c != barc]
    dotc = min(rest, key=lambda c: cnt[c])
    bgc = [c for c in rest if c != dotc][0]

    def real(cells):  # canonical (row, col) -> real grid cells
        return [(c, r) for r, c in cells] if vertical else list(cells)

    # canonical direction toward the band -> real Move op
    def move_op(toward_up):
        if vertical:
            return 23 if toward_up else 22   # canonical rows are real columns
        return 20 if toward_up else 21

    def runs(rows):
        out = []
        for r in rows:
            if out and abs(r - out[-1][-1]) == 1:
                out[-1].append(r)
            else:
                out.append([r])
        return out

    ops, sels = [], []
    for c in range(w):
        for side in (0, 1):
            if side == 0:   # above: nearest-first = descending rows; they fall down
                dots = [r for r in range(r1 - 1, -1, -1) if A[r, c] == dotc]
                edge, step = r1 - 1, -1
                op = move_op(False)
            else:           # below: nearest-first = ascending rows; they rise up
                dots = [r for r in range(r2 + 1, h) if A[r, c] == dotc]
                edge, step = r2 + 1, 1
                op = move_op(True)
            if not dots:
                continue
            if barc == 0:
                # a 0-coloured bar cannot be grabbed by Move: grow the bar in
                # place and erase the dots it did not swallow
                k = len(dots)
                bar = [(edge + step * i, c) for i in range(k)]
                ops.append(0); sels.append(sel_of(real(bar)))
                left = [(r, c) for r in dots if (r, c) not in set(bar)]
                if left:
                    ops.append(int(bgc)); sels.append(sel_of(real(left)))
                continue
            nxt = edge          # next free slot next to band / landed dots
            zeros = set()       # cells of this line currently reading 0
            for run in runs(dots):
                dist = abs(run[0] - nxt)
                src = [(r, c) for r in run]
                # the run becomes band material, then is pulled onto the bar
                ops.append(int(barc)); sels.append(sel_of(real(src)))
                cur = list(src)
                d = -step       # canonical row delta toward the band
                for i in range(dist):
                    ops.append(op)
                    sels.append(sel_of(real(cur)) if i == 0 else sel_of([]))
                    cur = [(r + d, cc) for r, cc in cur]
                # the vacated original footprint reads 0 after the slide
                zeros = (zeros - set(cur)) | (set(src) - set(cur))
                nxt = nxt + step * len(run)
            # line settled: restore background on the vacated cells still at 0
            if bgc != 0 and zeros:
                ops.append(int(bgc)); sels.append(sel_of(real(sorted(zeros))))

    ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
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
                                f"for task 4093f84a"
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
                    f"Failed to build a complete episode for task 4093f84a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4093f84a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
