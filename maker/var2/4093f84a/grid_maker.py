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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, barc, dotc = random.sample(cols, 3)
    # the only discrete structural variant: the band runs in rows or in columns
    VARIANTS = [{"mirror": False}, {"mirror": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, barc=None, dotc=None, mirror=None) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    # after dmirror the grid is transposed -> swap the caps while building
    hub = max_w if mirror else max_h
    wub = max_h if mirror else max_w
    h = unifint(diff_lb, diff_ub, (7, hub))
    w = unifint(diff_lb, diff_ub, (7, wub))
    loci1, loci2 = sorted(sample(interval(2, h - 2, 1), 2))
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = sample(opts, num1)
    locs2 = sample(opts, num2)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = sample(interval(loci2 + 1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule, read off I alone: a solid band of `barc` lines cuts the grid in two.
    In each margin every line perpendicular to the band carries some scattered
    `dotc` cells; those dots collapse into ONE bar of that many cells grown out
    of the band's edge, and any dot the bar does not swallow disappears (-> bgc).

    Only Color ops are used, so a role that happens to be colour 0 is painted
    explicitly and travels fine (nothing here is carried by Copy/Paste/Move).

    The ROUTE (which margin first, which way to sweep, whether to wipe a line
    clean and draw the bar on top of it or draw the bar and then rub the strays
    out, whether to lay the tidy bars of a margin down in one stroke) is decided
    from I only — the same input always gives the same route, and two different
    inputs generally give different ones. O is never inspected.
    """
    Ia = np.asarray(I, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # --- band colour: the colour living ONLY in complete, contiguous lines
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

    # canonical frame: the band is made of ROWS, margins are above / below
    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if bool((A[r] == barc).all())]
    r1, r2 = band[0], band[-1]

    # of the two remaining colours the sparse one is the dots, the other the bg
    cnt = {}
    for v in Ia.flatten().tolist():
        cnt[v] = cnt.get(v, 0) + 1
    rest = [c for c in colors if c != barc]
    dotc = min(rest, key=lambda c: cnt[c])
    bgc = [c for c in rest if c != dotc][0] if len(rest) > 1 else dotc

    def real(cells):
        """canonical (row, col) -> real grid coordinates"""
        return [(c, r) for (r, c) in cells] if vertical else [(r, c) for (r, c) in cells]

    # --- read every dotted line: its bar, its strays, its whole margin strip
    lines = []
    for c in range(w):
        for side in (0, 1):
            rows = list(range(0, r1)) if side == 0 else list(range(r2 + 1, h))
            dots = [r for r in rows if A[r, c] == dotc]
            k = len(dots)
            if k == 0:
                continue
            if side == 0:
                bar = list(range(r1 - k, r1))
                stray = sorted(r for r in dots if r < r1 - k)
            else:
                bar = list(range(r2 + 1, r2 + 1 + k))
                stray = sorted(r for r in dots if r > r2 + k)
            runs = []
            for r in stray:
                if runs and r == runs[-1][-1] + 1:
                    runs[-1].append(r)
                else:
                    runs.append([r])
            lines.append({
                "side": side, "col": c, "k": k,
                "bar":  [(r, c) for r in bar],
                "strip": [(r, c) for r in rows],
                "runs": [[(r, c) for r in run] for run in runs],
            })

    tops = [t for t in lines if t["side"] == 0]
    bots = [t for t in lines if t["side"] == 1]

    # ── route choices, all read off I ────────────────────────────────────────
    # sweep towards the emptier end: start where the dotted lines are crowded
    left_load = sum(1 for t in lines if 2 * t["col"] < w)
    forward = left_load >= len(lines) - left_load          # True: low col -> high col
    # begin with the busier margin; if both are equally busy, with the deeper one
    if len(tops) != len(bots):
        top_first = len(tops) > len(bots)
    else:
        top_first = r1 >= h - 1 - r2
    # when the two margins are dotted on largely the SAME lines, a line is the
    # natural unit of work (do both of its ends, then move on); otherwise finish
    # one whole margin before starting the other
    cols_t = {t["col"] for t in tops}
    cols_b = {t["col"] for t in bots}
    column_major = bool(cols_t) and bool(cols_b) and \
        2 * len(cols_t & cols_b) >= min(len(cols_t), len(cols_b))
    # lay the tidy bars of a margin down in one stroke? decided by the colours
    group_tidy = dotc > bgc

    ops, sels = [], []

    def draw(t):
        if len(t["runs"]) >= 2:
            # a messy line: sweep the whole strip back to background, then draw
            # the bar on top of the clean strip
            ops.append(int(bgc)); sels.append(sel_of(real(t["strip"])))
            ops.append(int(barc)); sels.append(sel_of(real(t["bar"])))
        else:
            # at most one stray run: draw the bar, then rub that run out
            ops.append(int(barc)); sels.append(sel_of(real(t["bar"])))
            for run in t["runs"]:
                ops.append(int(bgc)); sels.append(sel_of(real(run)))

    def swept(ts):
        return sorted(ts, key=lambda t: t["col"] if forward else -t["col"])

    if column_major:
        for c in sorted({t["col"] for t in lines}, reverse=not forward):
            here = [t for t in lines if t["col"] == c]
            here.sort(key=lambda t: 0 if (t["side"] == 0) == top_first else 1)
            for t in here:
                draw(t)
    else:
        for margin in ([tops, bots] if top_first else [bots, tops]):
            if not margin:
                continue
            tidy = [t for t in margin if not t["runs"]]
            messy = [t for t in margin if t["runs"]]
            if group_tidy and len(tidy) >= 2:
                cells = []
                for t in swept(tidy):
                    cells.extend(t["bar"])
                ops.append(int(barc)); sels.append(sel_of(real(cells)))
                for t in swept(messy):
                    draw(t)
            else:
                for t in swept(margin):
                    draw(t)

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
