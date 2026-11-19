"""
ARC Task: 673ef223 (RE-ARC) — LLM-generated grid_maker
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
TRANSFORMS = ['identity', 'dmirror', 'cmirror', 'vmirror', 'hmirror', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]           # 4 is reserved for the dot caps
    bgc, barc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(TRANSFORMS):
        examples = [{'tf': t} for t in TRANSFORMS]
        examples += [{'tf': random.choice(TRANSFORMS)} for _ in range(n_ex - len(TRANSFORMS))]
        random.shuffle(examples)
    else:
        examples = [{'tf': t} for t in random.sample(TRANSFORMS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test orientation was demonstrated
    return {'bgc': bgc, 'barc': barc, 'dotc': dotc, 'instance_plan': plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, tf=None) -> dict:
    if tf is None:
        tf = random.choice(TRANSFORMS)
    swap = tf in ('dmirror', 'cmirror', 'rot90', 'rot270')   # these transpose the canvas
    hlim = max(5, max_w if swap else max_h)
    wlim = max(5, max_h if swap else max_w)
    h = unifint(diff_lb, diff_ub, (5, hlim))
    w = unifint(diff_lb, diff_ub, (5, wlim))
    barh = unifint(diff_lb, diff_ub, (2, (h - 1) // 2))
    ncells = unifint(diff_lb, diff_ub, (1, barh))
    sg = canvas(bgc, (barh, w))
    topsgi = fill(sg, barc, connect((0, 0), (barh - 1, 0)))
    botsgi = vmirror(topsgi)
    topsgo = tuple(e for e in topsgi)
    botsgo = tuple(e for e in botsgi)
    ilocs = random.sample(list(range(barh)), ncells)
    for k in ilocs:
        jloc = random.randint(2, w - 2)
        topsgi = fill(topsgi, dotc, {(k, jloc)})
        topsgo = fill(topsgo, 4, {(k, jloc)})
        topsgo = fill(topsgo, dotc, connect((k, 1), (k, jloc - 1)))
        botsgo = fill(botsgo, dotc, connect((k, 0), (k, w - 2)))
    outpi = (topsgi, botsgi)
    outpo = (topsgo, botsgo)
    rr = canvas(bgc, (1, w))
    while len(merge(outpi)) < h:
        idx = random.randint(0, len(outpi) - 1)
        outpi = outpi[:idx] + (rr,) + outpi[idx:]
        outpo = outpo[:idx] + (rr,) + outpo[idx:]
    gi = merge(outpi)
    go = merge(outpo)
    fmap = {'identity': identity, 'dmirror': dmirror, 'cmirror': cmirror, 'vmirror': vmirror,
            'hmirror': hmirror, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}
    fn = fmap[tf]
    return {'input': fn(gi), 'output': fn(go)}


def derive_operations(I, O):
    """
    Rule (read off I alone):
      * two straight bars of one colour sit on opposite grid borders;
      * single dots of another colour lie in the band of ONE of them (the dot bar);
      * each dot shoots a beam back to its own bar - every cell strictly between
        bar and dot becomes the dot colour - and the dot itself is capped with 4;
      * the twin bar echoes each dot: carried over by the offset between the two
        bars, the dot becomes a full line across the grid (every cell of that line
        that is not a bar cell).

    The ROUTE is measured from I, so one input always gives one route and the
    variety comes from the inputs differing:
      * order_mode  (bar length + dot count) : up the bar / down the bar / shortest beam first
      * phase_mode  (grid height + width)    : dot-by-dot / all beams then all echoes / echoes first
      * cap_first   (which border the dot bar sits on) : cap the dot then trail back, or trail then cap
      * copy        (three or more echoes)   : draw the first echo line, keep it, stamp the rest
    Nothing here looks at O.
    """
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]          # exact full-grid rectangle

    def bail():
        ops.append(34); sels.append(full)
        return ops, sels

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- connected components of non-background cells (4-conn, single colour) ----
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                rr, cc = stack.pop(); cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True; stack.append((nr, nc))
            comps.append((col, cells))
    if not comps:
        return bail()

    def on_border(p):
        return p[0] == 0 or p[1] == 0 or p[0] == hi - 1 or p[1] == wi - 1

    # bar colour: exactly two components, all cells on the border, neither a lone cell
    barc = None
    for col in sorted({c for c, _ in comps}):
        groups = [cells for c2, cells in comps if c2 == col]
        if (len(groups) == 2 and all(len(g) >= 2 for g in groups)
                and all(on_border(p) for g in groups for p in g)):
            barc = col; break
    if barc is None:
        barc = max(comps, key=lambda t: len(t[1]))[0]
    others = [c for c in sorted({c for c, _ in comps}) if c != barc]
    if not others:
        return bail()
    dotc = others[0]

    bars = [cells for col, cells in comps if col == barc]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    if len(bars) != 2 or not dots:
        return bail()

    # ---- "along" runs with the bars, "cross" crosses them (orientation-free) ----
    bar_rows = {p[0] for b in bars for p in b}
    bar_cols = {p[1] for b in bars for p in b}
    vertical = len(bar_rows) > len(bar_cols)      # bars run down columns
    if vertical:
        to_rc = lambda a, x: (a, x)
        along = lambda p: p[0]
        cross = lambda p: p[1]
        NA, NX = hi, wi
    else:
        to_rc = lambda a, x: (x, a)
        along = lambda p: p[1]
        cross = lambda p: p[0]
        NA, NX = wi, hi

    def run_cells(a, x1, x2):
        return [to_rc(a, x) for x in range(x1, x2 + 1)]

    dot_alongs = {along(d) for d in dots}
    dotbar = next((b for b in bars if {along(p) for p in b} & dot_alongs), bars[0])
    twinbar = bars[1] if dotbar is bars[0] else bars[0]
    cb = cross(dotbar[0])                          # border line the beams run back to
    cb2 = cross(twinbar[0])
    shift = min(along(p) for p in twinbar) - min(along(p) for p in dotbar)

    def runs(vals):
        out = []
        for v in sorted(vals):
            if out and v == out[-1][1] + 1:
                out[-1][1] = v
            else:
                out.append([v, v])
        return [(a, b) for a, b in out]

    # ------------- route choices, every one of them measured from I -------------
    nd = len(dots)
    barlen = len(dotbar)
    order_mode = (barlen + nd) % 3      # 0 up the bar, 1 down the bar, 2 shortest beam first
    phase_mode = (hi + wi) % 3          # 0 dot-by-dot, 1 beams then echoes, 2 echoes then beams
    cap_first = (cb != 0)               # dot bar on the far border -> cap first, then trail back
    if order_mode == 0:
        dots.sort(key=along)
    elif order_mode == 1:
        dots.sort(key=along, reverse=True)
    else:
        dots.sort(key=lambda d: (abs(cross(d) - cb), along(d)))

    # the twin bar's echo line for each dot: the whole line minus its bar cell
    echoes = []
    for d in dots:
        a2 = along(d) + shift
        if 0 <= a2 < NA:
            echoes.append((a2, runs([x for x in range(NX) if I[to_rc(a2, x)] != barc])))
        else:
            echoes.append(None)
    live = [e for e in echoes if e is not None]
    # every echo line is the same segment -> draw the first, keep it, stamp the rest
    use_copy = (dotc != 0 and len(live) >= 3
                and all(len(rs) == 1 for _, rs in live)
                and len({rs[0] for _, rs in live}) == 1)
    copied = [False]

    def emit_beam(d):
        a, x = along(d), cross(d)
        lo, up = (cb + 1, x - 1) if x > cb else (x + 1, cb - 1)
        beam = run_cells(a, lo, up) if lo <= up else []
        if cap_first:
            ops.append(4); sels.append(sel_of([to_rc(a, x)]))
            if beam:
                ops.append(int(dotc)); sels.append(sel_of(beam))
        else:
            if beam:
                ops.append(int(dotc)); sels.append(sel_of(beam))
            ops.append(4); sels.append(sel_of([to_rc(a, x)]))

    def emit_echo(i):
        if echoes[i] is None:
            return
        a2, rs = echoes[i]
        if use_copy and copied[0]:
            ops.append(30); sels.append(sel_of([to_rc(a2, rs[0][0])]))    # stamp the kept line
            return
        for (x1, x2) in rs:
            ops.append(int(dotc)); sels.append(sel_of(run_cells(a2, x1, x2)))
        if use_copy:
            ops.append(29); sels.append(sel_of(run_cells(a2, *rs[0])))    # keep this line
            copied[0] = True

    if phase_mode == 0:
        for i, d in enumerate(dots):
            emit_beam(d)
            emit_echo(i)
    elif phase_mode == 1:
        for d in dots:
            emit_beam(d)
        for i in range(len(echoes)):
            emit_echo(i)
    else:
        for i in range(len(echoes)):
            emit_echo(i)
        for d in dots:
            emit_beam(d)

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 673ef223"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 673ef223"
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
                                f"for task 673ef223"
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
                    f"Failed to build a complete episode for task 673ef223 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"673ef223-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
