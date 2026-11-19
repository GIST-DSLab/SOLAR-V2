"""
ARC Task: b27ca6d3 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colours
def sample_colors(num_examples=None) -> dict:
    # 3 is the marker colour the rule paints with, so it is never a picture colour
    cols = [c for c in range(10) if c != 3]
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc) -> dict:
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for _attempt in range(20):
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        ndots = unifint(diff_lb, diff_ub, (0, (h * w) // 5))
        nbars = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 12)))
        dot = frozenset({(dotc, (1, 1))}) | recolor(bgc, dneighbors((1, 1)))
        bar1 = fill(canvas(bgc, (4, 3)), dotc, {(1, 1), (2, 1)})
        bar2 = dmirror(bar1)
        bar1 = asobject(bar1)
        bar2 = asobject(bar2)
        opts = [dot] * ndots + [choice((bar1, bar2)) for k in range(nbars)]
        shuffle(opts)
        inds = shift(asindices(canvas(-1, (h + 2, w + 2))), (-1, -1))
        placed_bars = 0
        for elem in opts:
            loc = (-1, -1)
            tr = 0
            while not toindices(shift(elem, loc)).issubset(inds) and tr < 5:
                loc = choice(totuple(inds))
                tr += 1
            xx = shift(elem, loc)
            if toindices(xx).issubset(inds):
                gi = paint(gi, xx)
                if len(elem) == 12:
                    go = paint(go, {cel if cel[0] != bgc else (3, cel[1]) for cel in xx})
                    placed_bars += 1
                else:
                    go = paint(go, xx)
                inds = inds - toindices(xx)
        if placed_bars > 0:
            break
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- solver
def derive_operations(I, O, examples=None):
    """Rule: every two-cell blob gets a box drawn around it in the marker colour.

    The route (which box first, which way round, whether the box is outlined,
    traced side by side, laid down solid and the blob put back on top, and
    whether repeated boxes are stamped from the first one) is decided from the
    input picture alone, so the same picture always yields the same route.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = I.copy()
    ops, sels = [], []

    # the canvas colour the generator paints before dropping anything on it
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # the marker colour, read off the worked examples (not off O)
    fillc = 3
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            eo = np.asarray(ex[1], dtype=int)
            if ei.shape != eo.shape:
                continue
            d = eo[ei != eo]
            if d.size:
                fillc = int(Counter(d.tolist()).most_common(1)[0][0])
                break

    # --- read the picture: orthogonally connected blobs of a single colour ---
    seen = np.zeros((hi, wi), dtype=bool)
    doms = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) != 2:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            rect = [(rr, cc)
                    for rr in range(r0 - 1, r1 + 2)
                    for cc in range(c0 - 1, c1 + 2)
                    if 0 <= rr < hi and 0 <= cc < wi]
            ring = [p for p in rect
                    if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1)]
            doms.append({
                'cells': sorted(cells), 'col': col,
                'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1,
                'h': r1 - r0 + 1, 'w': c1 - c0 + 1,
                'rect': rect, 'ring': ring,
                'full': (r0 - 1 >= 0 and c0 - 1 >= 0 and r1 + 1 < hi and c1 + 1 < wi),
            })

    if not doms:
        return [34], [[0, 0, hi - 1, wi - 1]]

    ring_union = set()
    for d in doms:
        ring_union |= set(d['ring'])

    # --- route choices, all read off the input picture ---
    first = min(doms, key=lambda d: (d['r0'], d['c0']))
    nd = len(doms)
    nz = int(np.count_nonzero(I != bgc))
    dotc = first['col']
    order_idx = (first['r0'] * 3 + first['c0'] * 5 + nd) % 5
    style_idx = (nz + dotc) % 3
    stamp_flag = ((bgc + nd) % 2 == 1)

    cr, cc0 = (hi - 1) / 2.0, (wi - 1) / 2.0

    def order_key(d):
        if order_idx == 0:                       # reading order
            return (d['r0'], d['c0'], 0.0)
        if order_idx == 1:                       # backwards, bottom-right first
            return (-d['r0'], -d['c0'], 0.0)
        if order_idx == 2:                       # column by column, left to right
            return (d['c0'], d['r0'], 0.0)
        if order_idx == 3:                       # column by column, right to left
            return (-d['c0'], -d['r0'], 0.0)
        return (abs((d['r0'] + d['r1']) / 2.0 - cr)     # from the middle outwards
                + abs((d['c0'] + d['c1']) / 2.0 - cc0),
                d['r0'], d['c0'])

    plan = sorted(range(nd), key=lambda i: order_key(doms[i]))

    # a family of identical, fully on-grid boxes can be stamped from the first
    stamp_set, template = set(), None
    if stamp_flag:
        fams = {}
        for i in plan:
            d = doms[i]
            if d['full']:
                fams.setdefault((d['h'], d['w'], d['col']), []).append(i)
        best = None
        for _k, lst in fams.items():
            if len(lst) >= 2 and (best is None or len(lst) > len(best)):
                best = lst
        if best:
            template = best[0]
            stamp_set = set(best[1:])

    def stroke(cells, colour):
        """Paint cells that do not already hold `colour`; emit nothing if none do."""
        cells = [p for p in cells if int(G[p]) != colour]
        if not cells:
            return
        ops.append(colour)
        sels.append(sel_of(cells))
        for p in cells:
            G[p] = colour

    def draw_box(d):
        if style_idx == 0:
            # outline the blob in one go
            stroke(d['ring'], fillc)
        elif style_idx == 1:
            # trace the box side by side, clockwise from the top edge
            r0, r1, c0, c1 = d['r0'], d['r1'], d['c0'], d['c1']
            top = [(r0 - 1, c) for c in range(c0 - 1, c1 + 2)]
            right = [(r, c1 + 1) for r in range(r0, r1 + 1)]
            bottom = [(r1 + 1, c) for c in range(c0 - 1, c1 + 2)]
            left = [(r, c0 - 1) for r in range(r0, r1 + 1)]
            for side in (top, right, bottom, left):
                stroke([p for p in side if 0 <= p[0] < hi and 0 <= p[1] < wi], fillc)
        else:
            # lay the whole block down solid, then put the blob back on top
            stroke(d['rect'], fillc)
            back = {}
            for p in d['rect']:
                if p in ring_union:
                    continue
                back.setdefault(int(I[p]), []).append(p)
            for colour in sorted(back):
                stroke(back[colour], colour)   # colour may be 0 -> explicit Color0

    for i in plan:
        d = doms[i]
        if i in stamp_set:
            # stamp the finished box copied from the template
            ops.append(30)
            sels.append([d['r0'] - 1, d['c0'] - 1, 0, 0])
            t = doms[template]
            for rr in range(t['h'] + 2):
                for cc in range(t['w'] + 2):
                    v = int(G[t['r0'] - 1 + rr, t['c0'] - 1 + cc])
                    if v != 0:                       # 0 does not travel with a Paste
                        G[d['r0'] - 1 + rr, d['c0'] - 1 + cc] = v
            # whatever the clipboard could not carry (a blob whose colour is 0)
            missing = [p for p in d['cells'] if int(G[p]) != d['col']]
            if missing:
                ops.append(d['col'])
                sels.append(sel_of(missing))
                for p in missing:
                    G[p] = d['col']
        else:
            draw_box(d)
            if template is not None and i == template:
                # grab the finished block: this selection IS exactly that full rectangle
                ops.append(29)
                sels.append([d['r0'] - 1, d['c0'] - 1, d['h'] + 1, d['w'] + 1])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b27ca6d3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b27ca6d3"
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
                                f"for task b27ca6d3"
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
                    f"Failed to build a complete episode for task b27ca6d3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b27ca6d3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
