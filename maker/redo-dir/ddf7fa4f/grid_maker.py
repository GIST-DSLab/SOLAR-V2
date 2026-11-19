"""
ARC Task: ddf7fa4f (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------- variants
# The generator ends with rotf = choice((identity, rot90, rot180, rot270)),
# which decides WHICH border line carries the single-pixel markers
# (top / right / bottom / left).  That is a discrete structural case, so it is
# planned per episode instead of being re-rolled inside generate().
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a real role here: the rule is positional
    # (a marker recolours the rectangle it shares a column/row with), so the
    # object colours may stay random.
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(VARIANTS)["rot"]

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    hmax = max(10, min(30, int(max_h)))
    wmax = max(10, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (10, hmax))
    w = unifint(diff_lb, diff_ub, (10, wmax))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, min(w // 3, (h * w) // 36))))

    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    inds = {(i, j) for i in range(2, h) for j in range(w)}   # ij[0] > 1
    succ, tr, maxtr = 0, 0, 10 * nocc

    while succ < nocc and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        cands = [(i, j) for (i, j) in sorted(inds) if i <= h - oh and j <= w - ow]
        hastobein = {c for c in range(w) if gi[0][c] == bgc}
        cantbein = {c for c in range(w) if gi[0][c] != bgc}
        jopts = {
            j for j in range(w)
            if len(set(range(j, j + ow)) & hastobein) > 0
            and len(set(range(j, j + ow)) & cantbein) == 0
        }
        cands = [ij for ij in cands if ij[1] in jopts]
        if len(cands) == 0:
            continue
        loci, locj = random.choice(cands)
        locat_opts = [jj for jj in range(locj, locj + ow) if jj in hastobein]
        if not locat_opts:
            continue
        locat = random.choice(locat_opts)
        sq = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if sq <= inds:
            succ += 1
            nb = set()
            for (i, j) in sq:
                nb.add((i + 1, j)); nb.add((i - 1, j))
                nb.add((i, j + 1)); nb.add((i, j - 1))
            inds = (inds - sq) - nb
            col = random.choice(remcols)
            gr = random.choice([c for c in remcols if c != col])
            gi[0][locat] = col
            go[0][locat] = col
            for (i, j) in sq:
                gi[i][j] = gr
                go[i][j] = col

    def rot90cw(g):
        return [list(r) for r in zip(*g[::-1])]

    for _ in range(int(rot) % 4):
        gi = rot90cw(gi)
        go = rot90cw(go)

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


# ---------------------------------------------------------------- derivation
def derive_operations(I, O):
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- background: most common colour of the border ring (as the task builds it)
    ring = []
    for c in range(w):
        ring.append(int(I[0, c])); ring.append(int(I[h - 1, c]))
    for r in range(h):
        ring.append(int(I[r, 0])); ring.append(int(I[r, w - 1]))
    bgc = Counter(ring).most_common(1)[0][0]

    # --- the four border lines, in the order the task considers them
    lines = [
        [(0, c) for c in range(w)],            # top row
        [(r, 0) for r in range(h)],            # left column
        [(r, w - 1) for r in range(h)],        # right column
        [(h - 1, c) for c in range(w)],        # bottom row
    ]

    def lone_cells(line):
        """number of isolated single non-bg cells along the 1-wide strip"""
        vals = [int(I[r, c]) for (r, c) in line]
        n, i = 0, 0
        while i < len(vals):
            if vals[i] == bgc:
                i += 1
                continue
            j = i
            while j < len(vals) and vals[j] == vals[i]:
                j += 1
            if j - i == 1:
                n += 1
            i = j
        return n

    scores = [lone_cells(l) for l in lines]
    marker_line = lines[int(np.argmax(scores))]
    line_set = set(marker_line)

    # --- markers: every non-background cell sitting on that line
    markers = [((r, c), int(I[r, c])) for (r, c) in marker_line if int(I[r, c]) != bgc]

    # --- rectangles: 4-connected mono-colour components not touching the line
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(I[r, c]) == bgc:
                continue
            col = int(I[r, c])
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(I[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not (set(comp) & line_set):
                objects.append((col, sorted(comp)))

    def matches(mpos, cells):
        mr, mc = mpos
        return any(cr == mr or cc == mc for (cr, cc) in cells)

    # --- each marker claims the rectangle it lines up with; ambiguous markers
    #     wait until the competing rectangle has been claimed by its own marker
    rem_m = list(markers)
    rem_o = list(range(len(objects)))
    assignments = []          # (marker_pos, colour, object_index) in resolve order
    while rem_m and rem_o:
        round_pairs = []
        claimed = set()
        for (mpos, mcol) in rem_m:
            hits = [oi for oi in rem_o if matches(mpos, objects[oi][1])]
            if len(hits) == 1 and hits[0] not in claimed:
                claimed.add(hits[0])
                round_pairs.append((mpos, mcol, hits[0]))
        if not round_pairs:
            break
        round_pairs.sort(key=lambda t: t[0])
        assignments.extend(round_pairs)
        used_m = {(p, c) for (p, c, _) in round_pairs}
        rem_m = [m for m in rem_m if m not in used_m]
        rem_o = [oi for oi in rem_o if oi not in claimed]

    done = set()
    for (mpos, mcol, oi) in assignments:
        col, cells = objects[oi]
        done.add(oi)
        if col != mcol:
            ops.append(int(mcol))
            sels.append(sel_of(cells))

    # safety net: a rectangle no marker could claim keeps whatever colour it holds
    for oi, (col, cells) in enumerate(objects):
        if oi in done:
            continue
        tgt = int(O[cells[0][0], cells[0][1]])
        if tgt != col:
            ops.append(tgt)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # whole grid: full rectangle, bbox is exact
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
                        f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
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
                                f"for task ddf7fa4f"
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
                    f"Failed to build a complete episode for task ddf7fa4f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ddf7fa4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
