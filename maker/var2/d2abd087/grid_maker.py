"""
ARC Task: d2abd087 (RE-ARC) — LLM-generated grid_maker
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


# Discrete structural cases of this task: an instance may contain both label
# classes, only six-cell shapes, or no six-cell shape at all.  The examples must
# show both classes, otherwise the 6 -> 2 / else -> 1 rule is not learnable.
VARIANTS = [
    {"class_mode": "mix"},
    {"class_mode": "only6"},
    {"class_mode": "nosix"},
]


def sample_colors(num_examples=None) -> dict:
    # Only the background colour is an episode-level role (1 and 2 are the
    # hardcoded output labels; object colours are irrelevant to the rule).
    bgc = random.choice([c for c in range(10) if c not in (1, 2)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
        if not any(e["class_mode"] == "mix" for e in examples):
            examples[0] = {"class_mode": "mix"}
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, class_mode=None) -> dict:
    def uf(lb, ub, bounds):
        try:
            return unifint(lb, ub, bounds)  # noqa: F821  (RE-ARC utils, when present)
        except NameError:
            a, b = bounds
            if a > b:
                a, b = b, a
            lo = a + (b - a) * lb
            hi = a + (b - a) * ub
            lo_i = int(lo)
            if lo_i < lo:
                lo_i += 1
            lo_i = max(a, lo_i)
            hi_i = min(b, int(hi))
            if hi_i < lo_i:
                hi_i = lo_i
            return random.randint(lo_i, hi_i)

    if class_mode is None:
        class_mode = random.choice(["mix", "mix", "only6", "nosix"])

    hub = max(8, min(30, int(max_h)))
    wub = max(8, min(30, int(max_w)))
    remcols = [c for c in range(10) if c != bgc]

    best = None
    gi = [[bgc]]
    go = [[bgc]]
    for _attempt in range(16):
        h = uf(diff_lb, diff_ub, (8, hub))
        w = uf(diff_lb, diff_ub, (8, wub))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        inds = set((r, c) for r in range(h) for c in range(w))
        nobjs = uf(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
        if class_mode == "mix":
            nobjs = max(2, nobjs)
        need6 = class_mode in ("mix", "only6")
        needoth = class_mode in ("mix", "nosix")
        maxtrials = 4 * nobjs + 8
        tr = 0
        succ = 0
        while succ < nobjs and tr <= maxtrials and inds:
            tr += 1
            lb = uf(diff_lb, diff_ub, (1, 5))
            lopts = list(range(lb, 6))
            ubi = uf(diff_lb, diff_ub, (1, 5))
            uopts = list(range(7, 12 - ubi + 1))
            if class_mode == "only6":
                numcells = 6
            elif class_mode == "nosix":
                numcells = random.choice(lopts + uopts)
            elif need6:
                numcells = 6
            elif needoth:
                numcells = random.choice(lopts + uopts)
            else:
                numcells = 6 if random.random() < 0.5 else random.choice(lopts + uopts)

            sp = (random.randrange(5), random.randrange(5))
            obj = {sp}
            ok = True
            for _k in range(numcells - 1):
                cands = set()
                for (r, c) in obj:
                    for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nb[0] < 5 and 0 <= nb[1] < 5 and nb not in obj:
                            cands.add(nb)
                if not cands:
                    ok = False
                    break
                obj.add(random.choice(sorted(cands)))
            if not ok or len(obj) != numcells:
                continue

            mr = min(r for r, _ in obj)
            mc = min(c for _, c in obj)
            obj = set((r - mr, c - mc) for r, c in obj)
            loc = random.choice(sorted(inds))
            plcd = set((r + loc[0], c + loc[1]) for r, c in obj)
            if plcd.issubset(inds):
                col = random.choice(remcols)
                lab = 2 if len(obj) == 6 else 1
                for (r, c) in plcd:
                    gi[r][c] = col
                    go[r][c] = lab
                succ += 1
                if lab == 2:
                    need6 = False
                else:
                    needoth = False
                ring = set()
                for (r, c) in plcd:
                    ring |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
                inds = (inds - plcd) - ring

        if succ >= 1 and not (class_mode == "mix" and (need6 or needoth)):
            return {"input": gi, "output": go}
        if succ >= 1 and best is None:
            best = {"input": gi, "output": go}

    if best is not None:
        return best
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Rule: every 4-connected univalued shape of exactly 6 cells becomes 2,
    every other shape becomes 1.  The route (which shape first, which sweep
    direction, bucket-fill vs. paint-the-shape, one-by-one vs. one class at a
    time, base-coat-then-promote) is chosen deterministically from features of
    the INPUT, so the same instance always yields the same route and different
    instances yield different routes.  Nothing here reads O."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---- background colour -------------------------------------------------
    # From the demonstrations: every example output holds only 1, 2 and the
    # background.  (Fallback: commonest colour of I that is not a label.)
    bgc = None
    if examples:
        sets = []
        for pair in examples:
            try:
                if isinstance(pair, dict):
                    eo = pair.get("output")
                else:
                    eo = pair[1]
                if eo is None:
                    continue
                sets.append(set(np.asarray(eo, dtype=int).flatten().tolist()) - {1, 2})
            except Exception:
                continue
        if sets:
            inter = set.intersection(*sets) & set(I.flatten().tolist())
            if len(inter) == 1:
                bgc = int(inter.pop())
    if bgc is None:
        counts = Counter(I.flatten().tolist())
        for col, _ in counts.most_common():
            if col not in (1, 2):
                bgc = int(col)
                break
        if bgc is None:
            bgc = int(counts.most_common(1)[0][0])

    # ---- segment I into univalued 4-connected non-background shapes --------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []                      # (colour, sorted cells) in reading order
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((col, sorted(cells)))

    ops, sels = [], []
    cur = I.copy()

    def emit_color(cells, color):
        """Paint exactly these cells.  Works for colour 0 as well (Color0 is an
        ordinary op) -- and emits nothing when there is nothing to change."""
        cells = [(int(r), int(c)) for r, c in cells]
        if not cells:
            return
        if all(cur[r, c] == color for r, c in cells):
            return                     # would be an invisible no-op
        ops.append(int(color))
        sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = color

    def emit_fill(cells, color):
        """Bucket-fill the shape from one seed.  Shapes are 4-separated in this
        task, so the fill covers exactly the shape.  If the shape currently
        holds colour 0, paint the cells explicitly instead."""
        if not cells:
            return
        sr, sc = int(cells[0][0]), int(cells[0][1])
        if cur[sr, sc] == color:
            return                     # already labelled: a fill would do nothing
        if cur[sr, sc] == 0:
            emit_color(cells, color)
            return
        ops.append(10 + int(color))
        sels.append(sel_of([(sr, sc)]))   # mask with exactly one True cell
        for r, c in cells:
            cur[int(r), int(c)] = color

    if comps:
        nobj = len(comps)
        total = sum(len(cs) for _, cs in comps)
        first_col = comps[0][0]
        sig_a = hi * 3 + wi * 5 + nobj * 7 + bgc
        sig_b = total * 3 + first_col * 5 + nobj * 2 + wi
        mode = sig_a % 6
        order_id = sig_b % 6
        bit_group = (sig_a // 6) % 2
        bit_op = (sig_b // 6) % 2

        def order_key(comp):
            _col, cells = comp
            r0, c0 = cells[0]
            cmin = min(c for _, c in cells)
            rat = min(r for r, c in cells if c == cmin)
            if order_id == 0:          # reading order
                return (r0, c0, 0)
            if order_id == 1:          # column sweep, left -> right
                return (cmin, rat, 0)
            if order_id == 2:          # reading order, bottom-right first
                return (-r0, -c0, 0)
            if order_id == 3:          # column sweep, right -> left
                return (-cmin, rat, 0)
            if order_id == 4:          # outward from the top-left corner
                return (r0 + c0, r0, c0)
            return (-len(cells), r0, c0)   # biggest shapes first

        ordered = sorted(comps, key=order_key)
        sixes = [cp for cp in ordered if len(cp[1]) == 6]
        others = [cp for cp in ordered if len(cp[1]) != 6]
        six_cells = [cell for _, cs in sixes for cell in cs]
        oth_cells = [cell for _, cs in others for cell in cs]
        all_cells = [cell for _, cs in ordered for cell in cs]

        # A base coat is only meaningful if something survives it uncovered.
        if mode == 4 and not any(col != 1 for col, _ in others):
            mode = 3
        elif mode == 5 and not any(col != 2 for col, _ in sixes):
            mode = 3

        if mode == 0:
            # bucket-fill each shape in turn
            for col, cells in ordered:
                emit_fill(cells, 2 if len(cells) == 6 else 1)
        elif mode == 1:
            # paint each shape's own cells in turn
            for col, cells in ordered:
                emit_color(cells, 2 if len(cells) == 6 else 1)
        elif mode == 2:
            # deal with one label class completely, then the other
            groups = [(sixes, 2), (others, 1)] if bit_group else [(others, 1), (sixes, 2)]
            for group, tgt in groups:
                for col, cells in group:
                    if bit_op:
                        emit_fill(cells, tgt)
                    else:
                        emit_color(cells, tgt)
        elif mode == 3:
            # select every shape of a class at once and paint the class
            batches = [(six_cells, 2), (oth_cells, 1)] if bit_group else [(oth_cells, 1), (six_cells, 2)]
            for cells, tgt in batches:
                emit_color(cells, tgt)
        elif mode == 4:
            # base coat: every shape becomes 1, then promote the six-cell ones
            emit_color(all_cells, 1)
            if bit_op:
                emit_color(six_cells, 2)
            else:
                for col, cells in sixes:
                    emit_color(cells, 2)
        else:
            # base coat the other way round: everything 2, then demote the rest
            emit_color(all_cells, 2)
            if bit_op:
                emit_color(oth_cells, 1)
            else:
                for col, cells in others:
                    emit_color(cells, 1)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid bbox: submit the whole canvas
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
                        f"num_examples+1 ({num_examples + 1}) for task d2abd087"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d2abd087"
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
                                f"for task d2abd087"
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
                    f"Failed to build a complete episode for task d2abd087 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d2abd087-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
