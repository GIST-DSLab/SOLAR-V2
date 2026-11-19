"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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

# The one structural branch of this task: the stripes either run across the grid
# (the answer is a column) or down it (the answer is a row).
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    # The rule is purely structural — "collapse repeated stripes" — so no colour
    # role has to be pinned.  One palette per episode keeps the instances looking
    # like a single puzzle.  0 stays in the palette on purpose: here it is an
    # ordinary stripe colour, not a background.
    palette = random.sample(range(10), random.randint(4, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, transposed=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if transposed is None:
        transposed = random.choice([True, False])
    if not palette or len(palette) < 2:
        palette = list(range(10))

    # before the optional transpose the grid is (stripes + duplicates) x thickness
    len_cap = max(3, min(30, max_w if transposed else max_h))
    thick_cap = max(1, min(30, max_h if transposed else max_w))

    h = unifint(diff_lb, diff_ub, (2, max(2, min(15, len_cap - 1))))
    h = max(2, min(h, len_cap - 1))
    w = unifint(diff_lb, diff_ub, (1, thick_cap))
    w = max(1, min(w, thick_cap))

    def draw(pool):
        out, lastc = [], -1
        for _ in range(h):
            c = random.choice([x for x in pool if x != lastc])
            out.append(c)
            lastc = c
        return out

    nonblack = [c for c in palette if c != 0]
    for _ in range(20):
        cols = draw(palette)
        # A picture striped in black and one other colour alone is degenerate: the
        # answer is then fixed by which stripes are black, so any route reaches it.
        if 0 not in cols or len(set(cols)) > 2:
            break
    else:
        cols = draw(nonblack if len(nonblack) >= 2 else palette)

    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]

    numinserts = unifint(diff_lb, diff_ub, (1, max(1, len_cap - h)))
    numinserts = max(1, min(numinserts, len_cap - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + [list(gi[loc])] + gi[loc + 1:]

    if transposed:                      # dmirror of a stack of solid lines == transpose
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Collapse repeated neighbours along BOTH axes with one and the same rule:
    first squeeze consecutive identical rows of the whole picture down to one,
    keep only those distinct rows, then squeeze consecutive identical columns of
    what is left down to one and keep only those.  No orientation test is needed:
    along the axis the stripes do not vary, everything is a single run, and the
    pass reduces to keeping its first line."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def runs_of(lines):
        out = []
        for ln in lines:
            if out and out[-1][0] == ln:
                out[-1][1] += 1
            else:
                out.append([ln, 1])
        return out

    row_runs = runs_of([tuple(I[r].tolist()) for r in range(hi)])
    col_runs = runs_of([tuple(I[:, c].tolist()) for c in range(wi)])
    n_r, n_c = len(row_runs), len(col_runs)

    # A Move only carries non-zero cells, so a black line would not travel.
    # Give black a stand-in colour for the duration (all cells of colour 0).
    present = set(int(v) for v in np.unique(I))
    spare = None
    if 0 in present:
        spare = next((c for c in range(1, 10) if c not in present), None)
        if spare is not None:
            ops.append(spare)
            sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]))

    def squeeze(runs, extent, span, axis_rows):
        """Slide everything past each repeat back over it: one grab of the slab
        (the full rectangle of lines past the repeat, `span` lines thick across),
        then empty-selection Moves so the same slab keeps gliding.
        Returns the colour each line along this axis holds afterwards (a line of
        colour 0 is not carried by a Move, so it is tracked to be put back)."""
        seq = []
        for ln, L in runs:
            seq += [ln[0] if ln[0] != 0 or spare is None else spare] * L
        live, cur = extent, 0
        for _line, L in runs:
            ex = L - 1
            if ex > 0:
                s = cur + L
                if s <= live - 1 and any(seq[j] != 0 for j in range(s, live)):
                    if axis_rows:
                        cells = [(r, c) for r in range(s, live) for c in range(span)]
                        op = 20                           # MoveU
                    else:
                        cells = [(r, c) for r in range(span) for c in range(s, live)]
                        op = 23                           # MoveL
                    ops.append(op); sels.append(sel_of(cells))
                    for _ in range(ex - 1):
                        ops.append(op); sels.append(sel_of([]))
                    new = seq[:s] + [0] * (live - s) + seq[live:]
                    for j in range(s, live):
                        if seq[j] != 0:
                            new[j - ex] = seq[j]
                    seq = new
                    live -= ex
                elif s > live - 1:
                    live = cur + 1                        # trailing repeat: crop drops it
            cur += 1
        return seq

    # pass 1: rows of the whole picture
    rseq = squeeze(row_runs, hi, wi, True)
    if n_r < hi:
        ops.append(33); sels.append([0, 0, n_r - 1, wi - 1])   # keep the distinct rows (full rectangle)
    # pass 2: columns of the surviving rows
    cseq = squeeze(col_runs, wi, n_r, False)
    if n_c < wi:
        ops.append(33); sels.append([0, 0, n_r - 1, n_c - 1])  # keep the distinct columns (full rectangle)

    # put black back on the kept lines whose stripe (read from I) is black
    if 0 in present:
        rv = [ln[0] for ln, _ in row_runs]
        cv = [ln[0] for ln, _ in col_runs]
        vary_rows = n_r > 1
        zc = [(r, c) for r in range(n_r) for c in range(n_c)
              if (rv[r] if vary_rows else cv[c]) == 0
              and (rseq[r] if vary_rows else cseq[c]) != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))

    ops.append(34); sels.append([0, 0, n_r - 1, n_c - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
