"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 1, 2 and 4 are reserved for the output marks; the background is drawn from the rest.
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    # The rectangle colours do not matter to the rule (each one becomes 2, 4 and 1),
    # so only the background is fixed for the episode.
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    cols = [c for c in range(10) if c not in (1, 2, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    num = unifint(diff_lb, diff_ub, (1, 9))
    free = np.ones((h, w), dtype=bool)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or not free.any():
            break
        oh = random.randint(3, 7)
        ow = random.randint(3, 7)
        subs = [(i, j) for i in range(h) for j in range(w)
                if free[i, j] and i < h - oh and j < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        col = random.choice(remcols)
        if free[loci:loci + oh, locj:locj + ow].all():
            remcols.remove(col)
            gi[loci:loci + oh, locj:locj + ow] = col
            go[loci:loci + oh, locj:locj + ow] = 2
            go[loci, locj:locj + ow] = 4
            go[loci + oh - 1, locj:locj + ow] = 4
            go[loci:loci + oh, locj] = 4
            go[loci:loci + oh, locj + ow - 1] = 4
            for (a, b) in [(loci, locj), (loci, locj + ow - 1),
                           (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)]:
                go[a, b] = 1
            free[loci:loci + oh, locj:locj + ow] = False
            succ += 1
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(G):
    hi, wi = G.shape
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = G[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and G[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            comps.append((int(col), cells))
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the episode shares one background, so use the colour present in I
    # and in every demonstration input (the most frequent one if several qualify).
    # If that gives nothing, fall back to I's majority colour.
    present = set(I.flatten().tolist())
    cand = set(present)
    if examples:
        for pair in examples:
            try:
                cand &= set(np.asarray(pair[0], dtype=int).flatten().tolist())
            except Exception:
                pass
    cnt = Counter(I.flatten().tolist())
    if cand:
        bgc = max(cand, key=lambda c: cnt[c])
    else:
        bgc = cnt.most_common(1)[0][0]

    # The planted rectangles are all solid, single-colour, non-background components.
    rects = []
    for col, cells in _components(I):
        if col == bgc:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        h = max(rs) - r0 + 1
        w = max(cs) - c0 + 1
        if len(cells) != h * w:
            continue
        rects.append((r0, c0, h, w))
    rects.sort()

    # Work one colour layer at a time across ALL rectangles together,
    # not one rectangle at a time.
    bodies, rings, corners = [], [], []
    for (r0, c0, h, w) in rects:
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                bodies.append((r, c))
                if r == r0 or r == r0 + h - 1 or c == c0 or c == c0 + w - 1:
                    rings.append((r, c))
        corners += [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)]

    ops, sels = [], []
    if bodies:
        # Layer 1: every rectangle's cells become 2.
        ops.append(2); sels.append(sel_of(sorted(set(bodies))))
        # Layer 2: every rectangle's outline becomes 4.
        ops.append(4); sels.append(sel_of(sorted(set(rings))))
        # Layer 3: every rectangle's four corners become 1.
        ops.append(1); sels.append(sel_of(sorted(set(corners))))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
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
                                f"for task b6afb2da"
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
                    f"Failed to build a complete episode for task b6afb2da "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
