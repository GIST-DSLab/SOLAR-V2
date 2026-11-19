"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    remcols = [c for c in cols if c != markercol]
    numbgc = random.randint(1, 8)
    bgcols = random.sample(remcols, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h, max_w, markercol=None, bgcols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    cols = list(range(1, 10))
    if markercol is None:
        markercol = random.choice(cols)
    if bgcols is None:
        rem = [c for c in cols if c != markercol]
        bgcols = random.sample(rem, unifint(diff_lb, diff_ub, (1, 8)))
    bgcols = list(bgcols)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = random.randint(2, h // 4)
    ow = random.randint(2, w // 4)

    gi = [[random.choice(bgcols) for _ in range(w)] for _ in range(h)]
    allinds = [(i, j) for i in range(h) for j in range(w)]
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    for (i, j) in random.sample(allinds, numbl):
        gi[i][j] = 0

    # break every pre-existing all-zero oh x ow block
    occs = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
            if all(gi[a][b] == 0 for a in range(i, i + oh) for b in range(j, j + ow))]
    tocover = set()
    for (i, j) in occs:
        tocover.add((i + random.randrange(oh), j + random.randrange(ow)))
    for (i, j) in tocover:
        gi[i][j] = random.choice(bgcols)

    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // ((oh + 2) * (ow + 2)))))
    inds = set(allinds)
    go = [row[:] for row in gi]
    tr, succ, maxtr = 0, 0, 5 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        li, lj = random.choice(cands)
        bd = [(li + a, lj + b) for a in range(oh) for b in range(ow)]
        plcd = set()
        for b in range(lj - 1, lj + ow + 1):
            plcd.add((li - 1, b)); plcd.add((li + oh, b))
        for a in range(li - 1, li + oh + 1):
            plcd.add((a, lj - 1)); plcd.add((a, lj + ow))
        if not plcd.issubset(inds):
            continue
        succ += 1
        inds -= plcd
        for (a, b) in bd:
            gi[a][b] = 0
            go[a][b] = 0
        for (a, b) in plcd:
            if succ == 1:
                gi[a][b] = markercol
            go[a][b] = markercol
        if succ > 1:
            ln1 = [(li - 1, b) for b in range(lj, lj + ow)]
            ln2 = [(li + oh, b) for b in range(lj, lj + ow)]
            ln3 = [(a, lj - 1) for a in range(li, li + oh)]
            ln4 = [(a, lj + ow) for a in range(li, li + oh)]
            for ln in (ln1, ln2, ln3, ln4):
                a, b = random.choice(ln)
                gi[a][b] = random.choice(bgcols)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    g = [[int(v) for v in row] for row in I.tolist()]
    ops, sels = [], []

    def box_cells(r0, c0, r1, c1):
        cells = set()
        for c in range(c0, c1 + 1):
            cells.add((r0, c)); cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0)); cells.add((r, c1))
        return cells

    def frame_candidates():
        out = []
        for col in range(1, 10):
            cells = {(r, c) for r in range(hi) for c in range(wi) if g[r][c] == col}
            if not cells:
                continue
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if cells != box_cells(r0, c0, r1, c1):
                continue
            if any(g[r][c] != 0 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
                continue
            out.append((len(cells), col, r0, c0, r1, c1))
        return out

    def blocks_for(oh, ow):
        # every all-zero oh x ow hole with room for a frame around it
        return [(r, c) for r in range(1, hi - oh) for c in range(1, wi - ow)
                if all(g[i][j] == 0 for i in range(r, r + oh) for j in range(c, c + ow))]

    def select_blocks(oh, ow, marker_block):
        used, chosen = set(), []
        cands = blocks_for(oh, ow)
        for (r, c) in [marker_block] + [b for b in cands if b != marker_block]:
            ring = box_cells(r - 1, c - 1, r + oh, c + ow)
            if ring & used:
                continue
            used |= ring
            chosen.append((r, c))
        return chosen

    cands = frame_candidates()
    if cands:
        scored = []
        for (_, col, r0, c0, r1, c1) in cands:
            n = len(select_blocks(r1 - r0 - 1, c1 - c0 - 1, (r0 + 1, c0 + 1)))
            scored.append((0 if n >= 2 else 1, col, r0, c0, r1, c1))
        scored.sort()
        _, markercol, mr0, mc0, mr1, mc1 = scored[0]
        oh, ow = mr1 - mr0 - 1, mc1 - mc0 - 1
        marker_block = (mr0 + 1, mc0 + 1)
        targets = [b for b in select_blocks(oh, ow, marker_block) if b != marker_block]
        if targets:
            # The marker frame is the stamp. CopyI over its full bounding rectangle:
            # its hollow 0 interior is not copied, so the clipboard holds just the ring.
            ops.append(28)
            sels.append([mr0, mc0, mr1 - mr0, mc1 - mc0])
            for (r, c) in targets:
                # stamp the ring around each hidden zero hole (top-left = ring corner)
                ops.append(30)
                sels.append([r - 1, c - 1, 0, 0])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
