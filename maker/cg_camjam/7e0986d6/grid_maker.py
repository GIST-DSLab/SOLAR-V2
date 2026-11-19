"""
ARC Task: 7e0986d6 (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    nsq = random.randint(1, 5)
    sqcols = random.sample(remcols, nsq)
    remcols = [c for c in remcols if c not in sqcols]
    nn = random.randint(1, len(remcols))
    noisecols = random.sample(remcols, nn)
    return {"bgc": bgc, "sqcols": sqcols, "noisecols": noisecols}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqcols=None, noisecols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    if bgc is None:
        c = sample_colors(); bgc, sqcols, noisecols = c["bgc"], c["sqcols"], c["noisecols"]
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    maxtr = 5 * numsq
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = random.randint(2, 7); ow = random.randint(2, 7)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        sq = {(a, b) for a in range(li, li + oh) for b in range(lj, lj + ow)}
        if sq <= inds:
            succ += 1
            outb = {(a, b) for a in range(li - 1, li + oh + 1) for b in range(lj - 1, lj + ow + 1)}
            inds = inds - sq - outb
            col = random.choice(sqcols)
            for a, b in sq:
                go[a][b] = col
    gi = [row[:] for row in go]
    namt = unifint(diff_lb, diff_ub, (1, (h * w) // 9))
    cands = {(i, j) for i in range(h) for j in range(w)}
    for _ in range(namt):
        if not cands:
            break
        li, lj = random.choice(sorted(cands))
        col = gi[li][lj]
        torem = {(a, b) for a in range(li - 1, li + 2) for b in range(lj - 1, lj + 2)
                 if (a, b) != (li, lj) and 0 <= a < h and 0 <= b < w and gi[a][b] == col}
        cands = cands - torem
        gi[li][lj] = random.choice(noisecols)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}

def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # 4-connected single-colour components of the input
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            st = [(r, c)]; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True; st.append((na, nb))
            comps.append((col, cells))
    counts = Counter(I.flatten().tolist())
    bgc = max(comps, key=lambda t: (len(t[1]), counts[t[0]]))[0]
    # a block is a solid rectangle (at least 2x2) with at most a few pixels punched out;
    # everything else that is not background is a noise speckle (a dot or a thin chain
    # of dots, which can line up through a one-cell gap between two blocks)
    def is_block(cells):
        rs = [a for a, _ in cells]; cs = [b for _, b in cells]
        bh = max(rs) - min(rs) + 1; bw = max(cs) - min(cs) + 1
        return bh >= 2 and bw >= 2 and len(cells) >= 3
    blocks = [(col, cells) for col, cells in comps if col != bgc and is_block(cells)]
    noise = [cell for col, cells in comps if col != bgc and not is_block(cells) for cell in cells]

    ops, sels = [], []
    claimed = set()
    # each noise pixel punched into a block takes back that block's colour
    for col, cells in sorted(blocks, key=lambda t: (min(a for a, _ in t[1]), min(b for _, b in t[1]))):
        r0 = min(a for a, _ in cells); r1 = max(a for a, _ in cells)
        c0 = min(b for _, b in cells); c1 = max(b for _, b in cells)
        holes = sorted(p for p in noise if r0 <= p[0] <= r1 and c0 <= p[1] <= c1 and p not in claimed)
        if holes:
            ops.append(int(col)); sels.append(sel_of(holes))
            claimed.update(holes)
    # every other noise pixel sits on open background and takes the background colour
    rest = sorted(p for p in noise if p not in claimed)
    if rest:
        ops.append(int(bgc)); sels.append(sel_of(rest))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
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
                                f"for task 7e0986d6"
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
                    f"Failed to build a complete episode for task 7e0986d6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7e0986d6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
