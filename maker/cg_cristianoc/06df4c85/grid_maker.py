"""
ARC Task: 06df4c85 (RE-ARC) — LLM-generated grid_maker
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
    # Rule (connect aligned same-coloured blocks) does not depend on which
    # foreground colours are used; fix only the block background and lattice colour.
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    def unifint(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, round(lo + (hi - lo) * d)))

    oh = unifint(1, 4)
    ow = unifint(1, 4)
    numh = unifint(3, (max_h + 1) // (oh + 1))
    numw = unifint(3, (max_w + 1) // (ow + 1))
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    if fullh > max_h or fullw > max_w:
        raise ValueError("grid too large")
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    ncols = unifint(1, min(8, (numh * numh) // 3))
    ccols = random.sample(remcols, ncols)

    gi = [[linc] * fullw for _ in range(fullh)]
    for a in range(numh):
        for b in range(numw):
            for i in range(oh):
                for j in range(ow):
                    gi[a * (oh + 1) + i][b * (ow + 1) + j] = bgc
    go = [row[:] for row in gi]

    def fillblk(g, col, loc):
        for i in range(oh):
            for j in range(ow):
                g[loc[0] + i][loc[1] + j] = col

    for col in ccols:
        inds = set()
        for a in range(numh):
            for b in range(numw):
                r, c = a * (oh + 1), b * (ow + 1)
                if all(go[r + i][c + j] == bgc for i in range(oh) for j in range(ow)):
                    inds.add((r, c))
        if not inds:
            break
        loc = random.choice(sorted(inds))
        narms = random.randint(1, 4)
        armdirs = random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], narms)
        for x, y in armdirs:
            arm = []
            for k in range(1, max(numh, numw)):
                nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                if nl not in inds:
                    break
                arm.append(nl)
            if len(arm) < 2:
                continue
            aidx = unifint(1, len(arm) - 1)
            fillblk(gi, col, arm[aidx])
            for p in arm[:aidx + 1]:
                fillblk(go, col, p)
        fillblk(gi, col, loc)
        fillblk(go, col, loc)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- the lattice: a fully uniform row can only be a separator line
    linc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            linc = int(I[r, 0])
            break
    line_rows = [r for r in range(hi) if bool(np.all(I[r] == linc))]
    line_cols = [c for c in range(wi) if bool(np.all(I[:, c] == linc))]
    oh, ow = line_rows[0], line_cols[0]
    rs = list(range(0, hi, oh + 1))
    cs = list(range(0, wi, ow + 1))
    nh, nw = len(rs), len(cs)

    B = [[int(I[rs[a], cs[b]]) for b in range(nw)] for a in range(nh)]
    bgc = Counter(x for row in B for x in row).most_common(1)[0][0]

    order = []
    for a in range(nh):
        for b in range(nw):
            if B[a][b] != bgc and B[a][b] not in order:
                order.append(B[a][b])

    cur = [row[:] for row in B]   # block colours as the grid stands now
    G = I.copy()                  # working grid tracker
    ops, sels = [], []

    # --- each aligned pair of same-coloured blocks is joined by ONE straight
    #     line: the full rectangle spanning both blocks (separators included)
    for col in order:
        blocks = sorted((a, b) for a in range(nh) for b in range(nw) if B[a][b] == col)
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                (a1, b1), (a2, b2) = blocks[i], blocks[j]
                if a1 == a2:
                    mids = [(a1, b) for b in range(min(b1, b2) + 1, max(b1, b2))]
                    r0, r1 = rs[a1], rs[a1] + oh - 1
                    c0, c1 = cs[min(b1, b2)], cs[max(b1, b2)] + ow - 1
                elif b1 == b2:
                    mids = [(a, b1) for a in range(min(a1, a2) + 1, max(a1, a2))]
                    r0, r1 = rs[min(a1, a2)], rs[max(a1, a2)] + oh - 1
                    c0, c1 = cs[b1], cs[b1] + ow - 1
                else:
                    continue
                if not any(cur[a][b] != col for a, b in mids):
                    continue  # nothing between them still needs this colour
                cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
                ops.append(col)
                sels.append(sel_of(cells))
                for r, c in cells:
                    G[r, c] = col
                for a, b in mids:
                    cur[a][b] = col

    # --- the lattice stays on top: repaint all separator rows/columns at once
    lattice = [(r, c) for r in range(hi) for c in range(wi)
               if r in line_rows or c in line_cols]
    if any(G[r, c] != linc for r, c in lattice):
        ops.append(linc)
        sels.append(sel_of(lattice))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
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
                                f"for task 06df4c85"
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
                    f"Failed to build a complete episode for task 06df4c85 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"06df4c85-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
