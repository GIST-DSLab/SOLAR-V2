"""
ARC Task: cbded52d (RE-ARC) — LLM-generated grid_maker
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
    # Background and separator-line colours are fixed per episode; marker colours are
    # irrelevant to the rule (any non-bg, non-line colour is completed the same way).
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi_), max(lo, hi_))

    oh = unifint(diff_lb, diff_ub, (1, min(4, (max_h + 1) // 3 - 1)))
    ow = unifint(diff_lb, diff_ub, (1, min(4, (max_w + 1) // 3 - 1)))
    numh = unifint(diff_lb, diff_ub, (3, min(31, max_h + 1) // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, min(31, max_w + 1) // (ow + 1)))
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = random.sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = [[linc] * fullw for _ in range(fullh)]
    for a in range(numh):
        for b in range(numw):
            for i in range(oh):
                for j in range(ow):
                    gi[a * (oh + 1) + i][b * (ow + 1) + j] = bgc
    go = [row[:] for row in gi]
    for col in ccols:
        inds = {(i, j) for i in range(fullh) for j in range(fullw) if go[i][j] == bgc}
        if not inds:
            break
        loc = random.choice(sorted(inds))
        narms = random.randint(1, 4)
        armdirs = random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], narms)
        succ = 0
        for x, y in armdirs:
            arm = []
            for k in range(1, max(numh, numw)):
                nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                if nl not in inds:
                    break
                arm.append(nl)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            r, c = arm[aidx]
            gi[r][c] = col
            for r, c in arm[:aidx + 1]:
                go[r][c] = col
            succ += 1
        if succ > 0:
            gi[loc[0]][loc[1]] = col
            go[loc[0]][loc[1]] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the golf entry and the verifier, unlike the maker's hub/arm walk):
    a lattice cell takes a colour when it lies BETWEEN two markers of that colour on the
    same row or column. The unit is a PAIR of aligned same-colour markers: each pair
    gets one Color op that selects exactly the lattice cells strictly between them
    (nameable from I: the lattice positions on the segment joining two markers).
    Pairs are taken shortest first; a pair whose in-between cells are already all
    that colour (e.g. two arm endpoints on opposite sides of the hub) is skipped,
    because painting it would change nothing.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # separator lines: any uniform row/column must be a line (every row crosses line columns)
    hl = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    vl = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    linc = int(I[hl[0], 0]) if hl else (int(I[0, vl[0]]) if vl else None)
    ph = (hi + 1) // (len(hl) + 1)
    pw = (wi + 1) // (len(vl) + 1)

    rest = Counter(int(v) for v in I.flatten().tolist() if v != linc)
    if not rest:
        return [34], [[0, 0, ho - 1, wo - 1]]
    bgc = rest.most_common(1)[0][0]

    G = I.copy()
    for col in sorted(c for c in rest if c != bgc):
        pts = sorted((int(r), int(c)) for r, c in np.argwhere(I == col))
        segs = []
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                p, q = pts[a], pts[b]
                if p[0] == q[0] and (q[1] - p[1]) % pw == 0:
                    cells = [(p[0], c) for c in range(p[1] + pw, q[1], pw)]
                elif p[1] == q[1] and (q[0] - p[0]) % ph == 0:
                    cells = [(r, p[1]) for r in range(p[0] + ph, q[0], ph)]
                else:
                    continue
                if cells:
                    segs.append(cells)
        segs.sort(key=len)
        for cells in segs:
            if all(G[r, c] == col for r, c in cells):
                continue  # this pair's span is already complete; op would be a no-op
            ops.append(int(col))
            sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task cbded52d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cbded52d"
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
                                f"for task cbded52d"
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
                    f"Failed to build a complete episode for task cbded52d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cbded52d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
