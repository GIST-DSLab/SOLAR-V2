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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lb = max(0.0, min(1.0, diff_lb))
    ub = max(lb, min(1.0, diff_ub))
    d = random.uniform(lb, ub)
    return min(b, max(a, round(a + (b - a) * d)))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=8, **kwargs) -> dict:
    cols = list(range(10))
    oh = _unifint(diff_lb, diff_ub, (1, 4))
    ow = _unifint(diff_lb, diff_ub, (1, 4))
    hi_h = max(3, (max_h + 1) // (oh + 1))
    hi_w = max(3, (max_w + 1) // (ow + 1))
    numh = _unifint(diff_lb, diff_ub, (3, hi_h))
    numw = _unifint(diff_lb, diff_ub, (3, hi_w))
    remcols = [c for c in cols if c not in (bgc, linc)]
    ncols = _unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = random.sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = np.full((fullh, fullw), linc, dtype=int)
    for a in range(numh):
        for b in range(numw):
            gi[a * (oh + 1):a * (oh + 1) + oh, b * (ow + 1):b * (ow + 1) + ow] = bgc
    go = gi.copy()
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for col in ccols:
        inds = set(map(tuple, np.argwhere(go == bgc).tolist()))
        if not inds:
            break
        loc = random.choice(sorted(inds))
        narms = random.randint(1, 4)
        armdirs = random.sample(dirs, narms)
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
            aidx = _unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi[endp] = col
            for p in arm[:aidx + 1]:
                go[p] = col
            succ += 1
        if succ > 0:
            gi[loc] = col
            go[loc] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (pairwise bridging): the grid is a lattice of cells separated by uniform
    lines.  Any two markers of the same colour that sit on one row or one column of
    the lattice (same in-cell offset) form a pair, and every lattice cell strictly
    between them takes that colour.  The unit of work is the PAIR: one Color op paints
    the whole bridge between the two markers.  Pairs are taken nearest-first per
    colour, so a long pair spanning an already-bridged hub only adds what is missing.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    hlines = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    vlines = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if hlines:
        linc = int(I[hlines[0], 0])
    elif vlines:
        linc = int(I[0, vlines[0]])
    else:
        linc = None
    ph = (hi + 1) // (len(hlines) + 1)
    pw = (wi + 1) // (len(vlines) + 1)

    rest = Counter(int(v) for v in I.flatten().tolist() if v != linc)
    if not rest:
        return [34], [[0, 0, ho - 1, wo - 1]]
    bgc = rest.most_common(1)[0][0]
    G = I.copy()

    for col in sorted(c for c in rest if c != bgc):
        pts = sorted((int(r), int(c)) for r, c in np.argwhere(I == col))
        pairs = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                if a[0] == b[0] and (b[1] - a[1]) % pw == 0:
                    n = abs(b[1] - a[1]) // pw
                    s = 1 if b[1] > a[1] else -1
                    bridge = [(a[0], a[1] + s * k * pw) for k in range(1, n)]
                elif a[1] == b[1] and (b[0] - a[0]) % ph == 0:
                    n = abs(b[0] - a[0]) // ph
                    s = 1 if b[0] > a[0] else -1
                    bridge = [(a[0] + s * k * ph, a[1]) for k in range(1, n)]
                else:
                    continue
                pairs.append((n, a, b, bridge))
        pairs.sort()
        for n, a, b, bridge in pairs:
            cells = [p for p in bridge if G[p] != col]
            if not cells:
                continue
            for p in cells:
                G[p] = col
            ops.append(int(col))
            sels.append(sel_of(cells))

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
