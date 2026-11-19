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
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (oh + 1))))
    numw = unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (ow + 1))))
    remcols = [c for c in cols if c not in (bgc, linc)]
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
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
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
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
    Reading: every pair of same-coloured markers sharing a row/column (at a whole number of
    lattice pitches apart) is connected; the lattice cells strictly between the pair take
    that colour. One Color op per pair = the in-between lattice cells (nameable from I).
    Pairs are processed shortest first, so a long pair spanning already-joined pieces
    becomes a no-op and is skipped.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    hl = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    vl = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    linc = int(I[hl[0], 0]) if hl else (int(I[0, vl[0]]) if vl else None)
    ph = (hi + 1) // (len(hl) + 1)
    pw = (wi + 1) // (len(vl) + 1)
    rest = Counter(int(v) for v in I.flatten() if v != linc)
    if rest:
        bgc = rest.most_common(1)[0][0]
        G = I.copy()
        for col in sorted(c for c in rest if c != bgc):
            pts = sorted((int(r), int(c)) for r, c in np.argwhere(I == col))
            pairs = []
            for i, a in enumerate(pts):
                for b in pts[i + 1:]:
                    if a[0] == b[0] and (b[1] - a[1]) % pw == 0:
                        pairs.append((b[1] - a[1], a, b))
                    elif a[1] == b[1] and (b[0] - a[0]) % ph == 0:
                        pairs.append((b[0] - a[0], a, b))
            pairs.sort()
            for _, a, b in pairs:
                if a[0] == b[0]:
                    seg = [(a[0], c) for c in range(a[1] + pw, b[1], pw)]
                else:
                    seg = [(r, a[1]) for r in range(a[0] + ph, b[0], ph)]
                if seg and any(G[p] != col for p in seg):
                    ops.append(int(col))
                    sels.append(sel_of(seg))
                    for p in seg:
                        G[p] = col
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
