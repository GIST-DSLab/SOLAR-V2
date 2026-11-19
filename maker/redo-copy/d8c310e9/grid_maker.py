"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    try:
        return unifint(diff_lb, diff_ub, (a, b))  # noqa: F821 (re-arc utility if present)
    except Exception:
        return random.randint(a, b)


def _rot_cw(g, k):
    """k clockwise quarter turns on a list-of-lists grid."""
    g = [list(r) for r in g]
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return g


# orientation applied by the generator: 0=identity, 1=rot90(CW), 2=rot180, 3=rot270(CCW)
VARIANTS = [{"orient": 0}, {"orient": 1}, {"orient": 2}, {"orient": 3}]


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, orient=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    if not ccols:
        ccols = random.sample(remcols, random.randint(1, 9))
    ccols = [c for c in ccols if c != bgc]
    if not ccols:
        ccols = [remcols[0]]

    if orient is None:
        orient = random.choice([0, 1, 2, 3])

    # after a 90/270 rotation the dimensions swap -> sample pre-rotation dims accordingly
    if orient % 2 == 0:
        hlim, wlim = max_h, max_w
    else:
        hlim, wlim = max_w, max_h
    if wlim < 7 or hlim < 3:
        orient = 0
        hlim, wlim = max_h, max_w

    h = _unifint(diff_lb, diff_ub, (3, max(3, hlim)))
    wlo = min(10, wlim)
    w = _unifint(diff_lb, diff_ub, (wlo, max(wlo, wlim)))

    phi = (w - 1) // 3
    if phi < 2:
        p = max(1, phi)
    else:
        p = _unifint(diff_lb, diff_ub, (2, phi))
    p = max(1, p)

    # bottom-anchored vertical runs, one per column of the base tile
    obj = set()
    for j in range(p):
        numcells = _unifint(diff_lb, diff_ub, (1, max(1, h - 1)))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj.add((random.choice(ccols), (ii, j)))

    minobj = obj | {(c, (i, j + p)) for c, (i, j) in obj}
    addonw = random.randint(0, p)
    addon = {cell for cell in obj if cell[1][1] < addonw}
    fullobj = minobj | addon
    leftshift = random.randint(0, addonw)
    fullobj = {(c, (i, j - leftshift)) for c, (i, j) in fullobj}

    gi = [[bgc] * w for _ in range(h)]
    for c, (i, j) in fullobj:
        if 0 <= i < h and 0 <= j < w:
            gi[i][j] = c

    go = [row[:] for row in gi]
    for k in range(w // (2 * p) + 2):
        for c, (i, j) in fullobj:
            jj = j + k * 2 * p
            if 0 <= i < h and 0 <= jj < w:
                go[i][jj] = c

    gi = _rot_cw(gi, orient)
    go = _rot_cw(go, orient)

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """Rule (measured from I only):

    The grid holds a striped band that starts at one edge and is periodic along
    one axis.  Bring the band into its canonical orientation (empty top row and
    empty last column  ->  band anchored bottom-left, growing rightwards),
    measure its horizontal period from the band itself, then keep stamping the
    band one period further out until the grid is full.  Each repetition is
    drawn as its own group of Color ops, colour by colour.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def rot_cw(a, k):
        return np.rot90(a, -k)  # k clockwise quarter turns

    idx = np.arange(hi * wi).reshape(hi, wi)

    # --- canonical orientation: top row and right column are pure background ---
    kbest = None
    for k in range(4):
        Ck = rot_cw(I, k)
        if len(set(Ck[0, :].tolist())) == 1 and len(set(Ck[:, -1].tolist())) == 1:
            kbest = k
            break
    if kbest is None:
        C = I.copy()
        Cidx = idx
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        C = np.array(rot_cw(I, kbest))
        Cidx = np.array(rot_cw(idx, kbest))
        bgc = int(C[0, 0])

    hc, wc = C.shape

    # --- the band (everything that is not background) ---
    obj = {}
    for r in range(hc):
        for c in range(wc):
            v = int(C[r, c])
            if v != bgc:
                obj[(r, c)] = v

    if not obj:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    minr = min(r for r, _ in obj)
    minc = min(c for _, c in obj)
    maxc = max(c for _, c in obj)
    wobj = maxc - minc + 1
    norm = {(r - minr, c - minc): v for (r, c), v in obj.items()}

    # --- horizontal period of the band (smallest left-shift that maps into it) ---
    period = wobj
    for p in range(1, wobj):
        ok = True
        for (r, c), v in norm.items():
            if c - p >= 0 and norm.get((r, c - p)) != v:
                ok = False
                break
        if ok:
            period = p
            break

    # --- stamp repetition after repetition, outwards ---
    work = C.copy()
    off = period
    while minc + off < wc:
        by_color = {}
        for (r, c), v in obj.items():
            cc = c + off
            if cc < wc and int(work[r, cc]) != v:      # judged against the grid as it is now
                by_color.setdefault(v, []).append((r, cc))
        for v in sorted(by_color):
            cells = by_color[v]
            real = []
            for r, c in cells:
                lin = int(Cidx[r, c])
                real.append((lin // wi, lin % wi))
            ops.append(int(v))
            sels.append(sel_of(real))
            for r, c in cells:
                work[r, c] = v
        off += period

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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
