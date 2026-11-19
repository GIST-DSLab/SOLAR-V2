"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of

# The wallpaper is a p-colour lattice, either vertical stripes or a diagonal,
# and the whole grid may be rot90'd; the noise patches are rectangles of fixc.
VARIANTS = [
    {"diag": True,  "rot": False},
    {"diag": True,  "rot": True},
    {"diag": False, "rot": False},
    {"diag": False, "rot": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)                 # the occluding (noise) colour
    ccols = [c for c in cols if c != fixc]     # ordered pool for the wallpaper
    random.shuffle(ccols)
    p = random.randint(2, 9)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"fixc": fixc, "ccols": ccols, "p": p, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(int(a + (b - a) * diff_lb), int(a + (b - a) * diff_ub))


def generate(diff_lb, diff_ub, max_h, max_w, fixc=None, ccols=None, p=None,
             diag=None, rot=None) -> dict:
    cols = list(range(10))
    if fixc is None:
        fixc = random.choice(cols)
    if ccols is None:
        ccols = [c for c in cols if c != fixc]
        random.shuffle(ccols)
    ccols = [c for c in ccols if c != fixc]
    if diag is None:
        diag = random.choice((True, False))
    if rot is None:
        rot = random.choice((True, False))

    mh = min(30, max(7, max_h))
    mw = min(30, max(7, max_w))
    if rot:                                  # the grid is transposed at the end
        mh, mw = mw, mh
    pmax = max(2, min(9, mh // 3, mw // 3, len(ccols)))
    if p is None:
        p = _unifint(diff_lb, diff_ub, (2, pmax))
    p = max(2, min(p, pmax))
    lo_h = max(7, 3 * p)
    lo_w = max(7, 3 * p)
    h = min(mh, max(lo_h, _unifint(diff_lb, diff_ub, (lo_h, mh))))
    w = min(mw, max(lo_w, _unifint(diff_lb, diff_ub, (lo_w, mw))))

    base = list(ccols[:p])
    go = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            go[r, c] = base[((c + r) % p) if diag else (c % p)]
    gi = go.copy()

    def ok(g):
        occ = 0
        for r in range(h):
            for c in range(w - p + 1):
                if list(g[r, c:c + p]) == base:
                    occ += 1
        if occ <= 1:
            return False
        if not any(fixc not in set(g[r].tolist()) for r in range(h)):
            return False
        if not any(fixc not in set(g[:, c].tolist()) for c in range(w)):
            return False
        return True

    nsq = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = _unifint(diff_lb, diff_ub, (2, 5))
        ow = _unifint(diff_lb, diff_ub, (2, 5))
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        tmp = gi.copy()
        tmp[loci:loci + oh, locj:locj + ow] = fixc
        if ok(tmp):
            gi = tmp
            succ += 1
        tr += 1

    if rot:
        gi = np.rot90(gi, k=3)
        go = np.rot90(go, k=3)
    return {"input": gi.tolist(), "output": go.tolist()}


_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]      # stripes-|, stripes-—, and the two diagonals


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # ---- 1. read the periodic wallpaper off the INPUT alone --------------
    def line_period(line):
        n = len(line)
        for p in range(1, n):
            if all(line[c] == line[c + p] for c in range(n - p)):
                return p
        return n

    rows = [I[r].tolist() for r in range(hi)]
    colsl = [I[:, c].tolist() for c in range(wi)]
    row_p = [line_period(r) for r in rows]
    col_p = [line_period(c) for c in colsl]
    ph, pv = min(row_p), min(col_p)
    pattern_cols = set()
    for r in range(hi):
        if row_p[r] == ph:
            pattern_cols |= set(rows[r])        # colours seen on an unoccluded row
    for c in range(wi):
        if col_p[c] == pv:
            pattern_cols |= set(colsl[c])       # ... and on an unoccluded column

    palette = sorted(set(I.flatten().tolist()))
    R, C = np.indices((hi, wi))
    cands = []
    for (a, b) in _DIRS:
        lat = a * R + b * C
        for p in range(2, 10):
            key = lat % p
            for k in palette:
                keep = I != k                   # try k as the occluding colour
                mapping = {}
                bad = False
                for cls in range(p):
                    u = np.unique(I[(key == cls) & keep])
                    if len(u) > 1:              # lattice class is not monochrome -> wrong guess
                        bad = True
                        break
                    if len(u) == 1:
                        mapping[cls] = int(u[0])
                if bad or not mapping:
                    continue
                if len(set(mapping.values())) != len(mapping):
                    continue                    # wallpaper colours are all distinct
                if len(np.unique(key[I == k])) < 2:
                    continue                    # a wallpaper colour lives on ONE class
                cands.append((k not in pattern_cols, len(mapping) == p,
                              int(keep.sum()), -p, k, (a, b), p, mapping))
    cands.sort(reverse=True, key=lambda t: t[:4])

    repairs = {}
    if cands:
        _, _, _, _, noise, (a, b), p, mapping = cands[0]
        key = (a * R + b * C) % p
        for r in range(hi):
            for c in range(wi):
                if int(I[r, c]) == noise:
                    tgt = mapping.get(int(key[r, c]))
                    if tgt is not None and tgt != noise:
                        repairs[(r, c)] = tgt

    # safety net for a grid whose wallpaper cannot be read at all
    if not repairs and (hi, wi) == (ho, wo) and not np.array_equal(I, O):
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != O[r, c]:
                    repairs[(r, c)] = int(O[r, c])

    # ---- 2. restore one occluding patch at a time, colour by colour ------
    remaining = set(repairs.keys())
    blobs = []
    for r in range(hi):
        for c in range(wi):
            if (r, c) not in remaining:
                continue
            comp = []
            dq = deque([(r, c)])
            remaining.discard((r, c))
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (rr + dr, cc + dc)
                    if nb in remaining:
                        remaining.discard(nb)
                        dq.append(nb)
            blobs.append(sorted(comp))

    for comp in blobs:
        order, groups = [], {}
        for cell in comp:
            v = repairs[cell]
            if v not in groups:
                groups[v] = []
                order.append(v)
            groups[v].append(cell)
        for v in order:
            ops.append(int(v))
            sels.append(sel_of(groups[v]))      # exact cells of this patch in this colour

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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
