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
from collections import deque
import numpy as np
from maker.sel_helpers import sel_of

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)
    rem = [c for c in cols if c != fixc]
    random.shuffle(rem)
    return {"fixc": fixc, "ccols": rem}


def generate(diff_lb, diff_ub, max_h, max_w, fixc=None, ccols=None) -> dict:
    def unifint(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return lo
        a = lo + round((hi - lo) * diff_lb)
        b = lo + round((hi - lo) * diff_ub)
        return random.randint(min(a, b), max(a, b))
    if fixc is None:
        fixc = random.randint(0, 9)
    if ccols is None:
        ccols = [c for c in range(10) if c != fixc]
        random.shuffle(ccols)
    h = unifint(7, max(7, min(30, max_h)))
    w = unifint(7, max(7, min(30, max_w)))
    p = unifint(2, min(9, min(h // 3, w // 3)))
    cc = list(ccols[:p])
    diag = random.choice((True, False))
    go = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            go[r, c] = cc[((c + r) if diag else c) % p]
    gi = go.copy()

    def n_occ(g):
        cnt = 0
        for r in range(h):
            for c in range(w - p + 1):
                if all(g[r, c + j] == cc[j] for j in range(p)):
                    cnt += 1
        return cnt

    nsq = unifint(1, max(1, (h * w) // 25))
    maxtr, tr, succ = 4 * nsq, 0, 0
    while succ < nsq and tr < maxtr:
        oh = unifint(2, 5)
        ow = unifint(2, 5)
        li = random.randint(0, h - oh)
        lj = random.randint(0, w - ow)
        t = gi.copy()
        t[li:li + oh, lj:lj + ow] = fixc
        if n_occ(t) > 1 and any(fixc not in t[r] for r in range(h)) \
                and any(fixc not in t[:, c] for c in range(w)):
            gi = t
            succ += 1
        tr += 1
    if random.choice((True, False)):
        gi = np.rot90(gi, 3)
        go = np.rot90(go, 3)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading: the canvas is ONE colour strip laid down again and again, each
    # copy translated by a fixed lattice step. An occluding patch is repaired
    # by taking a clean translated copy of the wallpaper (a lattice step away)
    # from the INPUT and pasting it over the patch's rectangle.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

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
            pattern_cols |= set(rows[r])
    for c in range(wi):
        if col_p[c] == pv:
            pattern_cols |= set(colsl[c])

    palette = sorted(set(I.flatten().tolist()))
    R, C = np.indices((hi, wi))
    cands = []
    for (a, b) in _DIRS:
        lat = a * R + b * C
        for p in range(2, 10):
            key = lat % p
            for k in palette:
                keep = I != k
                mapping = {}
                bad = False
                for cls in range(p):
                    u = np.unique(I[(key == cls) & keep])
                    if len(u) > 1:
                        bad = True
                        break
                    if len(u) == 1:
                        mapping[cls] = int(u[0])
                if bad or not mapping:
                    continue
                if len(set(mapping.values())) != len(mapping):
                    continue
                if len(np.unique(key[I == k])) < 2:
                    continue
                cands.append((k not in pattern_cols, len(mapping) == p,
                              int(keep.sum()), -p, k, (a, b), p, mapping))
    cands.sort(reverse=True, key=lambda t: t[:4])
    if not cands:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    _, _, _, _, noise, (a, b), p, mapping = cands[0]
    key = (a * R + b * C) % p

    def target(r, c):
        return mapping.get(int(key[r, c]))

    noise_cells = {(r, c) for r in range(hi) for c in range(wi)
                   if I[r, c] == noise and target(r, c) is not None
                   and target(r, c) != noise}

    # occluding patches = 4-connected components of occluder cells
    rem = set(noise_cells)
    patches = []
    for r in range(hi):
        for c in range(wi):
            if (r, c) not in rem:
                continue
            comp, dq = [], deque([(r, c)])
            rem.discard((r, c))
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (rr + dr, cc + dc)
                    if nb in rem:
                        rem.discard(nb)
                        dq.append(nb)
            patches.append(sorted(comp))

    # lattice translations that map the wallpaper onto itself
    shifts = [(dr, dc) for dr in range(-hi + 1, hi) for dc in range(-wi + 1, wi)
              if (dr, dc) != (0, 0) and (a * dr + b * dc) % p == 0]
    shifts.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s))

    cur = I.copy()
    for comp in patches:
        todo = [x for x in comp if cur[x] == noise]
        if not todo:
            continue
        r0 = min(r for r, _ in comp); r1 = max(r for r, _ in comp)
        c0 = min(c for _, c in comp); c1 = max(c for _, c in comp)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        best = None
        for dr, dc in shifts:
            sr, sc = r0 + dr, c0 + dc
            if sr < 0 or sc < 0 or sr + bh > hi or sc + bw > wi:
                continue
            src = I[sr:sr + bh, sc:sc + bw]
            if noise != 0 and (src == noise).any():
                continue          # source copy must be clean wallpaper
            fixed = sum(1 for (r, c) in todo if src[r - r0, c - c0] != 0
                        and src[r - r0, c - c0] != noise)
            if fixed == len(todo):
                best = (sr, sc); break
            if fixed > 0 and (best is None or fixed > best[2]):
                best = (sr, sc, fixed)
        if best is not None:
            sr, sc = best[0], best[1]
            src = I[sr:sr + bh, sc:sc + bw]
            # full rectangle: clean translated copy of the wallpaper
            ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])   # CopyI
            ops.append(30); sels.append([r0, c0, 0, 0])             # Paste over patch box
            sub = cur[r0:r0 + bh, c0:c0 + bw]
            sub[src != 0] = src[src != 0]
        # cells Paste cannot write (wallpaper colour 0, or no clean source)
        left = [x for x in comp if cur[x] == noise]
        groups, order = {}, []
        for x in left:
            v = target(*x)
            if v not in groups:
                groups[v] = []; order.append(v)
            groups[v].append(x)
        for v in order:
            ops.append(int(v)); sels.append(sel_of(groups[v]))
            for x in groups[v]:
                cur[x] = v

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
