"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def _read_runs(I):
    """From the input alone: background, the vertical run (column cj, rows
    vr0..vr1) and the horizontal run (row ri, cols hc0..hc1)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    cj = Counter(c for _, c in fg).most_common(1)[0][0]
    ri = Counter(r for r, _ in fg).most_common(1)[0][0]

    def longest(flags):
        best, cur = None, None
        for p, f in enumerate(flags):
            if f:
                cur = (cur[0], p) if cur else (p, p)
                if best is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
            else:
                cur = None
        return best

    vr0, vr1 = longest([I[r, cj] != bgc for r in range(h)])
    hc0, hc1 = longest([I[ri, c] != bgc for c in range(w)])
    return bgc, cj, vr0, vr1, ri, hc0, hc1


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lo, hi):
        a = lo + int(round((hi - lo) * diff_lb))
        b = lo + int(round((hi - lo) * diff_ub))
        return random.randint(min(a, b), max(a, b))

    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = unifint(10, max(10, max_h))
        w = unifint(10, max(10, max_w))
        ph = unifint(2, 9)
        pw = unifint(2, 9)
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        col = random.choice(remcols)
        hbar = {}
        for k in range(ph):
            r = locih + k
            hbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else random.choice(remcols)
        wbar = {}
        for k in range(pw):
            c = locjw + k
            wbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else random.choice(remcols)
        gi = [[bgc] * w for _ in range(h)]
        for (r, c), v in list(hbar.items()) + list(wbar.items()):
            gi[r][c] = v
        go = [row[:] for row in gi]
        for k in range(h // ph + 1):
            for s in (k * ph, -k * ph):
                for (r, c), v in hbar.items():
                    if 0 <= r + s < h:
                        go[r + s][c] = v
        for k in range(w // pw + 1):
            for s in (k * pw, -k * pw):
                for (r, c), v in wbar.items():
                    if 0 <= c + s < w:
                        go[r][c + s] = v
        # keep only instances whose two runs are unambiguous from the input
        if _read_runs(gi) == (bgc, locjh, locih, locih + ph - 1,
                              lociw, locjw, locjw + pw - 1):
            return {"input": gi, "output": go}


def _plan_line(L, s0, n, needed, t):
    """Shortest CopyI/CopyO/Paste sequence tiling a line of length L from the
    run at [s0, s0+n).  State = covered interval [a,b) + clipboard (phase, len).
    Returns a list of ("copy", x, y) / ("paste", p)."""
    if not needed:
        return []
    need = set(needed)

    def grow(a, b):
        # cells already right (or 0-cells, laid later) join the tiled stretch
        while a > 0 and a - 1 not in need:
            a -= 1
        while b < L and b not in need:
            b += 1
        return a, b

    start = grow(s0, s0 + n) + (None,)
    prev = {start: None}
    dq = deque([start])
    goal = None
    while dq:
        st = dq.popleft()
        a, b, clip = st
        if all(a <= p < b for p in needed):
            goal = st
            break
        nxt = []
        # copy a segment of the already-tiled stretch (anchored at either end)
        segs = set()
        for x in range(a, b):
            segs.add((x, b))
        for y in range(a + 1, b + 1):
            segs.add((a, y))
        cur = None if clip is None else tuple(t(clip[0] + k) for k in range(clip[1]))
        for x, y in segs:
            c2 = (x % n, y - x)
            if tuple(t(x + k) for k in range(y - x)) != cur:   # a copy must change the clipboard
                nxt.append(((a, b, c2), ("copy", x, y)))
        # paste the clipboard just past either end
        if clip is not None:
            ph, ln = clip
            cands = []
            p = b - ((b - ph) % n)          # largest aligned p <= b
            if p >= L:
                p -= n
            cands.append(p)
            lo = max(0, a - ln)
            cands.append(lo + ((ph - lo) % n))  # smallest aligned p >= lo
            for p in set(cands):
                if p < 0 or p >= L or p > b or p + ln < a:
                    continue
                na, nb = min(a, p), min(L, max(b, p + ln))
                if any((q < a or q >= b) and na <= q < nb for q in needed):
                    nxt.append((grow(na, nb) + (clip,), ("paste", p)))
        for s2, act in nxt:
            if s2 not in prev:
                prev[s2] = (st, act)
                dq.append(s2)
    acts = []
    st = goal
    while prev[st] is not None:
        st, act = prev[st]
        acts.append(act)
    return acts[::-1]


def derive_operations(I, O=None, examples=None):
    """Rule (read from I only): the vertical run tiles its whole column with
    period = its length, the horizontal run tiles its whole row likewise.
    Each line is laid with copy/paste of ever larger already-tiled stretches
    (CopyI of the run itself, then CopyO of the grown stretch).  ARCLE never
    copies a 0, so if a run holds colour 0 the period's 0-cells on both lines
    are laid in with one Color0 at the end."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, cj, vr0, vr1, ri, hc0, hc1 = _read_runs(I)
    nv, nh = vr1 - vr0 + 1, hc1 - hc0 + 1
    rv = [int(I[vr0 + k, cj]) for k in range(nv)]
    rh = [int(I[ri, hc0 + k]) for k in range(nh)]
    tv = lambda r: rv[(r - vr0) % nv]
    th = lambda c: rh[(c - hc0) % nh]

    MAXD = 30
    g = np.zeros((MAXD, MAXD), dtype=int); g[:h, :w] = I
    clip, cdim = np.zeros((MAXD, MAXD), dtype=int), (0, 0)
    ops, sels = [], []
    for axis in (0, 1):
        if axis == 0:
            L, s0, n, t = h, vr0, nv, tv
            cell = lambda p: (p, cj)
        else:
            L, s0, n, t = w, hc0, nh, th
            cell = lambda p: (ri, p)
        needed = [p for p in range(L) if t(p) != 0 and g[cell(p)] != t(p)]
        for act in _plan_line(L, s0, n, needed, t):
            if act[0] == "copy":
                _, x, y = act
                r0, c0 = cell(x)
                eh, ew = (y - x - 1, 0) if axis == 0 else (0, y - x - 1)
                use_input = s0 <= x and y <= s0 + n
                src = I if use_input else g[:h, :w]
                sub = src[r0:r0 + eh + 1, c0:c0 + ew + 1]
                clip[:, :] = 0
                clip[:eh + 1, :ew + 1] = np.where(sub > 0, sub, 0)
                cdim = (eh + 1, ew + 1)
                # CopyI from the untouched run, CopyO from a stretch laid by pastes;
                # selection = that 1-wide segment of the line (full rectangle)
                ops.append(28 if use_input else 29); sels.append([r0, c0, eh, ew])
            else:
                r0, c0 = cell(act[1])
                er, ec = min(r0 + cdim[0], MAXD), min(c0 + cdim[1], MAXD)
                patch = clip[:er - r0, :ec - c0]
                g[r0:er, c0:ec] = np.where(patch > 0, patch, g[r0:er, c0:ec])
                ops.append(30); sels.append([r0, c0, 0, 0])
    # the 0s of the period, which no copy can carry: every cell of the two
    # lines whose period value is 0 and which does not already hold 0
    zeros = [(r, cj) for r in range(h) if r != ri and tv(r) == 0] + \
            [(ri, c) for c in range(w) if th(c) == 0]
    zeros = [rc for rc in zeros if g[rc] != 0]
    if zeros:
        ops.append(0); sels.append(sel_of(zeros))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
