"""
ARC Task: a78176bb (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, mirrored=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi <= lo:
            return lo
        x = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + int(round((hi - lo) * x))))

    if mirrored is None:
        mirrored = random.choice((True, False))
    h = unifint(6, max(6, max_h))
    w = unifint(6, max(6, max_w))
    nlns = unifint(1, max(1, (h + w) // 8))
    remcols = [c for c in range(10) if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    fullinds = {(r, c) for r in range(h) for c in range(w)}
    inds = set(fullinds)

    def neighbors(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    def fill(g, col, cells):
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                g[r, c] = col

    spopts = [(i, 0) for i in range(h - 5, -1, -1)] + [(0, i) for i in range(1, w - 4)]
    succ, tr, maxtr = 0, 0, 10 * nlns
    while succ < nlns and tr < maxtr:
        tr += 1
        if not spopts:
            break
        sp = random.choice(spopts)
        ln = {(sp[0] + k, sp[1] + k) for k in range(max(h, w))} & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln, key=lambda x: x[0])
        trid1 = random.randint(2, min(5, len(lno) - 3))
        trid2 = random.randint(2, min(5, len(lno) - 3))
        tri1 = {(i, j) for i in range(trid1) for j in range(trid1) if j >= i}
        a = random.choice(lno[1:-trid1 - 1]); triloc1 = (a[0], a[1] + 1)
        tri2 = {(j, i) for i in range(trid2) for j in range(trid2) if j >= i}
        a = random.choice(lno[1:-trid2 - 1]); triloc2 = (a[0] + 1, a[1])
        spo2 = (sp[0], sp[1] - trid2 - 2)
        nexlin2 = {(spo2[0] + k, spo2[1] + k) for k in range(max(h, w))} & fullinds
        spo1 = (sp[0] - trid1 - 2, sp[1])
        nexlin1 = {(spo1[0] + k, spo1[1] + k) for k in range(max(h, w))} & fullinds
        items = [(tri1, triloc1, nexlin1), (tri2, triloc2, nexlin2)]
        random.shuffle(items)
        lncol = None
        for idx, (tri, triloc, nexlin) in enumerate(items):
            tri = {(i + triloc[0], j + triloc[1]) for i, j in tri}
            fullobj = ln | tri | nexlin
            if idx == 0:
                lncol, tricol = random.sample(remcols, 2)
            else:
                tricol = random.choice([c for c in remcols if c != lncol])
            if (fullobj.issubset(inds) if idx == 0 else True):
                succ += 1
                nb = set()
                for p in fullobj:
                    nb |= neighbors(p)
                inds = (inds - fullobj) - nb
                fill(gi, tricol, tri)
                fill(gi, lncol, ln)
                fill(go, lncol, ln)
                fill(go, lncol, nexlin)
    if mirrored:
        gi = gi[::-1, :]
        go = go[::-1, :]
    # Reject the rare scenes where an unchecked second marker collides with another
    # line so that the stated rule no longer reproduces the output (caller retries).
    ops, sels = derive_operations(gi, go)
    G = gi.copy()
    for o, s in zip(ops, sels):
        if o == 34:
            break
        for r, c in s["cells"]:
            G[r, c] = o
    if not np.array_equal(G, go):
        raise ValueError("inconsistent scene")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    A = np.asarray(I, dtype=int)
    h, w = A.shape
    # Background: canvas colour the generator paints first; the scene is sparse lines.
    bgc = int(Counter(A.flatten().tolist()).most_common(1)[0][0])

    def components(G, conn8):
        seen = np.zeros((h, w), dtype=bool)
        d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dirs = d4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else d4
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(G[r, c]) == bgc:
                    continue
                col = int(G[r, c])
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(G[ny, nx]) == col:
                            seen[ny, nx] = True; q.append((ny, nx))
                out.append((col, set(cells)))
        return out

    def is_dr_line(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        n = r1 - r0
        if n != c1 - c0 or len(cells) != n + 1:
            return False
        return cells == {(r0 + k, c0 + k) for k in range(n + 1)}

    def is_straight(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            return cells == {(r0, c) for c in range(c0, c1 + 1)}
        if c0 == c1:
            return cells == {(r, c0) for r in range(r0, r1 + 1)}
        return is_dr_line(cells)

    # Canonical frame: guide lines run down-right; otherwise the scene is up-down mirrored.
    flip = not any(len(cells) >= 3 and is_dr_line(cells) for _, cells in components(A, True))
    B = A[::-1, :].copy() if flip else A

    def back(rc):
        r, c = rc
        return (h - 1 - r, c) if flip else (r, c)

    triangles = [(col, cells) for col, cells in components(B, False) if not is_straight(cells)]

    G = A.copy()                       # working grid as it stands
    ops, sels = [], []

    def paint(col, cells):
        cells = [p for p in cells if 0 <= p[0] < h and 0 <= p[1] < w]
        if cells and any(G[r, c] != col for r, c in cells):
            ops.append(int(col)); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = col

    inb = lambda p: 0 <= p[0] < h and 0 <= p[1] < w
    for tcol, cells in triangles:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if (r0, c1) in cells:          # marker sits above-right of its guide line
            tip, off, anchor = (r0, c1), (-1, 1), (r0 + 1, c0)
        else:                          # marker sits below-left of its guide line
            tip, off, anchor = (r1, c0), (1, -1), (r0, c0 + 1)
        lcol = int(B[anchor])
        seed = (tip[0] + off[0], tip[1] + off[1])   # cell just past the marker's tip
        # The marker "fires": a new line of the guide's colour starts at the seed
        # and grows in both diagonal directions, parallel to its guide line.
        up = [p for p in ((seed[0] - k, seed[1] - k) for k in range(1, h + w)) if inb(p)]
        dn = [p for p in ((seed[0] + k, seed[1] + k) for k in range(1, h + w)) if inb(p)]
        if inb(seed):
            paint(lcol, [back(seed)])
        paint(lcol, [back(p) for p in up])
        paint(lcol, [back(p) for p in dn])
        # The spent marker is removed: whatever of it still shows its own colour.
        left = [back(p) for p in cells if G[back(p)] == tcol]
        paint(bgc, left)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a78176bb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a78176bb"
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
                                f"for task a78176bb"
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
                    f"Failed to build a complete episode for task a78176bb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a78176bb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
