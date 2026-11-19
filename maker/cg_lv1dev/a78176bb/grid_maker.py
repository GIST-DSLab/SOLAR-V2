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
import numpy as np
from collections import Counter, deque
from maker.sel_helpers import sel_of

FLIP_VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(FLIP_VARIANTS):
        ex = [dict(v) for v in FLIP_VARIANTS]
        ex += [dict(random.choice(FLIP_VARIANTS)) for _ in range(n_ex - len(FLIP_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(FLIP_VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirrored=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + d * (b - a)))))

    if mirrored is None:
        mirrored = random.choice([False, True])
    remcols = [c for c in range(10) if c != bgc]
    for _attempt in range(100):
        h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
        w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
        nlns = unifint(diff_lb, diff_ub, (1, (h + w) // 8))
        full = {(i, j) for i in range(h) for j in range(w)}
        inds = set(full)
        gi = np.full((h, w), bgc, dtype=int)
        go = np.full((h, w), bgc, dtype=int)

        def nbrs(cells):
            out = set()
            for (i, j) in cells:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di or dj:
                            out.add((i + di, j + dj))
            return out

        spopts = [(i, 0) for i in range(h - 5, -1, -1)] + [(0, j) for j in range(1, w - 4)]
        succ, tr = 0, 0
        while succ < nlns and tr < 10 * nlns and spopts:
            tr += 1
            sp = random.choice(spopts)
            ln = {(sp[0] + k, sp[1] + k) for k in range(max(h, w))} & full
            if not ln.issubset(inds):
                continue
            lno = sorted(ln)
            if len(lno) < 5:
                continue
            t1 = random.randint(2, min(5, len(lno) - 3))
            t2 = random.randint(2, min(5, len(lno) - 3))
            a1 = random.choice(lno[1:-t1 - 1])
            tri1 = {(a1[0] + i, a1[1] + 1 + j) for i in range(t1) for j in range(t1) if j >= i}
            a2 = random.choice(lno[1:-t2 - 1])
            tri2 = {(a2[0] + 1 + i, a2[1] + j) for i in range(t2) for j in range(t2) if i >= j}
            s2 = (sp[0], sp[1] - t2 - 2)
            nex2 = {(s2[0] + k, s2[1] + k) for k in range(max(h, w))} & full
            s1 = (sp[0] - t1 - 2, sp[1])
            nex1 = {(s1[0] + k, s1[1] + k) for k in range(max(h, w))} & full
            inds0 = set(inds)
            lncol = None
            items = [(tri1, nex1), (tri2, nex2)]
            random.shuffle(items)
            for idx, (tri, nex) in enumerate(items):
                fullobj = ln | tri | nex
                if lncol is None:
                    lncol, tricol = random.sample(remcols, 2)
                else:
                    tricol = random.choice([c for c in remcols if c != lncol])
                if fullobj.issubset(inds0) and fullobj.issubset(inds | ln):
                    succ += 1
                    inds = (inds - fullobj) - nbrs(fullobj)
                    for (i, j) in tri:
                        gi[i, j] = tricol
                    for (i, j) in ln:
                        gi[i, j] = lncol
                        go[i, j] = lncol
                    for (i, j) in nex:
                        go[i, j] = lncol
        if succ == 0:
            continue
        if mirrored:
            gi = gi[::-1, :]
            go = go[::-1, :]
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    A = np.asarray(I, dtype=int)
    h, w = A.shape
    # Background: the colour every demonstration canvas (and I) is laid on.
    bc = Counter(A.flatten().tolist())
    if examples:
        for ei, _ in examples:
            bc.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = int(bc.most_common(1)[0][0])

    def comps(G, conn8):
        seen = np.zeros((h, w), dtype=bool)
        d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dirs = d4 + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else [])
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or G[r, c] == bgc:
                    continue
                col = int(G[r, c])
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and G[ny, nx] == col:
                            seen[ny, nx] = True; q.append((ny, nx))
                out.append((col, set(cells)))
        return out

    # Orientation: guide lines run either down-right (r-c constant) or down-left (r+c constant).
    dr_key = lambda r, c: r - c
    dl_key = lambda r, c: r + c
    key = dl_key
    for col, cells in comps(A, True):
        if len(cells) >= 3:
            rs = sorted(cells)
            if all((rs[k + 1][0] - rs[k][0], rs[k + 1][1] - rs[k][1]) == (1, 1)
                   for k in range(len(rs) - 1)):
                key = dr_key
                break

    # Markers: 4-connected one-colour blobs spanning more than one diagonal.
    markers = [(col, cells) for col, cells in comps(A, False)
               if len({key(r, c) for r, c in cells}) > 1]

    ops, sels = [], []
    for col, cells in markers:
        # The marker covers a band of diagonals [lo, hi]; its guide line is the
        # diagonal touching it on one side. The new line goes one gap past the other side.
        ks = [key(r, c) for r, c in cells]
        lo, hi = min(ks), max(ks)
        guide = None
        for r, c in sorted(cells):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                y, x = r + dy, c + dx
                if 0 <= y < h and 0 <= x < w and (y, x) not in cells and A[y, x] != bgc:
                    guide = (y, x); break
            if guide:
                break
        if guide is None:
            continue
        gcol = int(A[guide])
        gk = key(*guide)
        target = lo - 2 if gk > hi else hi + 2
        line = [(r, c) for r in range(h) for c in range(w) if key(r, c) == target]
        # Shoot the parallel full diagonal in the guide's colour, then retire the marker.
        if line:
            ops.append(gcol); sels.append(sel_of(line))
        ops.append(10 + bgc); sels.append(sel_of([min(cells)]))

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
