"""
ARC Task: 91714a58 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from random import randint, choice, sample

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _comps(G, bgc):
    """4-connected, single-colour components of the non-background cells."""
    h = len(G)
    w = len(G[0])
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or G[r][c] == bgc:
                continue
            col = G[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and G[nx][ny] == col:
                        seen[nx][ny] = True
                        stack.append((nx, ny))
            out.append((col, cells))
    return out


def _max_other_comp(G, bgc, rect):
    """Largest component that is not the planted rectangle."""
    best = 0
    for col, cells in _comps(G, bgc):
        if cells and cells[0] in rect:
            continue
        best = max(best, len(cells))
    return best


def _is_full_rect(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), (r0, c0, r1, c1)


# ----------------------------------------------------------- sample_colors ---
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, targc = sample(cols, 2)
    return {"bgc": bgc, "targc": targc}


# ----------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, targc) -> dict:
    hb = max(6, int(max_h))
    wb = max(6, int(max_w))
    h = _unifint(diff_lb, diff_ub, (6, hb))
    w = _unifint(diff_lb, diff_ub, (6, wb))

    remcols = [c for c in range(10) if c != bgc]
    othercols = [c for c in remcols if c != targc]

    nnoise = _unifint(diff_lb, diff_ub, (1, (h * w) // 2))

    ih = randint(2, h // 2)
    iw = randint(2, w // 2)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    rect = {(i, j) for i in range(loci, loci + ih) for j in range(locj, locj + iw)}
    area = ih * iw

    # cells orthogonally touching the rectangle: never given the rectangle's colour,
    # so the rectangle stays its own object (exactly the planted block).
    border = set()
    for (i, j) in rect:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (i + di, j + dj)
            if 0 <= p[0] < h and 0 <= p[1] < w and p not in rect:
                border.add(p)

    allinds = [(i, j) for i in range(h) for j in range(w)]
    outside = [p for p in allinds if p not in rect]

    gi = None
    for _ in range(64):
        n = min(max(1, nnoise), len(allinds))
        noise = list(sample(allinds, n))
        if outside and not any(p not in rect for p in noise):
            noise.append(choice(outside))          # guarantee visible noise
        g = [[bgc] * w for _ in range(h)]
        for p in noise:
            if p in rect:
                continue
            pool = othercols if p in border else remcols
            g[p[0]][p[1]] = choice(pool)
        for (i, j) in rect:
            g[i][j] = targc
        if _max_other_comp(g, bgc, rect) < area:   # rectangle is the strict largest object
            gi = g
            break
        nnoise = max(1, nnoise // 2)
    if gi is None:
        gi = [[bgc] * w for _ in range(h)]
        p = choice(outside) if outside else (0, 0)
        gi[p[0]][p[1]] = choice(othercols)
        for (i, j) in rect:
            gi[i][j] = targc

    go = [[bgc] * w for _ in range(h)]
    for (i, j) in rect:
        go[i][j] = targc

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# -------------------------------------------------------- derive_operations ---
def derive_operations(I, O, examples=None):
    """
    Rule: the grid holds one solid block plus scattered single-cell noise.
    Keep the block, wipe everything else back to the background.

    Account used here: lift the block off the input into the clipboard,
    lay a clean background over the whole canvas, then put the block back.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    whole = [(r, c) for r in range(hi) for c in range(wi)]

    # --- background: the colour every example input is drawn on -------------
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = cnt.most_common(1)[0][0]

    # --- the colour the demonstrations keep (episode convention) -----------
    kept = None
    if examples:
        seen = set()
        ok = True
        for ex in examples:
            try:
                eo = np.asarray(ex[1], dtype=int)
            except Exception:
                ok = False
                break
            vals = set(eo.flatten().tolist()) - {bgc}
            if len(vals) == 1:
                seen.add(vals.pop())
            else:
                ok = False
        if ok and len(seen) == 1:
            kept = seen.pop()

    # --- locate the block: the largest single-colour component of I --------
    grid = I.tolist()
    comps = _comps(grid, bgc)
    if not comps or np.array_equal(I, O):
        return [34], [sel_of(whole)]

    cands = [t for t in comps if kept is not None and t[0] == kept]
    if not cands:
        cands = comps
    best = max(cands, key=lambda t: (len(t[1]), 1 if _is_full_rect(t[1])[0] else 0))
    targc, cells = best[0], sorted(best[1])
    solid, (r0, c0, r1, c1) = _is_full_rect(cells)

    ops, sels = [], []

    if solid and targc != 0:
        # 1. lift the block out of the INPUT into the clipboard
        #    (selection is exactly the block = exactly its bounding rectangle)
        ops.append(28)
        sels.append(sel_of(cells))
        # 2. lay a clean background over the whole canvas (erases all the noise)
        ops.append(int(bgc))
        sels.append(sel_of(whole))
        # 3. set the block back down at its own place
        ops.append(30)
        sels.append(sel_of([(r0, c0)]))
    else:
        # clipboard can't carry colour 0 / a ragged block: lay the background,
        # then draw the block back on top of it in one stroke.
        ops.append(int(bgc))
        sels.append(sel_of(whole))
        ops.append(int(targc))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append(sel_of(whole))
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
                        f"num_examples+1 ({num_examples + 1}) for task 91714a58"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 91714a58"
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
                                f"for task 91714a58"
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
                    f"Failed to build a complete episode for task 91714a58 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"91714a58-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
