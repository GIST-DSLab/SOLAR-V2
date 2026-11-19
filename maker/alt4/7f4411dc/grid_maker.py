"""
ARC Task: 7f4411dc (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # bgc / fgc are the only randomly sampled colors of the generator.
    # 0 is excluded from both roles: CopyI/Paste treat 0 as "nothing", and a
    # full-foreground Color0 would be indistinguishable from a grid reset.
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < nsq:
        oh = randint(2, 6)
        ow = randint(2, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        obj = shift(obj, loc)
        if obj.issubset(inds):
            go = fill(go, fgc, obj)
            succ += 1
            inds = (inds - obj) - outbox(obj)
        tr += 1
    inds = ofcolor(go, bgc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    nnoise = max(1, nnoise)          # keep input != output (speck removal visible)
    gi = tuple(e for e in go)
    for k in range(nnoise):
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        inds = inds - dneighbors(loc)
        gi = fill(gi, fgc, {loc})
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
def _block_cells(A, c):
    """cells belonging to at least one 2x2 square entirely of color c."""
    h, w = A.shape
    S = set()
    for r in range(h - 1):
        for k in range(w - 1):
            if A[r, k] == c and A[r, k + 1] == c and A[r + 1, k] == c and A[r + 1, k + 1] == c:
                S.add((r, k)); S.add((r, k + 1)); S.add((r + 1, k)); S.add((r + 1, k + 1))
    return S


def _components(cells):
    """4-connected components of a cell set."""
    rest = set(cells)
    out = []
    while rest:
        seed = next(iter(rest))
        rest.discard(seed)
        comp = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in rest:
                    rest.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _as_pair(ex):
    if isinstance(ex, dict):
        return np.asarray(ex.get('input'), dtype=int), np.asarray(ex.get('output'), dtype=int)
    return np.asarray(ex[0], dtype=int), np.asarray(ex[1], dtype=int)


def _detect_block_color(I, examples):
    """Which color forms the solid blocks that survive (the other is background)."""
    cols = sorted(set(I.flatten().tolist()))
    # preferred: read the convention off the demonstrations
    votes = Counter()
    if examples:
        for ex in examples:
            try:
                Ie, Oe = _as_pair(ex)
            except Exception:
                continue
            if Ie.shape != Oe.shape:
                continue
            for c in sorted(set(Ie.flatten().tolist())):
                S = _block_cells(Ie, c)
                kept = {(int(r), int(k)) for r, k in zip(*np.where(Oe == c))}
                if S and S == kept:
                    votes[c] += 1
    for c, _ in votes.most_common():
        if c in cols:
            return c
    # fallback: structural signature of the block color inside I alone --
    # its 2x2-covered cells form perfect solid rectangles and every remaining
    # cell of that color is an isolated speck.
    best = None
    for c in cols:
        S = _block_cells(I, c)
        if not S:
            continue
        comps = _components(S)
        if not all(len(cp) == _bbox(cp)[2] * _bbox(cp)[3] for cp in comps):
            continue
        rest = {(int(r), int(k)) for r, k in zip(*np.where(I == c))} - S
        if any((r + 1, k) in rest or (r, k + 1) in rest for (r, k) in rest):
            continue
        score = (len(rest), -len(S))
        if best is None or score > best[0]:
            best = (score, c)
    if best is not None:
        return best[1]
    return cols[-1]


def derive_operations(I, O, examples=None):
    """The picture is a background sheet carrying a few solid blocks; the loose
    specks are not part of it.  So: wipe the foreground off the sheet, then
    stamp each block back with CopyI/Paste (one copy per distinct block size,
    pasted at every place that size occurs)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    if np.array_equal(I, O):
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    fgc = _detect_block_color(I, examples)
    others = [c for c in Counter(I.flatten().tolist()).most_common() if c[0] != fgc]
    bgc = others[0][0] if others else 0

    fg_cells = sorted((int(r), int(k)) for r, k in zip(*np.where(I == fgc)))
    blocks = _block_cells(I, fgc)
    comps = _components(blocks)
    rects = []
    solid = True
    for cp in comps:
        r0, c0, bh, bw = _bbox(cp)
        if bh * bw != len(cp):
            solid = False
            break
        rects.append((r0, c0, bh, bw))

    if fgc == 0 or not rects or not solid:
        # degenerate fallback: just take the loose specks off the sheet
        specks = [p for p in fg_cells if p not in blocks]
        if specks:
            ops.append(int(bgc)); sels.append(sel_of(specks))
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # 1. clear the sheet: every cell the foreground currently occupies
    #    (already-background cells are left alone; the block cells stay in the
    #    selection because they are re-stamped on top afterwards).
    ops.append(int(bgc)); sels.append(sel_of(fg_cells))

    # 2. stamp the blocks back, grouped by size: one CopyI per distinct size,
    #    one Paste per occurrence of that size.
    rects.sort(key=lambda t: (t[0], t[1]))
    groups = {}
    order = []
    for (r0, c0, bh, bw) in rects:
        key = (bh, bw)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((r0, c0))
    for key in order:
        bh, bw = key
        r0, c0 = groups[key][0]
        # bbox selection == exactly the solid block (full rectangle of fgc)
        ops.append(28); sels.append([r0, c0, bh - 1, bw - 1])
        for (r, c) in groups[key]:
            ops.append(30); sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
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
                                f"for task 7f4411dc"
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
                    f"Failed to build a complete episode for task 7f4411dc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7f4411dc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
