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
import random
import numpy as np
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colours: bgc = canvas colour, fgc = colour of the solid blocks AND the speckle
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC port: hardcoded 30s -> max_h/max_w, colours injected)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
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
    bginds = ofcolor(go, bgc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(bginds) // 2 - 1)))
    gi = tuple(e for e in go)
    for k in range(nnoise):
        if len(bginds) == 0:
            break
        loc = choice(totuple(bginds))
        bginds = bginds - dneighbors(loc)
        gi = fill(gi, fgc, {loc})
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
#    Rule (read off the examples): the grid carries solid blocks of fgc plus
#    isolated fgc speckle.  A cell survives iff it belongs to a 2x2 patch that is
#    entirely fgc -- i.e. exactly the solid blocks survive, all speckle dies.
#
#    Account of the answer: wipe the board to the background (one stroke), then
#    put each solid block back, copied straight out of the input (CopyI reads the
#    ORIGINAL input, which the wipe never touches).  The work is grouped by the
#    objects the rule is about: one block, one copy, one paste.
# ----------------------------------------------------------------------------
def _keep_cells(G, fg):
    """Cells covered by at least one 2x2 patch that is entirely `fg`."""
    h, w = G.shape
    keep = set()
    for r in range(h - 1):
        for c in range(w - 1):
            if G[r, c] == fg and G[r, c + 1] == fg and G[r + 1, c] == fg and G[r + 1, c + 1] == fg:
                keep.add((r, c))
                keep.add((r, c + 1))
                keep.add((r + 1, c))
                keep.add((r + 1, c + 1))
    return keep


def _predict(G, fg, bg):
    out = np.full(G.shape, bg, dtype=int)
    for (r, c) in _keep_cells(G, fg):
        out[r, c] = fg
    return out


def _as_pairs(examples):
    pairs = []
    if not examples:
        return pairs
    for ex in examples:
        if isinstance(ex, dict):
            a, b = ex.get("input"), ex.get("output")
        else:
            a, b = ex[0], ex[1]
        if a is None or b is None:
            continue
        pairs.append((np.asarray(a, dtype=int), np.asarray(b, dtype=int)))
    return pairs


def _roles(I, O, examples):
    """Which colour is the block/speckle colour, which is the canvas colour.
    Decided from the demonstrations (same episode => same colour roles)."""
    cols = sorted(set(I.flatten().tolist()) | set(O.flatten().tolist()))
    pairs = _as_pairs(examples)
    if pairs:
        for fg in cols:
            for bg in cols:
                if bg == fg:
                    continue
                if all(np.array_equal(_predict(a, fg, bg), b) for a, b in pairs):
                    return fg, bg
    for fg in cols:
        for bg in cols:
            if bg != fg and np.array_equal(_predict(I, fg, bg), O):
                return fg, bg
    bg = Counter(O.flatten().tolist()).most_common(1)[0][0]
    rest = [c for c in cols if c != bg]
    return (rest[0] if rest else bg), bg


def _blocks(keep):
    """Split the surviving cells into their connected blocks (4-connectivity)."""
    rem = set(keep)
    out = []
    while rem:
        seed = min(rem)
        comp = []
        dq = deque([seed])
        rem.discard(seed)
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in rem:
                    rem.discard((nr, nc))
                    dq.append((nr, nc))
        out.append(sorted(comp))
    out.sort(key=lambda cells: (cells[0][0], cells[0][1]))
    return out


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    whole = [(r, c) for r in range(h) for c in range(w)]

    if np.array_equal(I, O):
        # nothing to clean: the grid already shows only its solid blocks
        ops.append(34); sels.append(sel_of(whole))
        return ops, sels

    fgc, bgc = _roles(I, O, examples)
    blocks = _blocks(_keep_cells(I, fgc))

    # (1) clear the whole board to the background -- this is the base layer the
    #     blocks get drawn back onto (selection = the whole grid, a full rectangle).
    ops.append(int(bgc)); sels.append(sel_of(whole))

    # (2) one solid block at a time: lift it out of the untouched input and set it
    #     back down in its own place.
    for cells in blocks:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        r1, c1 = max(rs), max(cs)
        solid_rect = (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells)
        if fgc != 0 and solid_rect:
            # selection is exactly the block (a full rectangle of fgc cells)
            ops.append(28); sels.append(sel_of(cells))          # CopyI: block out of the input
            ops.append(30); sels.append(sel_of([(r0, c0)]))     # Paste: back at its own corner
        else:
            # fgc == 0 is invisible to Copy/Paste -> draw the block instead
            ops.append(int(fgc)); sels.append(sel_of(cells))

    ops.append(34); sels.append(sel_of(whole))
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
