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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly two colours: the canvas background and the
    # colour of the solid rectangle.  Both are fixed for the whole episode.
    # The per-cell noise colours are irrelevant to the rule (only "is it noise
    # or is it the solid block" matters), so they stay random inside generate().
    cols = list(range(10))
    bgc = random.choice(cols)
    targc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "targc": targc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, targc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    lo_h = min(6, max_h)
    lo_w = min(6, max_w)
    h = unifint(diff_lb, diff_ub, (lo_h, max_h))
    w = unifint(diff_lb, diff_ub, (lo_w, max_w))

    remcols = [c for c in range(10) if c != bgc]
    noisecols_nt = [c for c in remcols if c != targc]

    ih = random.randint(2, max(2, h // 2))
    iw = random.randint(2, max(2, w // 2))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)

    nnoise = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    noise = random.sample(inds, min(nnoise, len(inds)))

    gi = [[bgc] * w for _ in range(h)]
    for (i, j) in noise:
        gi[i][j] = random.choice(remcols)

    blockcells = set()
    for i in range(loci, loci + ih):
        for j in range(locj, locj + iw):
            gi[i][j] = targc
            blockcells.add((i, j))

    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # No block-coloured noise glued to the block: that would turn the block's
    # connected component into a non-rectangle (and desynchronise the declarative
    # spec from the generator's output).
    for i in range(h):
        for j in range(w):
            if (i, j) in blockcells or gi[i][j] != targc:
                continue
            if any((i + da, j + db) in blockcells for da, db in nbrs):
                gi[i][j] = random.choice(noisecols_nt)

    def components(g):
        seen = [[False] * w for _ in range(h)]
        comps = []
        for i in range(h):
            for j in range(w):
                if seen[i][j] or g[i][j] == bgc:
                    continue
                col = g[i][j]
                stack = [(i, j)]
                seen[i][j] = True
                cur = []
                while stack:
                    a, b = stack.pop()
                    cur.append((a, b))
                    for da, db in nbrs:
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na][nb] and g[na][nb] == col:
                            seen[na][nb] = True
                            stack.append((na, nb))
                comps.append(cur)
        return comps

    # The kept object must be unambiguously THE biggest object: shrink any noise
    # blob that would rival the rectangle.
    area = ih * iw
    for _ in range(200):
        bad = [cp for cp in components(gi) if set(cp) != blockcells and len(cp) >= area]
        if not bad:
            break
        for cp in bad:
            k = len(cp) - area + 1
            for (a, b) in random.sample(cp, min(k, len(cp))):
                gi[a][b] = bgc

    # Guarantee the transformation is non-trivial: at least one noise cell
    # outside the rectangle, so input != output.
    outside = [(i, j) for i in range(h) for j in range(w)
               if (i, j) not in blockcells and gi[i][j] != bgc]
    if not outside:
        cands = [(i, j) for i in range(h) for j in range(w)
                 if (i, j) not in blockcells
                 and not any((i + da, j + db) in blockcells for da, db in nbrs)]
        i, j = random.choice(cands)
        gi[i][j] = random.choice(noisecols_nt)

    go = [[bgc] * w for _ in range(h)]
    for (i, j) in blockcells:
        go[i][j] = targc

    return {
        'input': tuple(tuple(r) for r in gi),
        'output': tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    """
    Rule: the grid holds one solid monochrome rectangle (>= 2x2) buried in
    single-cell colour noise.  Everything that is not that rectangle is wiped
    back to the background; the rectangle itself is kept, untouched, where it is.

    Route: grab the rectangle off the INPUT (CopyI) so it survives the wipe,
    wipe the whole canvas to background in one stroke, then put the rectangle
    back down on that clean base.  Preserve-then-destroy-then-restore; the base
    fill is deliberately whole-canvas (layering), it is not subtracted around
    the block.

    Note on geometry: the declarative spec's mirror only builds the 3x1 twin of
    a 1x3 template used to strip one-cell-thin appendages from the kept object.
    A >= 2x2 solid rectangle has no such appendage, and the rectangle is a
    single uniform colour, so any Flip/Rotate of it (or of its region) leaves
    the grid bit-identical -- a no-op op.  No geometric op is emitted, because
    none of them would move a single pixel here.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []
    whole = [(r, c) for r in range(hi) for c in range(wi)]

    # Background: the colour the canvas is made of (dominant in O, where only
    # the kept rectangle is left).
    bgc = Counter(O.flatten().tolist()).most_common(1)[0][0]

    # Locate the kept object in I: the largest same-coloured 4-connected
    # component that exactly fills its own bounding box, with both sides >= 2.
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            rs = [a for a, _ in cells]
            cs = [b for _, b in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            bh, bw = r1 - r0 + 1, c1 - c0 + 1
            if bh >= 2 and bw >= 2 and len(cells) == bh * bw:
                if best is None or bh * bw > best[4] * best[3]:
                    best = (r0, c0, bh, bw, len(cells) // max(1, bh), col)

    if best is not None:
        r0, c0, bh, bw = best[0], best[1], best[2], best[3]
        targc = best[5]
    else:  # defensive fallback: read the kept region straight off the canvas
        nz = np.argwhere(O != bgc)
        r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
        bh = int(nz[:, 0].max()) - r0 + 1
        bw = int(nz[:, 1].max()) - c0 + 1
        targc = int(O[r0, c0])

    block_cells = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]

    if np.array_equal(I, O):
        ops.append(34); sels.append(sel_of(whole))
        return ops, sels

    if targc != 0:
        # CopyI: hold on to the rectangle (exact rectangle of cells) before the wipe.
        ops.append(28); sels.append(sel_of(block_cells))
        # Wipe the entire canvas back to background -- this is the rule's erasure.
        ops.append(int(bgc)); sels.append(sel_of(whole))
        # Put the rectangle back down, unchanged, at its own position.
        ops.append(30); sels.append(sel_of([(r0, c0)]))
    else:
        # Clipboard treats 0 as "nothing", so a black rectangle is redrawn instead.
        ops.append(int(bgc)); sels.append(sel_of(whole))
        ops.append(0); sels.append(sel_of(block_cells))

    ops.append(34); sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))
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
