"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# The picture is painted from colours 1-9 only; colour 0 is reserved for the
# frame of empty rows/columns the picture sits inside (exactly as in the
# original generator).  Keeping 0 out of the palette is what makes the frame
# readable at all -- and it also keeps every Copy/Paste/Move in the route able
# to carry the whole picture.  The route below is nevertheless written so that
# a 0-coloured picture cell is put down explicitly with Color0 wherever a
# clipboard or a grabbed object could not carry it.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    npal = random.randint(4, 9)
    palette = random.sample(list(range(1, 10)), npal)
    return {"palette": palette}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, **kwargs) -> dict:
    if palette is None:
        palette = sample(interval(1, 10, 1), 5)
    cols = list(palette)

    h = unifint(diff_lb, diff_ub, (2, max(2, min(10, max_h))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(10, max_w))))
    nc = unifint(diff_lb, diff_ub, (2, max(2, min(9, len(cols)))))
    gi = canvas(-1, (h, w))
    inds = totuple(asindices(gi))
    colss = sample(cols, min(nc, len(cols)))
    for ij in inds:
        gi = fill(gi, choice(colss), {ij})
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(e for e in gi)
    h, w = shape(gi)

    # at least one axis really grows, so input and output are never identical
    fullh = unifint(diff_lb, diff_ub, (min(h + 1, max_h), max_h))
    fullw = unifint(diff_lb, diff_ub, (min(w + 1, max_w), max_w))
    fullh = max(fullh, h)
    fullw = max(fullw, w)
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))

    while h < inh or w < inw:
        opts = []
        if h < inh:
            opts.append((h, identity))
        elif w < inw:
            opts.append((w, dmirror))
        dim, mirrf = choice(opts)
        idx = randint(0, dim - 1)
        gi = mirrf(gi)
        gi = gi[:idx + 1] + gi[idx:]
        gi = mirrf(gi)
        h, w = shape(gi)

    while h < fullh or w < fullw:
        opts = []
        if h < fullh:
            opts.append(identity)
        elif w < fullw:
            opts.append(dmirror)
        mirrf = choice(opts)
        gi = mirrf(gi)
        gi = merge(tuple(sample((((0,) * width(gi),), gi), 2)))
        gi = mirrf(gi)
        h, w = shape(gi)

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off I alone): a picture sits inside a frame of empty (0) rows and
    columns; inside it every distinct row has been stretched into a run of
    identical rows, and likewise every distinct column.  Undo the stretching --
    keep one line per run of identical lines -- and throw the frame away.

    The route taken to do that is decided from the input picture itself, the way
    two people would attack the same puzzle differently:

      * crop_first  -- if the picture is comfortably inset on every side
                       (every margin >= 2) trim the frame off first and work on
                       a tight grid; otherwise work in place and crop at the end.
      * rows_first  -- start with whichever direction is longer.
      * method      -- 'squeeze': delete each run of duplicates by sliding the
                       rest of the picture over it (Move);
                       'compact': lift each block of surviving lines out and lay
                       it down where it belongs (Copy + Paste).
                       Chosen by the colour of the picture's first cell.
      * toward_end  -- gather the survivors at the top-left or at the
                       bottom-right of the picture.  Chosen by the colour of the
                       picture's last cell.

    Nothing here looks at O.
    """
    I = np.asarray(I, dtype=int)
    G = I.copy()
    ops, sels = [], []
    is_input = [True]          # True while the working grid is still == input

    # ---------------- primitives (each one also simulates its grid effect) ----
    def rect(r, c, h, w):
        return [(int(i), int(j)) for i in range(r, r + h) for j in range(c, c + w)]

    def paint(color, cells):
        """Color<color> on exactly the cells that do not already hold it."""
        cells = [(r, c) for (r, c) in cells
                 if 0 <= r < G.shape[0] and 0 <= c < G.shape[1] and G[r, c] != color]
        if not cells:
            return
        ops.append(int(color))
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = color
        is_input[0] = False

    def copy_paste(sr, sc, h, w, dr, dc):
        """Copy the block at (sr,sc) and lay it down at (dr,dc)."""
        src = I if is_input[0] else G
        block = np.array(src[sr:sr + h, sc:sc + w], dtype=int)
        tgt = np.array(G[dr:dr + h, dc:dc + w], dtype=int)
        carried = block != 0
        new = tgt.copy()
        new[carried] = block[carried]
        if not np.array_equal(new, tgt):
            ops.append(28 if is_input[0] else 29)
            sels.append(sel_of(rect(sr, sc, h, w)))
            ops.append(30)
            sels.append(sel_of([(int(dr), int(dc))]))
            G[dr:dr + h, dc:dc + w] = new
            is_input[0] = False
        # colour-0 cells are invisible to the clipboard: put them down by hand
        paint(0, [(int(dr + i), int(dc + j)) for i, j in zip(*np.where(~carried))])

    def do_move(cells, dr, dc, steps):
        """Grab the block once, then slide it `steps` cells by (dr,dc)."""
        if steps <= 0 or not cells:
            return
        grabbed = [(r, c, int(G[r, c])) for (r, c) in cells if G[r, c] != 0]
        gh, gw = G.shape
        if grabbed:
            op = 20 if dr < 0 else 21 if dr > 0 else (23 if dc < 0 else 22)
            ops.append(op)
            sels.append(sel_of(cells))               # first Move grabs the block
            for _ in range(steps - 1):
                ops.append(op)
                sels.append(sel_of([]))              # empty -> keep it grabbed
            for (r, c) in cells:
                G[r, c] = 0
            for (r, c, v) in grabbed:
                rr, cc = r + dr * steps, c + dc * steps
                if 0 <= rr < gh and 0 <= cc < gw:
                    G[rr, cc] = v
            is_input[0] = False
        # colour-0 cells of the block do not travel with the grab: place them
        held = {(r, c) for (r, c, _) in grabbed}
        tail = []
        for (r, c) in cells:
            if (r, c) not in held:
                rr, cc = r + dr * steps, c + dc * steps
                if 0 <= rr < gh and 0 <= cc < gw:
                    tail.append((rr, cc))
        paint(0, tail)

    def runs(reps):
        """Group survivors that share one shift -- such a group travels as one."""
        out, s = [], 0
        for k in range(1, len(reps) + 1):
            if k == len(reps) or reps[k] - k != reps[s] - s:
                out.append((s, k - 1))
                s = k
        return out

    # ---------------- one direction's worth of de-duplication ----------------
    def collapse(axis, reps, bx, method, toward_end):
        R0, C0, R1, C1 = bx
        base, end = (R0, R1) if axis == 0 else (C0, C1)
        lo, hi2 = (C0, C1) if axis == 0 else (R0, R1)
        span = hi2 - lo + 1
        n = end - base + 1
        k = len(reps)
        if k == n:
            return bx                                   # nothing stretched here

        def band(start, cnt):
            if cnt <= 0:
                return []
            return rect(start, lo, cnt, span) if axis == 0 else rect(lo, start, span, cnt)

        if method == 'compact':
            off = (n - k) if toward_end else 0
            groups = runs(reps)
            for (a, b) in (list(reversed(groups)) if toward_end else groups):
                src, dst, cnt = base + reps[a], base + off + a, b - a + 1
                if src == dst:
                    continue                            # already in place
                if axis == 0:
                    copy_paste(src, lo, cnt, span, dst, lo)
                else:
                    copy_paste(lo, src, span, cnt, lo, dst)
            new_base = base + off
        else:
            lens = [(reps[i + 1] if i + 1 < k else n) - reps[i] for i in range(k)]
            if not toward_end:
                cur = end
                for i in range(k - 1, -1, -1):          # lowest run first
                    m = lens[i]
                    if m < 2:
                        continue
                    blk = base + reps[i] + m            # everything past the run
                    L = cur - blk + 1
                    if L > 0:
                        do_move(band(blk, L), -1 if axis == 0 else 0,
                                0 if axis == 0 else -1, m - 1)
                    cur -= (m - 1)
                    paint(0, band(cur + 1, m - 1))      # clear what is left over
                new_base = base
            else:
                cur = base
                for i in range(k):                      # highest run first
                    m = lens[i]
                    if m < 2:
                        continue
                    s = base + reps[i]
                    L = s - cur                         # everything before the run
                    if L > 0:
                        do_move(band(cur, L), 1 if axis == 0 else 0,
                                0 if axis == 0 else 1, m - 1)
                    paint(0, band(cur, m - 1))          # clear what is left over
                    cur += (m - 1)
                new_base = base + (n - k)

        if axis == 0:
            return (new_base, C0, new_base + k - 1, C1)
        return (R0, new_base, R1, new_base + k - 1)

    # ---------------- read the picture and the duplicate runs off I ----------
    nz = np.argwhere(I != 0)
    if nz.size == 0:
        ops.append(34)
        sels.append(sel_of(rect(0, 0, I.shape[0], I.shape[1])))
        return ops, sels

    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    core = I[r0:r1 + 1, c0:c1 + 1]
    ch, cw = core.shape
    row_reps = [0] + [i for i in range(1, ch) if not np.array_equal(core[i], core[i - 1])]
    col_reps = [0] + [j for j in range(1, cw) if not np.array_equal(core[:, j], core[:, j - 1])]

    # ---------------- route, decided entirely from the input -----------------
    margins = (r0, I.shape[0] - 1 - r1, c0, I.shape[1] - 1 - c1)
    crop_first = min(margins) >= 2
    rows_first = ch >= cw
    method = 'squeeze' if int(core[0, 0]) % 2 == 0 else 'compact'
    toward_end = (int(core[-1, -1]) % 2 == 1)

    box = (r0, c0, r1, c1)
    if crop_first:
        ops.append(33)
        sels.append(sel_of(rect(r0, c0, ch, cw)))       # full rectangle: the picture
        G = G[r0:r0 + ch, c0:c0 + cw].copy()
        is_input[0] = False
        box = (0, 0, ch - 1, cw - 1)

    for axis, reps in ([(0, row_reps), (1, col_reps)] if rows_first
                       else [(1, col_reps), (0, row_reps)]):
        box = collapse(axis, reps, box, method, toward_end)

    R0, C0, R1, C1 = box
    ho, wo = R1 - R0 + 1, C1 - C0 + 1
    if (R0, C0) != (0, 0) or (ho, wo) != tuple(G.shape):
        ops.append(33)
        sels.append(sel_of(rect(R0, C0, ho, wo)))       # full rectangle: the answer
        G = G[R0:R0 + ho, C0:C0 + wo].copy()

    ops.append(34)
    sels.append(sel_of(rect(0, 0, ho, wo)))
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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
