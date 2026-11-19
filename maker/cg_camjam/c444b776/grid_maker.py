"""
ARC Task: c444b776 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background and the divider-line colour are episode-level roles.
    # The object colours (ccols) do not matter to the rule: the source tile is
    # simply copied as it is.
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]

    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (2, 9))
        w = unifint(diff_lb, diff_ub, (2, 9))
        nh = unifint(diff_lb, diff_ub, (1, 3))
        nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h, w, nh, nw = 2, 2, 1, 2
        fullh, fullw = 2, 5

    gi = np.full((fullh, fullw), linc, dtype=int)
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(inds, numcels)
    small = np.full((h, w), bgc, dtype=int)
    for (i, j) in cels:
        small[i, j] = random.choice(ccols)

    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)
    for (a, b) in llocs:
        gi[a:a + h, b:b + w] = small if (a, b) == srcloc else bgc
    go = gi.copy()
    for (a, b) in llocs:
        go[a:a + h, b:b + w] = small
    return {"input": gi.tolist(), "output": go.tolist()}


def _bands(n, divs):
    bands, prev = [], 0
    for d in divs:
        if d > prev:
            bands.append((prev, d - 1))
        prev = d + 1
    if prev < n:
        bands.append((prev, n - 1))
    return bands


def _structure(I, bgc, linc_hint=None):
    hi, wi = I.shape
    cands = set()
    for r in range(hi):
        if len(set(I[r, :].tolist())) == 1 and int(I[r, 0]) != bgc:
            cands.add(int(I[r, 0]))
    for c in range(wi):
        if len(set(I[:, c].tolist())) == 1 and int(I[0, c]) != bgc:
            cands.add(int(I[0, c]))
    order = sorted(cands, key=lambda x: (x != linc_hint, x))
    best = None
    for lc in order:
        drs = [r for r in range(hi) if np.all(I[r, :] == lc)]
        dcs = [c for c in range(wi) if np.all(I[:, c] == lc)]
        rb, cb = _bands(hi, drs), _bands(wi, dcs)
        if len(rb) * len(cb) < 2:
            continue
        hs = {b - a for a, b in rb}
        ws = {b - a for a, b in cb}
        if len(hs) != 1 or len(ws) != 1:
            continue
        # dividers must sit exactly between equal tiles
        th, tw = hs.pop() + 1, ws.pop() + 1
        if drs != [k * (th + 1) - 1 for k in range(1, len(rb))]:
            continue
        if dcs != [k * (tw + 1) - 1 for k in range(1, len(cb))]:
            continue
        best = (lc, rb, cb)
        break
    return best


def derive_operations(I, O, examples=None):
    # Reading: the grid is a lattice of equal tiles divided by lines. One tile
    # carries a pattern and the rest are blank. Every tile ends up as the
    # overlay of all tiles. The code-golf entry (max over congruent positions)
    # reads the task the same way: the blank tiles add nothing to the overlay,
    # so the overlay is the source tile. That makes this the maker's
    # copy-the-source-tile-into-each-other-tile route. The only change is
    # grouping: each destination tile's 0-valued cells are fixed with one
    # selection, the cells that copy lands on.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background: the majority colour, voted across the demonstration inputs
    # (they all share the episode's background).
    cnt = Counter(I.flatten().tolist())
    linc_hint = None
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = max(cnt, key=cnt.get)
    if examples:
        votes = Counter()
        for ex_i, _ in examples:
            st = _structure(np.asarray(ex_i, dtype=int), bgc)
            if st is not None:
                votes[st[0]] += 1
        if votes:
            linc_hint = votes.most_common(1)[0][0]

    st = _structure(I, bgc, linc_hint)
    if st is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    _, row_bands, col_bands = st

    src = None
    for (r0, r1) in row_bands:
        for (c0, c1) in col_bands:
            if np.any(I[r0:r1 + 1, c0:c1 + 1] != bgc):
                src = (r0, r1, c0, c1)
                break
        if src is not None:
            break
    if src is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    sr0, sr1, sc0, sc1 = src
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    tile = I[sr0:sr1 + 1, sc0:sc1 + 1]

    # The paste changes something only if the tile has a pattern cell that is
    # non-zero (cells equal to bgc land on bgc).
    paste_useful = bool(np.any((tile != bgc) & (tile != 0)))
    # Pattern cells coloured 0 cannot travel through the clipboard.
    zero_rel = [(dr, dc) for dr in range(sh) for dc in range(sw)
                if bgc != 0 and int(tile[dr, dc]) == 0]

    if paste_useful:
        # CopyI the whole source tile: a full rectangle of the lattice.
        ops.append(28); sels.append([sr0, sc0, sh - 1, sw - 1])

    for (r0, r1) in row_bands:
        for (c0, c1) in col_bands:
            if (r0, r1, c0, c1) == src:
                continue
            if paste_useful:
                ops.append(30); sels.append([r0, c0, 0, 0])
            if zero_rel:
                ops.append(0)
                sels.append(sel_of([(r0 + dr, c0 + dc) for dr, dc in zero_rel]))

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
                        f"num_examples+1 ({num_examples + 1}) for task c444b776"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c444b776"
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
                                f"for task c444b776"
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
                    f"Failed to build a complete episode for task c444b776 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c444b776-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
