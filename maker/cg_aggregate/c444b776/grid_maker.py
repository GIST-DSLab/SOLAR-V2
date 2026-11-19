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
    # bgc and the divider-line colour are fixed for the whole episode;
    # the panel contents' colours don't affect the rule, so they stay free.
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=5, **kw) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    remcols = [c for c in range(10) if c not in (bgc, linc)]
    for _ in range(200):
        h = unifint(2, 9)
        w = unifint(2, 9)
        nh = unifint(1, 3)
        nw = unifint(1 if nh > 1 else 2, 3)
        fullh = h * nh + nh - 1
        fullw = w * nw + nw - 1
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h, w, nh, nw = 2, 2, 1, 2
        fullh, fullw = 2, 5
    numcol = unifint(1, 8)
    ccols = random.sample(remcols, numcol)
    numcels = unifint(1, (h * w) // 2)
    inds = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(inds, numcels)
    small = [[bgc] * w for _ in range(h)]
    for (i, j) in cels:
        small[i][j] = random.choice(ccols)
    gi = [[linc] * fullw for _ in range(fullh)]
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    src = random.choice(llocs)
    go = [row[:] for row in gi]
    for (a, b) in llocs:
        for i in range(h):
            for j in range(w):
                go[a + i][b + j] = small[i][j]
                gi[a + i][b + j] = small[i][j] if (a, b) == src else bgc
    return {"input": gi, "output": go}


def _full_lines(G, bgc):
    """colour -> [full uniform rows, full uniform cols] (non-background)."""
    hi, wi = G.shape
    lines = {}
    for r in range(hi):
        if len(set(G[r].tolist())) == 1 and G[r, 0] != bgc:
            lines.setdefault(int(G[r, 0]), [[], []])[0].append(r)
    for c in range(wi):
        if len(set(G[:, c].tolist())) == 1 and G[0, c] != bgc:
            lines.setdefault(int(G[0, c]), [[], []])[1].append(c)
    return lines


def derive_operations(I, O=None, examples=None):
    # Reading: the divider lines cut the grid into equal panels, and every panel
    # ends up as the overlay of all the panels. The golf entry's max-overlay
    # works the same way: only one panel has anything painted in it, so the
    # overlay just stamps that panel onto every blank one. Here the golf reading
    # and the maker's agree on the unit (one panel), the order (one stamp per
    # panel) and what stays fixed (the panel grid), so the route stays
    # copy-panel -> paste into each blank panel, plus restoring colour-0
    # content, which a paste can't write.
    # Changes from the maker's version: the divider colour is now checked
    # against the examples, so a single-colour content row isn't mistaken for a
    # divider. Zero-cell fixes now take one op per panel instead of one per
    # cell. No-op copy/pastes are skipped.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    exs = [np.asarray(e[0], dtype=int) for e in (examples or [])]

    # background: majority colour pooled over I and the demonstration inputs
    cnt = Counter(I.flatten().tolist())
    for E in exs:
        cnt.update(E.flatten().tolist())
    bgc = max(cnt, key=cnt.get)

    # divider colour: forms full lines in I and, consistently, in the examples
    lines = _full_lines(I, bgc)
    cands = set(lines)
    for E in exs:
        lc = set(_full_lines(E, bgc))
        if cands & lc:
            cands &= lc
    if cands:
        linc = max(cands, key=lambda k: (len(lines[k][0]) > 0 and len(lines[k][1]) > 0,
                                         len(lines[k][0]) + len(lines[k][1])))
        div_rows, div_cols = lines[linc]
    else:
        div_rows, div_cols = [], []

    def bands(divs, n):
        out, prev = [], 0
        for d in divs:
            if d > prev:
                out.append((prev, d - 1))
            prev = d + 1
        if prev < n:
            out.append((prev, n - 1))
        return out

    panels = [(r0, r1, c0, c1) for (r0, r1) in bands(div_rows, hi)
              for (c0, c1) in bands(div_cols, wi)]
    src = next((p for p in panels
                if np.any(I[p[0]:p[1] + 1, p[2]:p[3] + 1] != bgc)), None)

    ops, sels = [], []
    if src is None:
        return [34], [[0, 0, hi - 1, wi - 1]]

    sr0, sr1, sc0, sc1 = src
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    S = I[sr0:sr1 + 1, sc0:sc1 + 1]
    # a paste changes a blank panel only if the source holds nonzero, non-bg cells
    has_paint = bool(np.any((S != 0) & (S != bgc)))
    if has_paint:
        # the painted panel, as a full rectangle, goes to the clipboard
        ops.append(28); sels.append([sr0, sc0, sh - 1, sw - 1])
    # colour-0 content of the source (Paste can't write 0s when bgc != 0)
    zero = [(dr, dc) for dr in range(sh) for dc in range(sw)
            if bgc != 0 and S[dr, dc] == 0]

    for p in panels:
        if p == src:
            continue
        r0, c0 = p[0], p[2]
        if has_paint:
            ops.append(30); sels.append([r0, c0, 0, 0])     # stamp the panel here
        if zero:
            # the landing cells of the source's 0-valued content in this panel
            ops.append(0)
            sels.append(sel_of([(r0 + dr, c0 + dc) for dr, dc in zero]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
