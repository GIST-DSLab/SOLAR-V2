"""
ARC Task: 80af3007 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    fgc = random.choice([c for c in range(10) if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    from random import randint, sample, choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    if bgc is None:
        bgc = random.choice(range(10))
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])
    hmax = max(2, min(5, int((max_h - 2) ** 0.5)))
    wmax = max(2, min(5, int((max_w - 2) ** 0.5)))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = [(i, j) for i in range(h) for j in range(w)]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    locs = set(sample(inds, numc)) | set(sample(corners, 3))
    pat = [[bgc] * w for _ in range(h)]
    for (i, j) in locs:
        pat[i][j] = fgc
    go = [[bgc] * (w * w) for _ in range(h * h)]
    for (a, b) in locs:
        for (i, j) in locs:
            go[a * h + i][b * w + j] = fgc
    fullh = unifint(diff_lb, diff_ub, (h ** 2 + 2, max_h))
    fullw = unifint(diff_lb, diff_ub, (w ** 2 + 2, max_w))
    gi = [[bgc] * fullw for _ in range(fullh)]
    loci = randint(1, fullh - h ** 2 - 1)
    locj = randint(1, fullw - w ** 2 - 1)
    for r in range(h * h):
        for c in range(w * w):
            gi[loci + r][locj + c] = pat[r // h][c // w]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    # Background: canvas colour; the generator always leaves a >=1 margin, so the corner is bgc.
    bgc = int(I[0, 0])
    fg = np.argwhere(I != bgc)
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    fgc = int(I[fg[0][0], fg[0][1]])
    H, W = r1 - r0 + 1, c1 - c0 + 1          # h^2 x w^2 upscaled object
    h, w = int(round(H ** 0.5)), int(round(W ** 0.5))
    # small pattern read off the block grid (each h x w block is uniform)
    pat = [[I[r0 + i * h, c0 + j * w] == fgc for j in range(w)] for i in range(h)]

    ops, sels = [], []
    # 1. Crop the canvas to the upscaled object (full rectangle, bg blocks included).
    ops.append(33); sels.append([r0, c0, H - 1, W - 1])

    holes = [(a, b) for a in range(h) for b in range(w) if not pat[a][b]]
    if not holes:
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    # source block: first solid corner block (3 corners are always solid)
    src = next(p for p in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)] if pat[p[0]][p[1]])
    i0, j0 = src
    # 2. Carve the pattern's holes (read from I) into the source block in one stroke:
    #    the block now holds the pattern at 1:1 scale.
    ops.append(bgc); sels.append(sel_of([(i0 * h + a, j0 * w + b) for a, b in holes]))
    # 3. bgc == 0: the clipboard cannot carry 0s, so wipe every other solid block first.
    if bgc == 0:
        wipe = [(r, c) for r in range(H) for c in range(W)
                if pat[r // h][c // w] and (r // h, c // w) != src]
        ops.append(0); sels.append(sel_of(wipe))

    def plan(transpose):
        n = h if not transpose else w
        m = w if not transpose else h

        def solid(k, l):
            return pat[k][l] if not transpose else pat[l][k]
        s_line = i0 if not transpose else j0
        s_pos = j0 if not transpose else i0
        groups, order = {}, []
        for k in range(n):
            key = tuple(solid(k, l) for l in range(m))
            if not any(key):
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(k)
        skey = tuple(solid(s_line, l) for l in range(m))
        groups[skey].remove(s_line)
        groups[skey].insert(0, s_line)
        acts = [('copyblock',)]
        # stamp the pattern block onto the solid blocks of one representative line per kind
        for key in order:
            k = groups[key][0]
            for l in range(m):
                if key[l] and not (k == s_line and l == s_pos):
                    acts.append(('pasteblock', k, l))
        # identical block-lines: copy the finished line once and paste it whole
        for key in order:
            rest = groups[key][1:]
            if rest:
                acts.append(('copyline', groups[key][0]))
                for k in rest:
                    acts.append(('pasteline', k))
        return acts

    best = None
    for tr in (False, True):
        a = plan(tr)
        if best is None or len(a) < len(best[1]):
            best = (tr, a)
    tr, acts = best
    for act in acts:
        if act[0] == 'copyblock':
            ops.append(29); sels.append([i0 * h, j0 * w, h - 1, w - 1])
        elif act[0] == 'pasteblock':
            k, l = act[1], act[2]
            i, j = (k, l) if not tr else (l, k)
            ops.append(30); sels.append([i * h, j * w, 0, 0])
        elif act[0] == 'copyline':
            k = act[1]
            ops.append(29)
            sels.append([k * h, 0, h - 1, W - 1] if not tr else [0, k * w, H - 1, w - 1])
        else:
            k = act[1]
            ops.append(30)
            sels.append([k * h, 0, 0, 0] if not tr else [0, k * w, 0, 0])
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 80af3007"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 80af3007"
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
                                f"for task 80af3007"
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
                    f"Failed to build a complete episode for task 80af3007 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"80af3007-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
