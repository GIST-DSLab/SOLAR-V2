"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    c1, c2, c3, c4 = random.sample([c for c in range(10) if c != bgc], 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "instance_plan": plan}


def _nfront(g):
    n = sum(1 for r in range(g.shape[0]) if len(set(g[r].tolist())) == 1)
    n += sum(1 for c in range(g.shape[1]) if len(set(g[:, c].tolist())) == 1)
    return n


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    h = _unifint(diff_lb, diff_ub, (7, max_h))
    w = _unifint(diff_lb, diff_ub, (7, max_w))
    loci1 = random.randint(1, h - 4)
    locj1 = random.randint(1, w - 4)
    loci1 -= _unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1 -= _unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci2 = _unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = _unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))
    gi = np.full((h, w), bgc, dtype=int)
    fronts = [("r", loci1, c1), ("r", loci2, c2), ("c", locj1, c3), ("c", locj2, c4)]
    random.shuffle(fronts)
    for k, i, c in fronts:
        if k == "r":
            gi[i, :] = c
        else:
            gi[:, i] = c
    cands = [tuple(x) for x in np.argwhere(gi == bgc)]
    nn = len(cands)

    def pick():
        k = _unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        return random.sample(cands, k)

    noise = pick()
    for r, c in noise:
        gi[r, c] = c1
    tries = 0
    while _nfront(gi) > 4:
        for r, c in noise:
            gi[r, c] = bgc
        noise = pick()
        for r, c in noise:
            gi[r, c] = c1
        tries += 1
        if tries > 200:
            raise ValueError("noise placement failed")
    go = gi[loci1:loci2 + 1, locj1:locj2 + 1].copy()
    for r, c in np.argwhere(go == c1):
        go[:r + 1, c] = c1
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Uses only I: O is never read.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def find_frame(bgc):
        cand_r = []
        for r in range(h):
            for v, n in Counter(I[r].tolist()).items():
                if v != bgc and n >= w - 2:
                    cand_r.append((r, v))
                    break
        cand_c = []
        for c in range(w):
            for v, n in Counter(I[:, c].tolist()).items():
                if v != bgc and n >= h - 2:
                    cand_c.append((c, v))
                    break
        best = None
        for a in range(len(cand_r)):
            for b in range(a + 1, len(cand_r)):
                r1, vt = cand_r[a]
                r2, vb = cand_r[b]
                if r2 - r1 < 2:
                    continue
                for x in range(len(cand_c)):
                    for y in range(x + 1, len(cand_c)):
                        k1, vl = cand_c[x]
                        k2, vr = cand_c[y]
                        if k2 - k1 < 2:
                            continue
                        four = [vt, vb, vl, vr]
                        if len(set(four)) != 4:
                            continue
                        ok = all(I[rr, j] == vv or j in (k1, k2)
                                 for rr, vv in ((r1, vt), (r2, vb)) for j in range(w))
                        ok = ok and all(I[i, kk] == vv or i in (r1, r2)
                                        for kk, vv in ((k1, vl), (k2, vr)) for i in range(h))
                        if not ok:
                            continue
                        pal = set(I[r1 + 1:r2, k1 + 1:k2].flatten().tolist()) - {bgc}
                        if len(pal) > 1:
                            continue
                        if len(pal) == 1 and next(iter(pal)) not in four:
                            continue
                        score = (r2 - r1) * (k2 - k1)
                        if best is None or score > best[0]:
                            best = (score, (r1, r2, k1, k2, vt, vb, vl, vr))
        return best

    bgc, frame = None, None
    for cb, _n in Counter(I.flatten().tolist()).most_common():
        frame = find_frame(cb)
        if frame is not None:
            bgc = cb
            break
    r1, r2, k1, k2, vt, vb, vl, vr = frame[1]
    H, W = r2 - r1 + 1, k2 - k1 + 1
    sub = I[r1:r2 + 1, k1:k2 + 1]

    # crop to the rectangle framed by the four lines (full rectangle -> bbox exact)
    ops, sels = [33], [[r1, k1, H - 1, W - 1]]

    pal = set(sub[1:H - 1, 1:W - 1].flatten().tolist()) - {bgc}
    if len(pal) == 1:
        nc = next(iter(pal))
        side = {vt: 'up', vb: 'down', vl: 'left', vr: 'right'}.get(nc)
        if side is not None:
            vert = side in ('up', 'down')
            g = sub if vert else sub.T
            L, N = (H, W) if vert else (W, H)
            toward_start = side in ('up', 'left')
            # each speckle shoots a ray toward the frame side of its own colour;
            # per line: from that side to the farthest speckle on the line
            rays = []
            for j in range(1, N - 1):
                hits = [i for i in range(1, L - 1) if g[i, j] == nc]
                if not hits:
                    continue
                a, b = (1, max(hits)) if toward_start else (min(hits), L - 2)
                if all(g[i, j] == nc for i in range(a, b + 1)):
                    continue
                rays.append((j, a, b))
            # adjacent lines whose rays span the same extent form one rectangle
            groups = []
            for j, a, b in rays:
                if groups and groups[-1][1] == j - 1 and (groups[-1][2], groups[-1][3]) == (a, b):
                    groups[-1][1] = j
                else:
                    groups.append([j, j, a, b])
            for j0, j1, a, b in groups:
                cells = [(i, j) for i in range(a, b + 1) for j in range(j0, j1 + 1)]
                if not vert:
                    cells = [(j, i) for i, j in cells]
                ops.append(int(nc))
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
