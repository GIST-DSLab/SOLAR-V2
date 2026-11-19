"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d + random.uniform(-0.5, 0.5)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample([c for c in range(10) if c != 8], 2)
    lim = min(30, max_h, max_w)
    w = _unifint(diff_lb, diff_ub, (2, min(10, lim - 1)))
    h = _unifint(diff_lb, diff_ub, (w + 1, lim))
    gi = np.full((h, w), bgc, dtype=int)
    sp = (h - 1, 0)
    gi[sp] = linc
    go = gi.copy()
    direc = 1
    while True:
        sp = (sp[0] - 1, sp[1] + direc)
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        if not (0 <= sp[0] < h and 0 <= sp[1] < w):
            break
        go[sp] = linc
    mfs = (lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, -1),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1))
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi).copy()
        go = fn(go).copy()
    gix, gox = gi.copy(), go.copy()
    numlins = _unifint(diff_lb, diff_ub, (1, 4))
    mirrors = [np.flipud, np.fliplr, lambda g: np.flipud(np.fliplr(g))]
    for k in range(numlins - 1):
        f = mirrors[k]
        gi[f(gix) == linc] = linc
        go[f(gox) == linc] = linc
    go[go == bgc] = 8
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    portrait = H > W
    L, S = (H, W) if portrait else (W, H)   # long / short extent

    # background: a cell that can never be a corner marker; markers: the other colour
    bseed = (1, 0) if H >= 3 else (0, 1)
    bgc = int(I[bseed])
    linc = int([v for v in np.unique(I) if int(v) != bgc][0])
    seeds = sorted({(r, c) for r in (0, H - 1) for c in (0, W - 1) if I[r, c] == linc})

    # canonical coords: t along the long axis measured from the anchor marker's end, u across
    anchor = seeds[0]
    at = anchor[0] if portrait else anchor[1]
    from_far = at != 0

    def rc(t, u):
        tt = L - 1 - t if from_far else t
        return (tt, u) if portrait else (u, tt)

    def tu(r, c):
        t, u = (r, c) if portrait else (c, r)
        return (L - 1 - t if from_far else t), u

    def rect(t0, t1):  # full-width rectangle covering t in [t0, t1)
        r0, c0 = rc(t0, 0)
        r1, c1 = rc(t1 - 1, S - 1)
        return [min(r0, r1), min(c0, c1), abs(r1 - r0), abs(c1 - c0)]

    def ray(seed):  # bouncing diagonal from a corner, split into straight strokes
        t, u = tu(*seed)
        dt = 1 if t == 0 else -1
        du = 1 if u == 0 else -1
        segs, seg = [], [(t, u)]
        while True:
            t += dt
            u += du
            if not (0 <= t < L):
                break
            seg.append((t, u))
            if u == 0 or u == S - 1:
                segs.append(seg)
                seg = [(t, u)]
                du = -du
        if len(seg) > 1 or not segs:
            segs.append(seg)
        return segs

    ops, sels = [], []
    cur = I.copy()

    # 1) background -> 8: one connected region, one FloodFill seed
    ops.append(18)
    sels.append(sel_of([bseed]))
    cur[cur == bgc] = 8

    # 2) the bouncing pattern repeats every P = 2(S-1) steps along the long axis.
    #    Draw one period (block t in [0, B)) of every marker's ray, one stroke at a time.
    P = 2 * (S - 1)
    tile = linc != 0           # Copy/Paste cannot carry colour 0 -> draw full rays instead
    B = min(P, L) if tile else L
    for s in seeds:
        for seg in ray(s):
            cells = [rc(t, u) for t, u in seg if t < B]
            cells = [p for p in cells if cur[p] != linc]
            if cells:
                ops.append(linc)
                sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = linc

    # 3) copy that period block and stamp it down the rest of the long axis
    if tile and B < L:
        copied = None
        start = P
        while start < L:
            n = min(P, L - start)
            src = rect(0, n)              # whole rectangular period block (or its leading part)
            dst = rect(start, start + n)
            sr, sc, sh, sw = src
            dr, dc = dst[0], dst[1]
            block = cur[sr:sr + sh + 1, sc:sc + sw + 1]
            if not np.array_equal(cur[dr:dr + sh + 1, dc:dc + sw + 1], block):
                if copied != n:
                    ops.append(29)
                    sels.append(src)
                    copied = n
                ops.append(30)
                sels.append([dr, dc, 0, 0])
                cur[dr:dr + sh + 1, dc:dc + sw + 1] = block
            start += P

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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
