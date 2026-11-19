"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
    # generator: bgc, noisec = sample(cols, 2); pattern colours are per-instance decor
    bgc, noisec = random.sample(list(range(10)), 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    if succ == 0:
        raise ValueError("no damage placed")
    if h <= min(max_h, max_w) and w <= min(max_h, max_w):
        rotf = choice((identity, rot90, rot180, rot270))
    else:
        rotf = choice((identity, rot180))
    return {'input': rotf(gi), 'output': rotf(go)}


def _period(I, clean, axis):
    n = I.shape[axis]
    for p in range(2, n):
        if axis == 0:
            m = clean[:n - p, :] & clean[p:, :]
            ok = np.all(I[:n - p, :][m] == I[p:, :][m])
        else:
            m = clean[:, :n - p] & clean[:, p:]
            ok = np.all(I[:, :n - p][m] == I[:, p:][m])
        if m.any() and ok:
            return p
    return n


def _noise_colour(I, examples):
    present = set(I.flatten().tolist())
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            s = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            cand = s if cand is None else (cand & s)
        if cand:
            cand = cand & present
            if len(cand) == 1:
                return next(iter(cand))
    # fallback: the colour whose removal leaves the tightest periodic tiling
    cnt = Counter(I.flatten().tolist())
    best = None
    for col in present:
        cl = (I != col)
        key = (_period(I, cl, 0) * _period(I, cl, 1), cnt[col], col)
        if best is None or key < best[0]:
            best = (key, col)
    return best[1]


def _sim(ops, sels, I):
    """Replay Color / CopyI / Paste the way ARCLE does (0 is never carried)."""
    h, w = I.shape
    g = np.zeros((30, 30), dtype=int); g[:h, :w] = I
    inp = g.copy()
    clip = None
    for op, sel in zip(ops, sels):
        if isinstance(sel, dict):
            cells = [tuple(x) for x in sel["cells"]]
        else:
            r, c, dh, dw = sel
            cells = [(rr, cc) for rr in range(r, r + dh + 1) for cc in range(c, c + dw + 1)]
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells); c1 = max(c for _, c in cells)
        if op <= 9:
            for r, c in cells:
                g[r, c] = op
        elif op == 28:
            clip = inp[r0:r1 + 1, c0:c1 + 1].copy()
        elif op == 30 and clip is not None:
            ch, cw = clip.shape
            sub = g[r0:r0 + ch, c0:c0 + cw]
            cp = clip[:sub.shape[0], :sub.shape[1]]
            sub[cp != 0] = cp[cp != 0]
    return g[:h, :w].copy()


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the grid is ONE period tile repeated
    # over the whole canvas; the noise blots are irrelevant.  The unit of repair
    # is the tile slot: CopyI one intact tile window from the input, then Paste
    # it onto every tile slot that still shows the noise colour.  Slots cut by
    # the top/left border get the matching cut of the tile.  Tile cells that
    # are 0 cannot travel by Paste; the noise cells left on them are painted 0.
    # O is never consulted.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    nc = _noise_colour(I, examples)
    noise = (I == nc)
    if not noise.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    clean = ~noise
    vp = _period(I, clean, 0)
    hp = _period(I, clean, 1)

    wins = [(a, b) for a in range(0, h - vp + 1) for b in range(0, w - hp + 1)
            if not noise[a:a + vp, b:b + hp].any()]
    # prefer a window aligned to the grid corner (no cut slots), then top-left
    wins.sort(key=lambda ab: ((ab[0] % vp != 0) + (ab[1] % hp != 0), ab))
    a, b = wins[0]
    pr, pc = a % vp, b % hp
    R0 = pr - vp if pr else 0
    C0 = pc - hp if pc else 0

    # slots grouped by the cut of the tile they need (full tile first)
    groups = {}
    for R in range(R0, h, vp):
        for C in range(C0, w, hp):
            r0, c0 = max(R, 0), max(C, 0)
            r1, c1 = min(R + vp, h), min(C + hp, w)
            if not noise[r0:r1, c0:c1].any():
                continue                            # intact slot: nothing to do
            # source rectangle = the part of the tile this slot starts with
            sr, sc = a + (r0 - R), b + (c0 - C)
            groups.setdefault((sr, sc, a + vp - sr, b + hp - sc), []).append((r0, c0))

    G = I.copy()
    order = sorted(groups, key=lambda k: (-(k[2] * k[3]), k))
    for key in order:
        sr, sc, bh, bw = key
        src = I[sr:sr + bh, sc:sc + bw]
        copied = False
        for (r0, c0) in groups[key]:
            sub = G[r0:r0 + bh, c0:c0 + bw]
            cp = src[:sub.shape[0], :sub.shape[1]]
            m = cp != 0
            if np.array_equal(sub[m], cp[m]):
                continue                           # paste would change nothing
            if not copied:
                # full rectangle: an intact tile (or its cut) copied verbatim
                ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                copied = True
            ops.append(30); sels.append([r0, c0, 0, 0])
            sub[m] = cp[m]

    # noise still visible sits on tile cells of colour 0 -> paint them 0
    left = [(r, c) for r in range(h) for c in range(w) if G[r, c] == nc]
    if left:
        ops.append(0); sels.append(sel_of(left))

    # drop any Copy/Paste whose removal leaves this trajectory's own result
    # unchanged (compared against our own replay, never against O)
    target = _sim(ops, sels, I)
    i = 0
    while i < len(ops):
        t_ops, t_sels = ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:]
        if np.array_equal(_sim(t_ops, t_sels, I), target):
            ops, sels = t_ops, t_sels
        else:
            i += 1

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
