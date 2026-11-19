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
    bgc, noisec = random.sample(range(10), 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    max_h = max(10, min(30, max_h)); max_w = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, remove(bgc, cols))
    numc = unifint(diff_lb, diff_ub, (2, min(9, len(remcols))))
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
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    if succ == 0:
        raise ValueError("no damage patch placed")
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _min_period(G, known, axis):
    # smallest p such that every pair of known cells p apart along `axis` agree
    n = G.shape[axis]
    for p in range(1, n):
        if axis == 0:
            a, b = G[:-p, :], G[p:, :]
            ka, kb = known[:-p, :], known[p:, :]
        else:
            a, b = G[:, :-p], G[:, p:]
            ka, kb = known[:, :-p], known[:, p:]
        both = ka & kb
        if not np.any(both & (a != b)):
            return p
    return n


def _periods_for(I, nc):
    known = I != nc
    return _min_period(I, known, 0), _min_period(I, known, 1)


def _noise_color(I, examples):
    present = set(np.unique(I).tolist())
    if examples:
        # the damage colour is the one the demonstrations always erase
        votes = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            for c in set(np.unique(ei).tolist()) - set(np.unique(eo).tolist()):
                votes[c] += 1
        for c, _ in votes.most_common():
            if c in present:
                return c
        if votes:
            return None          # this episode's damage colour is absent: nothing damaged
    h, w = I.shape
    best = None
    for c in sorted(present):
        vp, hp = _periods_for(I, c)
        if vp > h // 2 or hp > w // 2:
            continue
        key = (vp * hp, int((I == c).sum()))
        if best is None or key < best[0]:
            best = (key, c)
    return None if best is None else best[1]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    nc = _noise_color(I, examples)
    if nc is None or not np.any(I == nc):
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    noise = I == nc
    vp, hp = _periods_for(I, nc)

    # The grid is ONE wallpaper tile of size vp x hp repeated everywhere; measure it
    # from the undamaged cells (every phase class votes).
    tile = {}
    votes = {}
    for r in range(h):
        for c in range(w):
            if not noise[r, c]:
                votes.setdefault((r % vp, c % hp), Counter())[int(I[r, c])] += 1
    for k, v in votes.items():
        tile[k] = v.most_common(1)[0][0]

    G = I.copy()
    clip = None   # content currently held by the clipboard

    # Walk the wallpaper lattice tile by tile; every lattice tile that contains damage
    # gets an undamaged copy of the same tile (whole periods away) pasted over it.
    for R0 in range(0, h, vp):
        for C0 in range(0, w, hp):
            th, tw = min(vp, h - R0), min(hp, w - C0)
            win = noise[R0:R0 + th, C0:C0 + tw]
            if not win.any():
                continue
            # source: a clean lattice tile of the same (possibly edge-clipped) shape
            best = None
            for sr in range(0, h - th + 1, vp):
                for sc in range(0, w - tw + 1, hp):
                    if noise[sr:sr + th, sc:sc + tw].any():
                        continue
                    if (sr - R0) % vp or (sc - C0) % hp:
                        continue
                    d = abs(sr - R0) // vp + abs(sc - C0) // hp
                    if best is None or d < best[0]:
                        best = (d, sr, sc)
            if best is None:
                continue
            _, sr, sc = best
            src = I[sr:sr + th, sc:sc + tw]
            # would the paste visibly change anything? (Paste is transparent for 0)
            if not np.any(win & (src != 0) & (G[R0:R0 + th, C0:C0 + tw] != src)):
                continue

            # re-copy only when pasting what the clipboard already holds would not
            # give the same visible tile (Paste is transparent for 0, clipped at the edge)
            def _pasted(cl):
                out = G[R0:R0 + th, C0:C0 + tw].copy()
                part = cl[:th, :tw]
                out[:part.shape[0], :part.shape[1]][part != 0] = part[part != 0]
                return out
            if clip is None or not np.array_equal(_pasted(clip), _pasted(src)):
                ops.append(28); sels.append([sr, sc, th - 1, tw - 1])  # whole clean tile rectangle
                clip = src.copy()
            ops.append(30); sels.append([R0, C0, 0, 0])
            G[R0:R0 + th, C0:C0 + tw] = _pasted(clip)

    # Whatever damage the pastes could not carry (0-valued tile cells, or tiles with no
    # clean twin): paint those still-damaged cells with their wallpaper colour.
    rest = {}
    for r in range(h):
        for c in range(w):
            if G[r, c] == nc and (r % vp, c % hp) in tile:
                rest.setdefault(tile[(r % vp, c % hp)], []).append((r, c))
    for col in sorted(rest):
        ops.append(col); sels.append(sel_of(rest[col]))

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
