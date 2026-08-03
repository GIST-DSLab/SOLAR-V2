"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # fixc = the occluding blot colour; ccols_pool = the periodic pattern's colour cycle.
    # Pattern colours are kept non-zero so the region Copy/Paste below moves every cell.
    fixc = random.choice(list(range(10)))
    pool = [c for c in range(1, 10) if c != fixc]
    random.shuffle(pool)
    return {"fixc": fixc, "ccols_pool": pool}


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _candidates(I):
    """For every colour k in I, test the hypothesis 'k is the blot colour':
    mask k out, measure the grid's global (vperiod, hperiod), rebuild the tile.
    Returns list of (k, vp, hp, P, n_changed) for hypotheses that are consistent."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    out = []
    for k in sorted(set(I.flatten().tolist())):
        M = I != k
        if not M.any():
            continue
        hp = wi
        for cand in range(1, wi + 1):
            a, b = I[:, :wi - cand], I[:, cand:]
            mm = M[:, :wi - cand] & M[:, cand:]
            if not ((a != b) & mm).any():
                hp = cand
                break
        vp = hi
        for cand in range(1, hi + 1):
            a, b = I[:hi - cand, :], I[cand:, :]
            mm = M[:hi - cand, :] & M[cand:, :]
            if not ((a != b) & mm).any():
                vp = cand
                break
        T = np.zeros((vp, hp), dtype=int)
        ok = True
        for a in range(vp):
            for b in range(hp):
                vals = set(I[a::vp, b::hp][M[a::vp, b::hp]].tolist())
                if len(vals) != 1:
                    ok = False
                    break
                T[a, b] = vals.pop()
            if not ok:
                break
        if not ok:
            continue
        P = T[np.arange(hi) % vp][:, np.arange(wi) % hp]
        out.append((k, vp, hp, P, int((P != I).sum())))
    return out


def _analyze(I):
    cands = _candidates(I)
    if not cands:
        return None
    cands.sort(key=lambda t: t[4])
    return cands[0][:4]


def _components(mask):
    hi, wi = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _find_source(I, k, vp, hp, r0, c0, bh, bw):
    """Nearest blot-free region congruent to the blot's bbox modulo the pattern period."""
    hi, wi = I.shape
    best = None
    for kv in range(-(hi // vp + 1), hi // vp + 2):
        for kh in range(-(wi // hp + 1), wi // hp + 2):
            if kv == 0 and kh == 0:
                continue
            sr, sc = r0 + kv * vp, c0 + kh * hp
            if sr < 0 or sc < 0 or sr + bh > hi or sc + bw > wi:
                continue
            if (I[sr:sr + bh, sc:sc + bw] == k).any():
                continue
            key = (abs(kv) + abs(kh), abs(kv), abs(kh), sr, sc)
            if best is None or key < best:
                best = key
    return None if best is None else (best[3], best[4])


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


# --------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h, max_w, fixc, ccols_pool) -> dict:
    while True:
        h = _unifint(diff_lb, diff_ub, (7, max_h))
        w = _unifint(diff_lb, diff_ub, (7, max_w))
        pmax = min(9, h // 3, w // 3, len(ccols_pool))
        if pmax < 2:
            continue
        p = _unifint(diff_lb, diff_ub, (2, pmax))
        ccols = ccols_pool[:p]
        if random.choice((True, False)):
            go = np.array([[ccols[(c + r) % p] for c in range(w)] for r in range(h)], dtype=int)
        else:
            go = np.array([[ccols[c % p] for c in range(w)] for r in range(h)], dtype=int)
        if random.choice((True, False)) and w <= max_h and h <= max_w:
            go = np.rot90(go, 3)
        H, W = go.shape

        gi = go.copy()
        nsq = _unifint(diff_lb, diff_ub, (1, max(1, (H * W) // 25)))
        maxtr, tr, succ = 4 * nsq, 0, 0
        while succ < nsq and tr < maxtr:
            tr += 1
            oh = _unifint(diff_lb, diff_ub, (2, 5))
            ow = _unifint(diff_lb, diff_ub, (2, 5))
            loci = random.randint(0, H - oh)
            locj = random.randint(0, W - ow)
            tmp = gi.copy()
            tmp[loci:loci + oh, locj:locj + ow] = fixc
            clean_row = any(not (tmp[r] == fixc).any() for r in range(H))
            clean_col = any(not (tmp[:, c] == fixc).any() for c in range(W))
            if clean_row and clean_col:
                gi = tmp
                succ += 1
        if succ == 0:
            continue

        # accept only instances whose pattern (and blot colour) is recoverable from the input alone
        res = _analyze(gi)
        if res is None:
            continue
        k, vp, hp, P = res
        if k != fixc or not np.array_equal(P, go):
            continue
        comps = _components(gi == fixc)
        if not comps:
            continue
        if any(_find_source(gi, fixc, vp, hp, *_bbox(cl)) is None for cl in comps):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}


# ------------------------------------------------------- derive_operations ---
def derive_operations(I, O):
    """The grid is one periodic pattern (period vp x hp) with solid blots painted over it.
    For each blot: copy the pattern's own repeat of that same region from a clean spot
    (offset by whole periods) and paste it over the blot."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    res = _analyze(I)
    if res is not None:
        k, vp, hp, P = res
        comps = sorted(_components(I == k), key=lambda cl: _bbox(cl)[:2])
        for cells in comps:
            r0, c0, bh, bw = _bbox(cells)
            src = _find_source(I, k, vp, hp, r0, c0, bh, bw)
            if src is not None:
                sr, sc = src
                ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])   # CopyI clean repeat
                ops.append(30); sels.append([r0, c0, 0, 0])             # Paste over the blot
            else:
                # no clean repeat of the whole blot: rebuild this blot from the tile, row by row
                by_row = {}
                for (r, c) in cells:
                    by_row.setdefault(r, []).append(c)
                for r in sorted(by_row):
                    cs = sorted(by_row[r])
                    i = 0
                    while i < len(cs):
                        j = i
                        while (j + 1 < len(cs) and cs[j + 1] == cs[j] + 1
                               and P[r, cs[j + 1]] == P[r, cs[i]]):
                            j += 1
                        ops.append(int(P[r, cs[i]]))
                        sels.append([r, cs[i], 0, cs[j] - cs[i]])
                        i = j + 1

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
