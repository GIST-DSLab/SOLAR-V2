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
from collections import deque

import numpy as np

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """The generator's three colour roles: the canvas colour, the colour the
    noise patches are painted with, and the palette the repeating block is drawn
    from.  0 is kept out of the pool so every wallpaper cell is opaque to
    Copy/Paste."""
    pool = list(range(1, 10))
    bgc, noisec = random.sample(pool, 2)
    ccols = [c for c in pool if c != noisec]
    random.shuffle(ccols)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    b = max(a, b)
    ir = b - a
    lo = a + int(diff_lb * ir)
    hi = a + int(diff_ub * ir)
    return random.randint(min(lo, b), min(max(hi, lo), b))


# ------------------------------------------------------------- analysis -----
def _comp_count_per_color(A):
    """number of 4-connected same-colour blobs, per colour (the verifier's own
    test for the noise colour: it covers, so it makes the fewest blobs)"""
    h, w = A.shape
    seen = np.zeros((h, w), dtype=bool)
    cnt = {}
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(A[r, c])
            cnt[col] = cnt.get(col, 0) + 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and A[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    return cnt


def _periods(A, mask, axis):
    """every period the undamaged cells agree with, smallest first"""
    n = A.shape[axis]
    out = []
    for p in range(1, n + 1):
        if axis == 0:
            X, Y, mx, my = A[:n - p], A[p:], mask[:n - p], mask[p:]
        else:
            X, Y, mx, my = A[:, :n - p], A[:, p:], mask[:, :n - p], mask[:, p:]
        both = ~mx & ~my
        if not both.any() or np.all(X[both] == Y[both]):
            out.append(p)
    return out


def _analyze(I):
    """noise colour, the wallpaper's two periods and the wallpaper itself,
    read off the input alone"""
    h, w = I.shape
    cnt = _comp_count_per_color(I)
    order = sorted(cnt, key=lambda c: (cnt[c], int((I == c).sum()), c))
    for noisec in order:
        mask = (I == noisec)
        if not mask.any() or mask.all():
            continue
        vs = [p for p in _periods(I, mask, 0) if p <= h // 2]
        hs = [p for p in _periods(I, mask, 1) if p <= w // 2]
        for vper in vs:
            for hper in hs:
                F = np.full((vper, hper), -1, dtype=int)
                good = True
                for r in range(h):
                    for c in range(w):
                        if mask[r, c]:
                            continue
                        x, y = r % vper, c % hper
                        if F[x, y] < 0:
                            F[x, y] = I[r, c]
                        elif F[x, y] != I[r, c]:
                            good = False
                            break
                    if not good:
                        break
                if not good or (F < 0).any():
                    continue
                C = F[np.arange(h) % vper][:, np.arange(w) % hper]
                return noisec, vper, hper, F, C
    return None


def _merge_boxes(boxes):
    """damaged patches may overlap into one blob; repair one rectangle each"""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3]))
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(boxes)


def _damaged_boxes(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                r0 = r1 = r
                c0 = c1 = c
                while q:
                    y, x = q.popleft()
                    r0, r1, c0, c1 = min(r0, y), max(r1, y), min(c0, x), max(c1, x)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                boxes.append((r0, c0, r1, c1))
    return _merge_boxes(boxes)


def _mirror_params(F):
    """the wallpaper folds onto itself about j -> s-j (columns) and i -> t-i
    (rows); collect every s / t the repeating block admits"""
    vper, hper = F.shape
    cols = [s for s in range(hper)
            if all(F[x, y] == F[x, (s - y) % hper] for x in range(vper) for y in range(hper))]
    rows = [t for t in range(vper)
            if all(F[x, y] == F[(t - x) % vper, y] for x in range(vper) for y in range(hper))]
    return cols, rows


def _plan(I):
    """How the wallpaper is put back over each damaged rectangle: fold a clean
    stretch of it over the damage where the pattern's mirror symmetry reaches,
    else carry a clean repeat over, else continue the pattern cell by cell."""
    info = _analyze(I)
    if info is None:
        return None
    noisec, vper, hper, F, C = info
    h, w = I.shape
    mask = (I == noisec)
    mcols, mrows = _mirror_params(F)

    shifts = []
    for a in range(-(h // vper) - 1, h // vper + 2):
        for b in range(-(w // hper) - 1, w // hper + 2):
            if a or b:
                shifts.append((a * vper, b * hper))
    shifts.sort(key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))

    def clean(r0, c0, r1, c1):
        return not mask[r0:r1 + 1, c0:c1 + 1].any()

    plans = []
    for (r0, c0, r1, c1) in _damaged_boxes(mask):
        tgt = C[r0:r1 + 1, c0:c1 + 1]
        best = None
        if not np.array_equal(np.fliplr(tgt), tgt):
            for s in mcols:
                for k in range(-(w // hper) - 1, 2 * (w // hper) + 3):
                    sa = s + k * hper
                    cs0, cs1 = sa - c1, sa - c0
                    if cs0 < 0 or cs1 >= w or not clean(r0, cs0, r1, cs1):
                        continue
                    d = abs(cs0 - c0)
                    if best is None or d < best[0]:
                        best = (d, 26, (r0, cs0, r1, cs1))
        if not np.array_equal(np.flipud(tgt), tgt):
            for t in mrows:
                for k in range(-(h // vper) - 1, 2 * (h // vper) + 3):
                    ta = t + k * vper
                    rs0, rs1 = ta - r1, ta - r0
                    if rs0 < 0 or rs1 >= h or not clean(rs0, c0, rs1, c1):
                        continue
                    d = abs(rs0 - r0)
                    if best is None or d < best[0]:
                        best = (d, 27, (rs0, c0, rs1, c1))
        if best is not None:
            plans.append({"kind": "fold", "flip": best[1],
                          "dst": (r0, c0, r1, c1), "src": best[2]})
            continue
        src = None
        for (dr, dc) in shifts:
            sr0, sc0, sr1, sc1 = r0 + dr, c0 + dc, r1 + dr, c1 + dc
            if sr0 < 0 or sc0 < 0 or sr1 >= h or sc1 >= w:
                continue
            if clean(sr0, sc0, sr1, sc1):
                src = (sr0, sc0, sr1, sc1)
                break
        if src is not None:
            plans.append({"kind": "carry", "dst": (r0, c0, r1, c1), "src": src})
        else:
            plans.append({"kind": "draw", "dst": (r0, c0, r1, c1)})
    return {"noisec": noisec, "vper": vper, "hper": hper, "C": C,
            "mask": mask, "plans": plans}


# -------------------------------------------------------------- generate ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = None, noisec: int = None, ccols=None) -> dict:
    pool = list(range(1, 10))
    if bgc is None or noisec is None:
        bgc, noisec = random.sample(pool, 2)
    if not ccols:
        ccols = [c for c in pool if c != noisec]
        random.shuffle(ccols)
    ccols = [c for c in ccols if c != noisec]
    if len(ccols) < 2:
        ccols = [c for c in pool if c != noisec]

    hlim = max(10, min(30, int(max_h)))
    wlim = max(10, min(30, int(max_w)))

    for _attempt in range(200):
        h = _unifint(diff_lb, diff_ub, (10, hlim))
        w = _unifint(diff_lb, diff_ub, (10, wlim))
        hp = _unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = _unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        numc = _unifint(diff_lb, diff_ub, (2, len(ccols)))
        cc = ccols[:numc]

        # the repeating block is a mirror of itself, so the wallpaper it tiles
        # folds onto itself about every separator line and every block centre
        axis = random.choice(("v", "h"))
        pat = [[None] * wp for _ in range(hp)]
        for i in range(hp):
            for j in range(wp):
                if pat[i][j] is None:
                    v = random.choice(cc)
                    pat[i][j] = v
                    if axis == "v":
                        pat[i][wp - 1 - j] = v
                    else:
                        pat[hp - 1 - i][j] = v

        go = [[bgc] * w for _ in range(h)]
        locs = []
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci = (a + 1) + hp * a
                locj = (b + 1) + wp * b
                locs.append((loci, locj))
                for i in range(hp):
                    for j in range(wp):
                        r, c = loci + i, locj + j
                        if r < h and c < w:
                            go[r][c] = pat[i][j]

        # noise patches, under the generator's own acceptance conditions
        gi = [row[:] for row in go]
        numpatches = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph, pw = random.randint(2, 6), random.randint(2, 6)
            li, lj = random.randint(0, h - ph), random.randint(0, w - pw)
            trial = [row[:] for row in gi]
            for i in range(li, li + ph):
                for j in range(lj, lj + pw):
                    trial[i][j] = noisec
            if not any(loci + hp <= h and locj + wp <= w and
                       all(trial[loci + i][locj + j] == pat[i][j]
                           for i in range(hp) for j in range(wp))
                       for (loci, locj) in locs):
                continue                              # no intact block left
            if sum(1 for row in trial if noisec not in row) < 2:
                continue
            if sum(1 for c in range(w)
                   if all(trial[r][c] != noisec for r in range(h))) < 2:
                continue
            gi = trial
            succ += 1
        if succ < 1:
            continue

        opts = [0, 2] + ([1, 3] if (h <= wlim and w <= hlim) else [])
        k = random.choice(opts)
        gi_a = np.rot90(np.array(gi, dtype=int), k)
        go_a = np.rot90(np.array(go, dtype=int), k)

        # keep only instances whose wallpaper is legible from the input alone
        # and whose damage the mirror actually reaches
        p = _plan(gi_a)
        if p is None or p["noisec"] != noisec or not np.array_equal(p["C"], go_a):
            continue
        if not any(pl["kind"] == "fold" for pl in p["plans"]):
            continue
        return {"input": tuple(map(tuple, gi_a.tolist())),
                "output": tuple(map(tuple, go_a.tolist()))}

    raise ValueError("29ec7d0e: no valid instance sampled")


# ------------------------------------------------------------ derivation ----
def _rect(r0, c0, r1, c1):
    # full rectangle: bbox selection is exactly the cells meant
    return [r0, c0, r1 - r0, c1 - c0]


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    p = _plan(I)
    if p is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    C, mask = p["C"], p["mask"]
    cur = I.copy()

    for pl in p["plans"]:
        r0, c0, r1, c1 = pl["dst"]
        if not (cur[r0:r1 + 1, c0:c1 + 1] != C[r0:r1 + 1, c0:c1 + 1]).any():
            continue                                   # already whole again
        if pl["kind"] in ("fold", "carry"):
            sr0, sc0, sr1, sc1 = pl["src"]
            block = I[sr0:sr1 + 1, sc0:sc1 + 1]
            # the clean stretch of wallpaper this damage is restored from
            ops.append(28); sels.append(_rect(sr0, sc0, sr1, sc1))
            ops.append(30); sels.append([r0, c0, 0, 0])
            cur[r0:r1 + 1, c0:c1 + 1] = block
            if pl["kind"] == "fold":
                # fold it over: the wallpaper is its own mirror image
                ops.append(pl["flip"]); sels.append(_rect(r0, c0, r1, c1))
                cur[r0:r1 + 1, c0:c1 + 1] = (np.fliplr(block) if pl["flip"] == 26
                                             else np.flipud(block))
        rem = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if cur[r, c] != C[r, c]:
                    rem.setdefault(int(C[r, c]), []).append((r, c))
        for col in sorted(rem):
            ops.append(col); sels.append(sel_of(rem[col]))
            for (r, c) in rem[col]:
                cur[r, c] = col

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
