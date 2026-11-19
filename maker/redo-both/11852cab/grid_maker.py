"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """bgc plus the palette the concentric rings are drawn from — one palette per episode.

    The rule is purely structural (each broken ring of a 5x5 stack is completed by
    mirroring the patch onto itself), so the ring colours only have to stay consistent
    across the episode, not carry fixed roles."""
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, **kwargs) -> dict:
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    hub = max(7, min(30, int(max_h)))
    wub = max(7, min(30, int(max_w)))
    ccols = list(ccols)

    def predicted(g):
        """The rule applied to a grid: every isolated 5x5 ring-stack gets each of its
        rings completed to the 4-fold symmetric whole.  A pair is only handed out when
        its output is exactly what the rule says, so no instance can teach a solver
        (or a trajectory) something the rule does not do."""
        a = [list(row) for row in g]
        hh, ww = len(a), len(a[0])
        out = [list(row) for row in g]
        for rr in range(1, hh - 5):
            for cc in range(1, ww - 5):
                ok = True
                for i in range(-1, 6):
                    for j in range(-1, 6):
                        v = a[rr + i][cc + j]
                        inside = 0 <= i <= 4 and 0 <= j <= 4
                        if not inside and v != bgc:
                            ok = False
                        elif inside and (i + j) % 2 and v != bgc:
                            ok = False
                if not ok or a[rr + 2][cc + 2] == bgc:
                    continue
                pts = [(i, j) for i in range(5) for j in range(5)
                       if a[rr + i][cc + j] != bgc]
                if not any(i in (0, 4) or j in (0, 4) for i, j in pts):
                    continue
                if not any(0 < i < 4 and 0 < j < 4 and (i, j) != (2, 2) for i, j in pts):
                    continue
                if min(i for i, _ in pts) or max(i for i, _ in pts) != 4:
                    continue
                if min(j for _, j in pts) or max(j for _, j in pts) != 4:
                    continue
                for (i, j) in pts:
                    v = a[rr + i][cc + j]
                    for (x, y) in ((j, i), (4 - j, 4 - i), (4 - i, j), (i, 4 - j)):
                        out[rr + x][cc + y] = v
        return tuple(tuple(row) for row in out)

    res = None
    for _attempt in range(50):
        h = unifint(diff_lb, diff_ub, (7, hub))
        w = unifint(diff_lb, diff_ub, (7, wub))
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = shift(asindices(trim(gi)), UNITY)
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        while succ < nobjs and tr < maxtr:
            tr += 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            plcd = shift(bx, loc)
            if plcd.issubset(inds):
                inds = (inds - plcd) - outbox(plcd)
                ringcols = [choice(ccols) for k in range(4)]
                plcdrings = [shift(r, loc) for r in rings]
                gi = fill(gi, ringcols[0], plcdrings[0])
                go = fill(go, ringcols[0], plcdrings[0])
                idx = randint(1, 3)
                gi = fill(gi, ringcols[idx], plcdrings[idx])
                go = fill(go, ringcols[idx], plcdrings[idx])
                remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
                remringcols = ringcols[1:idx] + ringcols[idx + 1:]
                numrs = unifint(diff_lb, diff_ub, (1, 2))
                locs = sample((0, 1), numrs)
                remrings = [rr for j, rr in enumerate(remrings) if j in locs]
                remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
                tofillgi = merge(frozenset(
                    recolor(col, frozenset(sample(totuple(remring),
                                                  4 - unifint(diff_lb, diff_ub, (0, 3)))))
                    for remring, col in zip(remrings, remringcols)
                ))
                tofillgo = merge(frozenset(
                    recolor(col, remring) for remring, col in zip(remrings, remringcols)
                ))
                # the inner ring has to carry paint too, else the stack is not
                # readable as a ring-stack and the completion rule does not apply
                if min(shape(tofillgi)) == 5 and (idx == 1 or 0 in locs):
                    succ += 1
                    gi = paint(gi, tofillgi)
                    go = paint(go, tofillgo)
        if predicted(gi) != go:
            continue          # the pair would not obey the rule — throw it away
        res = {'input': gi, 'output': go}
        if gi != go:
            return res        # and it must actually show a completion happening
    if res is None:
        raise ValueError('no valid instance')
    return res


def derive_operations(I, O):
    """Each isolated 5x5 patch is a stack of concentric, 4-fold symmetric rings, some
    of which are only partly drawn.  Every such patch is mirrored onto itself —
    up<->down, left<->right, then across its diagonal — and after each mirror the
    patch's own marks are stamped back on top, so the reflected copies pile up until
    every ring is whole.  Which patches, which mirror axes, and how many of them are
    all measured from the input; O is never looked at."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # --- the mirrors of a 5x5 patch --------------------------------------------
    MIRRORS = (("h", lambda a: np.flipud(a)),     # up <-> down
               ("v", lambda a: np.fliplr(a)),     # left <-> right
               ("d", lambda a: a.T.copy()))       # across the main diagonal
    # ARCLE has no diagonal-mirror op: on a square patch that mirror is a quarter turn,
    # possibly followed by a straight flip.  Spellings are tried in order and the first
    # one whose every single step visibly changes the patch is the one emitted.
    SPELLING = {"h": ([27],),
                "v": ([26],),
                "d": ([25], [24], [25, 26], [27, 25], [26, 24], [24, 27])}
    OPFN = {24: lambda a: np.rot90(a, 1), 25: lambda a: np.rot90(a, 3),
            26: np.fliplr, 27: np.flipud}

    def completed(patch):
        """The same patch with every ring whole: each painted cell also shows up at its
        diagonal, anti-diagonal, up-down and left-right mirror position."""
        out = patch.copy()
        for i in range(5):
            for j in range(5):
                v = int(patch[i, j])
                if v == bgc:
                    continue
                for (a, b) in ((i, j), (j, i), (4 - j, 4 - i), (4 - i, j), (i, 4 - j)):
                    out[a, b] = v
        return out

    def is_patch(r, c):
        """True when a ring-stack sits at rows r..r+4, cols c..c+4 of the input."""
        if not (1 <= r <= h - 6 and 1 <= c <= w - 6):
            return False
        core = I[r:r + 5, c:c + 5]
        frame = I[r - 1:r + 6, c - 1:c + 6]          # isolated: bare one-cell frame
        if (frame[0] != bgc).any() or (frame[6] != bgc).any():
            return False
        if (frame[1:6, 0] != bgc).any() or (frame[1:6, 6] != bgc).any():
            return False
        for i in range(5):                           # rings only sit on even row+col
            for j in range(5):
                if (i + j) % 2 and core[i, j] != bgc:
                    return False
        if core[2, 2] == bgc:                        # the centre dot
            return False
        inner = [core[i, j] for i in range(1, 4) for j in range(1, 4)
                 if (i, j) != (2, 2)]
        if all(v == bgc for v in inner):             # inner ring carries paint
            return False
        outer = [core[i, j] for i in range(5) for j in range(5)
                 if i in (0, 4) or j in (0, 4)]
        if all(v == bgc for v in outer):             # outer ring carries paint
            return False
        nz = np.argwhere(core != bgc)                # paint spans the whole 5x5
        return (nz[:, 0].min() == 0 and nz[:, 0].max() == 4
                and nz[:, 1].min() == 0 and nz[:, 1].max() == 4)

    ops, sels = [], []
    grid = I.copy()
    for r in range(1, max(1, h - 5)):
        for c in range(1, max(1, w - 5)):
            if not is_patch(r, c):
                continue
            cur = grid[r:r + 5, c:c + 5].copy()
            goal = completed(cur)
            if np.array_equal(cur, goal):
                continue                             # every ring already whole
            rect = [r, c, 4, 4]                      # exactly the 5x5 patch: it is
            for name, fn in MIRRORS:                 # reflected whole, background too
                if np.array_equal(cur, goal):
                    break
                image = fn(cur)
                if np.array_equal(image, cur):
                    continue                         # already symmetric this way
                chain = None
                for cand in SPELLING[name]:
                    state = cur
                    good = True
                    for op in cand:
                        nxt = np.array(OPFN[op](state))
                        if np.array_equal(nxt, state):
                            good = False             # that step would show nothing
                            break
                        state = nxt
                    if good and np.array_equal(state, image):
                        chain = cand
                        break
                if chain is None:
                    continue
                for op in chain:
                    ops.append(op)
                    sels.append(list(rect))
                # stamp the patch's own marks back over the reflection, so the two
                # copies together make the rings longer
                restore = {}
                for i in range(5):
                    for j in range(5):
                        if cur[i, j] != bgc and image[i, j] != cur[i, j]:
                            restore.setdefault(int(cur[i, j]), []).append((r + i, c + j))
                nxt = image.copy()
                for col in sorted(restore):
                    ops.append(col)
                    sels.append(sel_of(restore[col]))
                    for (rr, cc) in restore[col]:
                        nxt[rr - r, cc - c] = col
                cur = nxt
                grid[r:r + 5, c:c + 5] = cur

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
