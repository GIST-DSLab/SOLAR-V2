"""
ARC Task: 73251a56 (RE-ARC) — LLM-generated grid_maker
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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
from maker.sel_helpers import sel_of


def _mirror_pos(mode, n, i, j):
    """Position of (i, j) reflected across the grid's symmetry diagonal."""
    return (j, i) if mode == "main" else (n - 1 - j, n - 1 - i)


def _solve_from_input(G):
    """Recover the clean, diagonal-symmetric colour fan from a noisy grid.

    The task's grids are square colour fans that are exact mirror images of
    themselves across ONE diagonal (main diagonal, or the anti-diagonal after a
    90/270 rotation), with that diagonal painted a single colour, then
    vandalised by axis-aligned rectangles of ONE noise colour which never land
    symmetrically on both sides of the diagonal (generator constraint cf2).

    For a candidate (noise colour c, diagonal `mode`): copy every non-c cell
    onto its mirror position, repaint the diagonal with the colour sitting at
    the diagonal's corner (noise never touches the border), and demand the
    result be perfectly mirror-symmetric.  Any colour that fails to appear in
    some asymmetric pair fails that test.  Among surviving candidates the true
    noise colour is the rarest one (verifier's argmin colorcount).

    Returns (count, colour, mode, repaired_grid) or None.  Reads only G.
    """
    G = [list(r) for r in G]
    n = len(G)
    if n == 0 or len(G[0]) != n:
        return None
    best = None
    pal = sorted({v for row in G for v in row})
    for mode in ("main", "anti"):
        if mode == "main":
            diag = [(k, k) for k in range(n)]
            dcol = G[0][0]
        else:
            diag = [(k, n - 1 - k) for k in range(n)]
            dcol = G[0][n - 1]
        for c in pal:
            R = [row[:] for row in G]
            for i in range(n):
                Gi = G[i]
                for j in range(n):
                    v = Gi[j]
                    if v != c:
                        pi, pj = _mirror_pos(mode, n, i, j)
                        R[pi][pj] = v
            for (i, j) in diag:
                R[i][j] = dcol
            ok = True
            for i in range(n):
                for j in range(n):
                    pi, pj = _mirror_pos(mode, n, i, j)
                    if R[i][j] != R[pi][pj]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cnt = sum(row.count(c) for row in G)
                if best is None or cnt < best[0]:
                    best = (cnt, c, mode, R)
    return best


def _components(cells):
    """4-connected components of a set of (r, c) cells."""
    todo = set(cells)
    out = []
    while todo:
        seed = min(todo)
        todo.discard(seed)
        comp = [seed]
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in todo:
                    todo.discard((nr, nc))
                    comp.append((nr, nc))
                    stack.append((nr, nc))
        out.append(sorted(comp))
    return out


# Structural variant: the random final rotation puts the fan's symmetry axis on
# either the main diagonal (identity / rot180) or the anti-diagonal (rot90 /
# rot270).  Both must be shown in the examples for the episode to be learnable.
VARIANTS = [{"axis": "main"}, {"axis": "anti"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)                      # the vandalising colour
    palette_order = [c for c in cols if c != noisec]  # band colours (noise excluded)
    random.shuffle(palette_order)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "palette_order": palette_order, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, palette_order, axis=None) -> dict:
    remcols = [c for c in palette_order if c != noisec]
    maxd = max(6, min(30, int(max_h), int(max_w)))
    lod = min(10, maxd)
    attempt = 0
    while True:
        attempt += 1
        d = unifint(diff_lb, diff_ub, (lod, maxd))
        h, w = d, d
        nsl = unifint(diff_lb, diff_ub, (2, max(2, min(9, h // 2))))
        nsl = max(2, min(nsl, len(remcols), h - 1))
        slopes = [0] + sorted(sample(interval(1, h - 1, 1), nsl - 1))
        ccols = remcols[:nsl]
        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))
        go = tuple(e for e in gi)
        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset(
            {ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0
        ndist = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
        if attempt > 40:
            ndist = min(ndist, 1 + attempt % 3)
        tr, succ = 0, 0
        maxtr = 10 * ndist
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = randint(1, min(5, h - 2))
            ow = randint(1, min(5, w - 2))
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi == go:
            continue
        if axis == "main":
            rotf = choice((identity, rot180))
        elif axis == "anti":
            rotf = choice((rot90, rot270))
        else:
            rotf = choice((identity, rot90, rot180, rot270))
        gi = rotf(gi)
        go = rotf(go)
        # keep only instances whose noise colour is recoverable from the input
        # alone (rarest symmetry-consistent candidate), so the episode is solvable
        best = _solve_from_input(gi)
        if best is None or best[1] != noisec:
            continue
        if tuple(tuple(r) for r in best[3]) != tuple(tuple(int(v) for v in r) for r in go):
            continue
        return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    G = [[int(v) for v in row] for row in I]

    ops, sels = [], []
    # everything below is measured from I: which colour is the vandal, which
    # diagonal is the mirror axis, and what colour each vandalised cell must
    # return to (its mirror partner, or the diagonal's own colour).
    best = _solve_from_input(G)
    if best is not None:
        _cnt, noise, _mode, R = best
        dirty = [(r, c) for r in range(hi) for c in range(wi)
                 if G[r][c] == noise and R[r][c] != noise]
        # repair one noise blob at a time; inside a blob, one colour band at a time
        for comp in sorted(_components(dirty), key=lambda cc: cc[0]):
            bands = {}
            for (r, c) in comp:
                bands.setdefault(R[r][c], []).append((r, c))
            for col in sorted(bands, key=lambda k: bands[k][0]):
                ops.append(int(col))
                sels.append(sel_of(bands[col]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 73251a56"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 73251a56"
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
                                f"for task 73251a56"
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
                    f"Failed to build a complete episode for task 73251a56 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"73251a56-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
