"""
ARC Task: 6d58a25d (RE-ARC) — LLM-generated grid_maker
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

# The template shape: a 'V' that spans 4 rows and 7 columns, apex on top, opening downward.
_RAW_SHP = [(0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)]
_MINJ = min(j for _, j in _RAW_SHP)
CANON_SHAPE = frozenset((i, j - _MINJ) for i, j in _RAW_SHP)

ROTS = [0, 1, 2, 3]  # number of np.rot90 applications made by the generator


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    v = lo + int(round(d * (hi - lo)))
    v = max(lo, min(hi, v))
    return random.randint(lo, v) if random.random() < 0.3 else v


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    c1 = random.choice([c for c in cols if c != bgc])
    c2 = random.choice([c for c in cols if c not in (bgc, c1)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, rot=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if c1 is None:
        c1 = random.choice([c for c in cols if c != bgc])
    if c2 is None:
        c2 = random.choice([c for c in cols if c not in (bgc, c1)])
    if rot is None:
        rot = random.choice(ROTS)
    # canonical grid dims (before rotation); a quarter-turn swaps them
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    lim_h = max(5, min(30, lim_h))
    lim_w = max(8, min(30, lim_w))
    h = _unifint(diff_lb, diff_ub, 5, lim_h)
    w = _unifint(diff_lb, diff_ub, 8, lim_w)
    g = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 4)
    locj = random.randint(0, w - 7)
    plcd = {(i + loci, j + locj) for i, j in CANON_SHAPE}
    rem = [(i, j) for i in range(h) for j in range(w) if (i, j) not in plcd]
    nnoise = _unifint(diff_lb, diff_ub, 1, max(1, len(rem) // 2 - 1))
    nois = random.sample(rem, nnoise)
    for (i, j) in nois:
        g[i, j] = c2
    for (i, j) in plcd:
        g[i, j] = c1
    go = g.copy()
    for (i, j) in nois:
        if any(c == j and r < i for r, c in plcd):
            last = max(r for r, c in plcd if c == j)
            for r in range(last + 1, h):
                go[r, j] = c2
    gi = np.rot90(g, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def _infer_bgc(I, examples):
    # Background = the colour that the rule paints over in the demonstrations.
    if examples:
        cnt = Counter()
        for EI, EO in examples:
            EI = np.asarray(EI, dtype=int)
            EO = np.asarray(EO, dtype=int)
            if EI.shape != EO.shape:
                continue
            for v in EI[EI != EO].tolist():
                cnt[v] += 1
        if cnt:
            b = cnt.most_common(1)[0][0]
            if b in set(I.flatten().tolist()):
                return b
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O, examples=None):
    """
    Rule read from I alone: rotate your view until the c1 'V' template sits apex-up.
    Each template column whose cells have a noise (c2) cell somewhere below them
    shoots a c2 ray from just under the template down to the grid edge.
    Like the golf reading, the ray only fills the column's cells that are not already
    c2 (cells already c2 are left alone). Each column's ray is one Color op, and the
    ops run along the template from one side to the other.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = _infer_bgc(I, examples)
    flat_idx = np.arange(hi * wi).reshape(hi, wi)

    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(flat_idx, k)
        for col in sorted(set(J.flatten().tolist())):
            if col == bgc:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) != len(CANON_SHAPE):
                continue
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            if frozenset((r - r0, c - c0) for r, c in cells) == CANON_SHAPE:
                found = (J, Jidx, col, cells)
                break
        if found is not None:
            break

    if found is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    J, Jidx, shape_col, shape_cells = found
    hJ = J.shape[0]
    others = [c for c in sorted(set(J.flatten().tolist())) if c not in (bgc, shape_col)]
    if not others:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    noise_col = others[0]

    by_col = {}
    for (r, c) in shape_cells:
        by_col.setdefault(c, []).append(r)

    for j in sorted(by_col):
        first_r, last_r = min(by_col[j]), max(by_col[j])
        fires = any(J[i, j] == noise_col for i in range(first_r, hJ))
        if not fires:
            continue
        # the column below the template, minus cells already holding the noise colour
        ray = [(i, j) for i in range(last_r + 1, hJ) if J[i, j] != noise_col]
        if not ray:
            continue
        pts = [divmod(int(Jidx[r, c]), wi) for (r, c) in ray]
        ops.append(int(noise_col))
        sels.append(sel_of(pts))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
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
                                f"for task 6d58a25d"
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
                    f"Failed to build a complete episode for task 6d58a25d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d58a25d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
