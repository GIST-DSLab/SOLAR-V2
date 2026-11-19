"""
ARC Task: c9f8e694 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical (key on the left) layout


def sample_colors(num_examples=None) -> dict:
    # bgc is hardcoded to 0 by the generator. sqc is fixed per episode (the rule itself does not depend on it).
    sqc = random.choice(range(1, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]  # the test orientation has been shown in the examples
    return {"sqc": sqc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, sqc=None, rot=None, **kw) -> dict:
    if sqc is None:
        sqc = random.choice(range(1, 10))
    if rot is None:
        rot = random.choice(ROTS)
    bgc = 0
    H, W = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (5, min(30, H)))
    w = _unifint(diff_lb, diff_ub, (5, min(30, W)))
    remcols = [c for c in range(1, 10) if c != sqc]
    nsq = _unifint(diff_lb, diff_ub, (1, 8))
    gir = np.zeros((h, w - 1), dtype=int)
    gil = [random.choice(remcols) for _ in range(h)]
    succ = fails = 0
    while succ < nsq and fails < nsq * 5:
        loci = random.randint(0, h - 3)
        locj = random.randint(0, w - 3)
        lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
        locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
        if lock < h and locl < w - 1:
            gir[loci:lock + 1, locj:locl + 1] = sqc
            succ += 1
        else:
            fails += 1
    if succ == 0:
        raise ValueError("no blocks placed")
    rows = set(np.nonzero(gir == sqc)[0].tolist())
    key = np.array([gil[i] if i in rows else bgc for i in range(h)])[:, None]
    gi = np.hstack([key, gir])
    giro = np.where(gir == sqc, np.repeat(key, w - 1, axis=1), gir)
    go = np.hstack([key, giro])
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_key(I):
    """Find the border line that acts as the colour key. Every other cell must be
    background or the one block colour, and each key entry must be non-zero
    exactly when its stripe contains block cells."""
    h, w = I.shape
    for kind, idx in [('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)]:
        if kind == 'col':
            line = I[:, idx]
            rest = np.delete(I, idx, axis=1)
        else:
            line = I[idx, :]
            rest = np.delete(I, idx, axis=0)
        cs = set(np.unique(rest).tolist()) - {0}
        if len(cs) != 1:
            continue
        sqc = cs.pop()
        if sqc in set(line.tolist()):
            continue
        stripe_has = (rest == sqc).any(axis=1 if kind == 'col' else 0)
        if all((line[i] != 0) == bool(stripe_has[i]) for i in range(len(line))):
            return kind, idx, line, sqc
    return None


def derive_operations(I, O, examples=None):
    """Each key entry paints the block cells on its own stripe: its row if the key
    is a column, its column if the key is a row.

    The golf reading (every non-zero cell in a row takes that row's first cell)
    divides the task the same way: one stripe per key entry, with the key cell
    already holding its own colour. The two readings agree, so the ops are the
    maker's: one Color op per key entry, on the block cells of that stripe
    (all of which can be read from I). O is never inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    found = _find_key(I)
    if found is not None:
        kind, kidx, key, sqc = found
        for i in range(len(key)):
            col = int(key[i])
            if col == 0:
                continue
            if kind == 'col':
                cells = [(i, c) for c in range(w) if c != kidx and I[i, c] == sqc]
            else:
                cells = [(r, i) for r in range(h) if r != kidx and I[r, i] == sqc]
            if not cells:
                continue
            ops.append(col)               # Color<key entry> on the block cells of this stripe
            sels.append(sel_of(cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
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
                                f"for task c9f8e694"
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
                    f"Failed to build a complete episode for task c9f8e694 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c9f8e694-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
