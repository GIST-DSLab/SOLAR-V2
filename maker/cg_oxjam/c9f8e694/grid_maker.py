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
from collections import Counter
from maker.sel_helpers import sel_of

# The code-golf reading and the maker's reading are the same, so this route keeps
# the maker's structure. In the golf version, every nonzero cell in a line
# perpendicular to the key takes that line's key entry. Both treat one stripe
# (the block cells that share one key entry) as the unit, and both walk along
# the key line. Nothing needs to change.


def sample_colors(num_examples=None) -> dict:
    # The background is fixed at 0. The block colour (sqc) and the key colours
    # are chosen at random for each instance. The rule only looks at where the
    # blocks are and what the key line says, never at specific colours, so no
    # colour needs to be fixed for the episode.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unif(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))

    while True:
        h = unif(5, max(5, max_h))
        w = unif(5, max(5, max_w))
        bgc = 0
        sqc = random.choice(range(1, 10))
        remcols = [c for c in range(1, 10) if c != sqc]
        nsq = unif(1, 8)
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
            continue
        rows = set(np.nonzero(gir == sqc)[0].tolist())
        key = np.array([gil[i] if i in rows else bgc for i in range(h)])
        giro = gir.copy()
        for i in range(h):
            giro[i][gir[i] == sqc] = key[i]
        gi = np.hstack([key[:, None], gir])
        go = np.hstack([key[:, None], giro])
        k = random.choice([0, 1, 2, 3])
        gi = np.rot90(gi, -k)
        go = np.rot90(go, -k)
        return {"input": gi.tolist(), "output": go.tolist()}


def _find_key(I):
    """Find the key line: the edge line with no block colour on it, where each
    entry is nonzero exactly when the perpendicular stripe through that entry
    contains block cells."""
    h, w = I.shape
    cnt = Counter(int(v) for v in I.flatten() if v != 0)
    if not cnt:
        return None
    # The key never uses the block colour. Every nonzero key entry has at least
    # 2 block cells in its stripe, so the block colour is the most common
    # nonzero colour.
    sqc = cnt.most_common(1)[0][0]
    for kind, idx in [('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)]:
        line = I[:, idx] if kind == 'col' else I[idx, :]
        if np.any(line == sqc):
            continue
        ok = True
        for i, v in enumerate(line):
            stripe = np.delete(I[i, :], idx) if kind == 'col' else np.delete(I[:, i], idx)
            if (v != 0) != bool(np.any(stripe == sqc)):
                ok = False
                break
        if ok:
            return kind, idx, line.copy(), sqc
    return None


def derive_operations(I, O, examples=None):
    """Each block cell takes the colour of the key-line entry on its own row
    (if the key is a column) or its own column (if the key is a row).
    Everything is measured from I. O is never inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    found = _find_key(I)
    if found is not None:
        kind, idx, key, sqc = found
        # Walk along the key line. For each key entry, recolour that entry's
        # stripe of block cells with one Color op.
        for i in range(len(key)):
            col = int(key[i])
            if col == 0:
                continue
            if kind == 'col':
                cells = [(i, c) for c in range(w) if c != idx and I[i, c] == sqc]
            else:
                cells = [(r, i) for r in range(h) if r != idx and I[r, i] == sqc]
            if not cells:
                continue
            ops.append(col)          # Color<key entry> on this stripe's block cells
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
