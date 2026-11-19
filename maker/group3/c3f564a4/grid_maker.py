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
from maker.sel_helpers import sel_of

VARIANTS = [
    {"diag": False, "rot": False},
    {"diag": False, "rot": True},
    {"diag": True, "rot": False},
    {"diag": True, "rot": True},
]


def sample_colors(num_examples=None) -> dict:
    fixc = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"fixc": fixc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    x = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * x))))


def generate(diff_lb, diff_ub, max_h, max_w, fixc, diag=None, rot=None) -> dict:
    if diag is None or rot is None:
        v = random.choice(VARIANTS)
        diag, rot = v["diag"], v["rot"]
    Hm, Wm = (max_w, max_h) if rot else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (7, max(7, Hm)))
    w = _unifint(diff_lb, diff_ub, (7, max(7, Wm)))
    p = _unifint(diff_lb, diff_ub, (2, min(9, min(h // 3, w // 3))))
    ccols = random.sample([c for c in range(10) if c != fixc], p)
    R, C = np.indices((h, w))
    key = (R + C) % p if diag else C % p
    go = np.array(ccols)[key]
    gi = go.copy()
    base = ccols

    def occ(g):
        n = 0
        for r in range(h):
            row = g[r].tolist()
            for c in range(w - p + 1):
                if row[c:c + p] == base:
                    n += 1
        return n

    nsq = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    tr = succ = 0
    while succ < nsq and tr < 4 * nsq:
        oh = _unifint(diff_lb, diff_ub, (2, 5))
        ow = _unifint(diff_lb, diff_ub, (2, 5))
        i = random.randint(0, h - oh)
        j = random.randint(0, w - ow)
        t = gi.copy()
        t[i:i + oh, j:j + ow] = fixc
        if occ(t) > 1 and any(fixc not in t[r] for r in range(h)) \
                and any(fixc not in t[:, c] for c in range(w)):
            gi = t
            succ += 1
        tr += 1
    if rot:
        gi = np.rot90(gi, -1)
        go = np.rot90(go, -1)
    return {"input": gi.tolist(), "output": go.tolist()}


_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def period(line):
        n = len(line)
        for q in range(1, n):
            if all(line[c] == line[c + q] for c in range(n - q)):
                return q
        return n

    # occluding colour: the one colour never seen on any unoccluded row/column
    rows = [I[r].tolist() for r in range(hi)]
    cols = [I[:, c].tolist() for c in range(wi)]
    rp = [period(x) for x in rows]
    cp = [period(x) for x in cols]
    seen = set()
    for r in range(hi):
        if rp[r] == min(rp):
            seen |= set(rows[r])
    for c in range(wi):
        if cp[c] == min(cp):
            seen |= set(cols[c])
    palette = sorted(set(I.flatten().tolist()))
    unseen = [k for k in palette if k not in seen]
    noise_cands = unseen if unseen else palette

    # wallpaper lattice: direction + period whose classes are monochrome off the noise
    R, C = np.indices((hi, wi))
    best = None
    for noise in noise_cands:
        keep = I != noise
        for (a, b) in _DIRS:
            for p in range(2, 10):
                key = (a * R + b * C) % p
                mapping, bad = {}, False
                for cls in range(p):
                    u = np.unique(I[(key == cls) & keep])
                    if len(u) > 1:
                        bad = True
                        break
                    if len(u) == 1:
                        mapping[cls] = int(u[0])
                if bad or not mapping or len(set(mapping.values())) != len(mapping):
                    continue
                score = (noise in unseen, len(mapping) == p, -p)
                if best is None or score > best[0]:
                    best = (score, noise, key, p, mapping)
    if best is not None:
        _, noise, key, p, mapping = best
        # restore the wallpaper one colour at a time across the WHOLE grid:
        # one stroke per wallpaper colour = every occluded cell on that lattice class
        for cls in range(p):
            v = mapping.get(cls)
            if v is None or v == noise:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where((I == noise) & (key == cls)))]
            if cells:
                ops.append(int(v))
                sels.append(sel_of(cells))
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
