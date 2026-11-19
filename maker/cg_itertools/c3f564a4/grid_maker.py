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

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
VARIANTS = [
    {"diagonal": False, "rotate": False},
    {"diagonal": False, "rotate": True},
    {"diagonal": True, "rotate": False},
    {"diagonal": True, "rotate": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)
    ccols = [c for c in cols if c != fixc]
    random.shuffle(ccols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"fixc": fixc, "ccols": ccols, "instance_plan": plan}


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, fixc, ccols,
             diagonal=None, rotate=None) -> dict:
    if diagonal is None:
        diagonal = random.choice((True, False))
    if rotate is None:
        rotate = random.choice((True, False))
    h = _unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = _unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    p = _unifint(diff_lb, diff_ub, (2, max(2, min(9, min(h // 3, w // 3)))))
    pc = list(ccols[:p])
    if diagonal:
        go = [[pc[(c + r) % p] for c in range(w)] for r in range(h)]
    else:
        go = [[pc[c % p] for c in range(w)] for r in range(h)]
    gi = [row[:] for row in go]

    def n_occ(g):
        n = 0
        for row in g:
            for j in range(w - p + 1):
                if row[j:j + p] == pc:
                    n += 1
        return n

    nsq = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    tr = succ = 0
    while succ < nsq and tr < 4 * nsq:
        oh = _unifint(diff_lb, diff_ub, (2, 5))
        ow = _unifint(diff_lb, diff_ub, (2, 5))
        li = random.randint(0, h - oh)
        lj = random.randint(0, w - ow)
        tmp = [row[:] for row in gi]
        for r in range(li, li + oh):
            for c in range(lj, lj + ow):
                tmp[r][c] = fixc
        if (n_occ(tmp) > 1 and any(fixc not in row for row in tmp)
                and any(all(tmp[r][c] != fixc for r in range(h)) for c in range(w))):
            gi = tmp
            succ += 1
        tr += 1
    if rotate:
        gi = [list(r) for r in zip(*gi[::-1])]
        go = [list(r) for r in zip(*go[::-1])]
    return {"input": gi, "output": go}


def _read_wallpaper(I, noise):
    """Find the stripe lattice (direction, period) that the non-noise cells obey."""
    hi, wi = I.shape
    R, C = np.indices((hi, wi))
    keep = I != noise
    best = None
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
            if bad or not mapping:
                continue
            if len(set(mapping.values())) != len(mapping):
                continue
            cand = (len(mapping) == p, -p, (a, b), p, mapping)
            if best is None or cand[:2] > best[:2]:
                best = cand
    return best


def _guess_noise(I):
    # fallback without demonstrations: the colour whose removal leaves the
    # cleanest full stripe lattice, preferring the smallest period
    best = None
    for k in sorted(set(I.flatten().tolist())):
        res = _read_wallpaper(I, k)
        if res is None:
            continue
        sc = (res[0], res[1], int((I != k).sum()))
        if best is None or sc > best[0]:
            best = (sc, k)
    return None if best is None else best[1]


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # occluder colour: the one the demonstrations show present in inputs and
    # gone from outputs
    noise = None
    if examples:
        cnt = {}
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            for k in set(ei.flatten().tolist()) - set(eo.flatten().tolist()):
                cnt[k] = cnt.get(k, 0) + 1
        if cnt:
            noise = max(cnt, key=lambda k: cnt[k])
    if noise is None:
        noise = _guess_noise(I)

    if noise is not None and (I == noise).any():
        res = _read_wallpaper(I, noise)
        if res is not None:
            _, _, (a, b), p, mapping = res
            R, C = np.indices((hi, wi))
            key = (a * R + b * C) % p
            # The wallpaper is one formula over the whole grid (colour = stripe
            # class). Restore it one stripe family at a time: every occluded cell
            # lying on the stripes of colour v gets v, across all patches at once.
            for cls in range(p):
                v = mapping.get(cls)
                if v is None or v == noise:
                    continue
                cells = [(int(r), int(c)) for r, c in zip(*np.where((key == cls) & (I == noise)))]
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
