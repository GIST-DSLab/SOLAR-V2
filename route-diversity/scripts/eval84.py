"""경로가 한 가지뿐이던 84개가 지금 몇 가지인가, 무엇이 갈랐는가."""
import sys, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
TASKS = json.load(open("/hdd_data/yunho/cctmp/solo84.json"))
NEW = ("golf84", "resample1", "barc1")
N = 5

def arms_for(t):
    return [d.name for d in sorted(ROOT.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and (d / t / "grid_maker.py").exists()]

def signature(derive, pairs, eps):
    out = []
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if g is None or g.shape != O.shape or not bool((g == O).all()):
            return None
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            rr, cc = np.where(m)
            steps.append((cm.NAME.get(int(op)), frozenset(zip(rr.tolist(), cc.tolist()))))
        out.append(tuple(steps))
    return tuple(out)

rows = []
for t in TASKS:
    pairs = cm.draw(t, N, [0])
    if not pairs:
        continue
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    sigs = {}
    for a in arms_for(t):
        d = cm.load_derive(ROOT / a / t / "grid_maker.py")
        if d is None:
            continue
        s = signature(d, pairs, eps)
        if s is not None:
            sigs[a] = s
    groups = []
    for a, s in sigs.items():
        for g in groups:
            if sigs[g[0]] == s:
                g.append(a); break
        else:
            groups.append([a])
    idx = {a: i for i, g in enumerate(groups) for a in g}
    old = {v for k, v in idx.items() if k not in NEW}
    rows.append({"task": t, "routes": len(groups),
                 "solved": {a: (a in sigs) for a in NEW},
                 "new": {a: bool(a in idx and idx[a] not in old and
                                 idx[a] not in {v for k, v in idx.items()
                                                if k in NEW and k != a and
                                                list(NEW).index(k) < list(NEW).index(a)})
                         for a in NEW}})

json.dump(rows, open("/hdd_data/yunho/cctmp/eval84.json", "w"))
print(f"대상 {len(rows)}개  (원래 전부 경로 1가지)")
for a in NEW:
    print(f"  {a:10s} 다 푼 과제 {sum(1 for r in rows if r['solved'][a]):3d}"
          f"   새 경로를 낸 과제 {sum(1 for r in rows if r['new'][a]):3d}")
print(f"\n경로 가짓수 분포 {dict(sorted(collections.Counter(r['routes'] for r in rows).items()))}")
print(f"2가지 이상이 된 과제 {sum(1 for r in rows if r['routes']>1)}/{len(rows)}")
print(f"여전히 1가지뿐 {sum(1 for r in rows if r['routes']==1)}")
