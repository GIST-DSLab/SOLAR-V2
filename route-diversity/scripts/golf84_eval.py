"""경로가 한 가지뿐이던 84개에 golf84를 더하면 몇 가지가 되는가."""
import sys, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
TASKS = json.load(open("/hdd_data/yunho/cctmp/solo84.json"))
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
    has = "golf84" in sigs
    new = has and len({idx["golf84"]} - {v for k, v in idx.items() if k != "golf84"}) > 0
    rows.append({"task": t, "routes": len(groups), "golf84": has, "new": bool(new)})

json.dump(rows, open("/hdd_data/yunho/cctmp/golf84_eval.json", "w"))
made = sum(1 for r in rows if r["golf84"])
new = sum(1 for r in rows if r["new"])
print(f"대상 {len(rows)}개")
print(f"golf84 maker가 인스턴스를 다 푸는 과제 {made}")
print(f"그중 다른 어디서도 안 나온 경로를 낸 과제 {new}")
print(f"경로 가짓수 분포: {dict(sorted(collections.Counter(r['routes'] for r in rows).items()))}")
print(f"여전히 1가지뿐인 과제 {sum(1 for r in rows if r['routes']==1)}")
