"""참조 6개 전체 집계 + 400과제 경로 총수 재집계.

cg_cgi / cg_oxjam / cg_camjam / cg_aggregate / cg_itertools / cg_lv1dev
참조를 하나씩 늘릴 때 새 경로가 얼마나 늘고 얼마나 겹치는지,
그리고 모든 maker를 합친 과제당 경로 총수가 얼마인지.

과제마다 저장하고 다시 돌리면 이어서 간다. SOLAR_CHUNK만큼만 하고 끝내서,
프로세스가 짧게 살다 죽게 한다 -- 못 찾은 누수가 있어도 거기서 회수된다.
"""
import sys, os, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
sys.path.insert(0, "/hdd_data/yunho/cctmp")
import isolate
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
OUT = Path("/hdd_data/yunho/cctmp/eval_cg6.json")
CG = ["cg_cgi", "cg_oxjam", "cg_camjam", "cg_aggregate", "cg_itertools", "cg_lv1dev"]
N = 5
CHUNK = int(os.environ.get("SOLAR_CHUNK", "25"))

def sig(derive, pairs, eps):
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

def others(t):
    return [d.name for d in sorted(ROOT.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and d.name not in CG and (d / t / "grid_maker.py").exists()]

tasks = sorted(p.name for p in (ROOT / "arc-agi-1").iterdir()
               if p.is_dir() and p.name != "__pycache__")
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {r["task"] for r in rows}
todo = [t for t in tasks if t not in done][:CHUNK]
if not todo:
    print("done"); raise SystemExit

def measure(t):
    """과제 하나를 끝까지. 자식 프로세스 안에서 돈다."""
    pairs = cm.draw(t, N, [0])
    if not pairs:
        return {"task": t, "skip": True}
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    old = set()
    for a in others(t):
        d = cm.load_derive(ROOT / a / t / "grid_maker.py")
        if d is None: continue
        s = sig(d, pairs, eps)
        if s: old.add(s)
    new = {}
    for a in CG:
        p = ROOT / a / t / "grid_maker.py"
        if not p.is_file(): continue
        d = cm.load_derive(p)
        if d is None: continue
        s = sig(d, pairs, eps)
        if s: new[a] = s
    row = {"task": t, "old": len(old), "solved": sorted(new),
           "total": len(old | set(new.values()))}
    for k in range(1, len(CG) + 1):
        row[f"fresh{k}"] = len({new[a] for a in CG[:k] if a in new} - old)
    return row


for t in todo:
    # 과제마다 자식 프로세스. 폭주하는 maker가 있어도 그 과제만 잃는다.
    got, why = isolate.run(measure, t, gib=3.0, seconds=600)
    rows.append(got if got else {"task": t, "failed": why})
    tmp = OUT.with_suffix(".part"); tmp.write_text(json.dumps(rows)); tmp.replace(OUT)
print(f"{len(rows)}/400")
