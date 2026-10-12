"""새 분해를 시도했다가 인스턴스를 못 맞춘 10개에 실패 쌍을 보여준다.

틀린 입출력 쌍을 그대로 보여주는 것이 규칙을 말로 설명하는 것보다 잘 먹혔다는
전례가 있다. 여기서는 그 maker가 실제로 내놓은 격자와 정답 격자를 나란히 준다.
"""
import sys, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
fails = json.load(open("/hdd_data/yunho/cctmp/fails10.json"))
base = json.load(open("/hdd_data/yunho/cctmp/group41_feedback.json"))
base.update(json.load(open("/hdd_data/yunho/cctmp/group105_feedback.json")))

def gs(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in np.asarray(g))

out = {}
for arm, tasks in fails.items():
    for t in tasks:
        d = cm.load_derive(ROOT / arm / t / "grid_maker.py")
        pairs = cm.draw(t, 5, [0]) or []
        eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
        bad = None
        for i, (I, O) in enumerate(pairs):
            shown = eps[i % len(eps)][0] if eps else None
            g, ops, sels = cm.replay(d, I, O, shown)
            if g is None or g.shape != O.shape or not bool((g == O).all()):
                bad = (I, O, g); break
        if bad is None:
            continue
        I, O, g = bad
        src = (ROOT / arm / t / "grid_maker.py").read_text()
        i0 = src.find("def derive_operations"); j0 = src.find("\nclass GridMaker", i0)
        code = src[i0:j0 if j0 > 0 else None].rstrip()
        extra = (
            "\n\nYou were asked this before and wrote the function below. It carves the "
            "task a different way, which is what was wanted, but it does not reproduce "
            "the answer on every instance this task's generator makes.\n\n```python\n"
            + code + "\n```\n\nOn this instance it is wrong.\nINPUT:\n" + gs(I)
            + "\nCORRECT OUTPUT:\n" + gs(O) + "\nWHAT YOUR OPERATIONS PRODUCED:\n"
            + ("(the route did not run)" if g is None else gs(g))
            + "\n\nKeep the new decomposition and make it cover this instance too. "
              "If the decomposition cannot be made to cover it, go back to the one "
              "that works and say why the other reading does not hold.")
        out[t] = {"feedback": base.get(t, {}).get("feedback", "") + extra}

json.dump(out, open("/hdd_data/yunho/cctmp/retry10_feedback.json", "w"), indent=1)
json.dump(sorted(out), open("/hdd_data/yunho/cctmp/retry10.json", "w"))
print("wrote", len(out), sorted(out))
