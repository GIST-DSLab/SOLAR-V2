"""BARC seed를 참조로 붙인다.

seed는 RE-ARC verifier와 같은 과제를 푸는 다른 사람의 풀이인데, 어휘가 다르다
-- object_interior, random_sprite, Color.YELLOW 같은 이름과 '무슨 개념인가'를
적은 주석이 붙어 있다. 노리는 것은 정확도가 아니라 분해다. 같은 규칙을 다른
단위로 쪼개면 ARCLE 경로도 달라진다.

generate_input()은 떼어낸다. 인스턴스는 RE-ARC generator에서 나오고, seed의
생성기를 보여주면 그쪽을 베껴 쓸 이유가 생긴다.
"""
import sys, json, re
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import choose_maker as cm
from pathlib import Path

SEEDS = Path("/hdd_data/yunho/barc/seeds")
ROOT = Path("/hdd_data/yunho/solar-traj/maker/arc-agi-1")
TAIL = """

That is one person's reading of this task, written in their own vocabulary rather than ARCLE's. Where it carves the problem differently from the maker above -- different objects, a different order, a different thing held fixed -- follow the reading in the reference, not the one already here. Where the two agree, there is nothing to change and you should say so.
The point is not a shorter route or a more accurate one. It is a different decomposition of the same rule, so write the operations that reading implies even when they are more numerous.
Two limits hold regardless.
Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection, and the route then teaches nothing.
And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations."""

def seed_text(tid):
    hits = sorted(SEEDS.glob(f"{tid}*.py"))
    if not hits:
        return None
    src = hits[0].read_text()
    i = src.find("def generate_input")
    if i > 0:
        src = src[:i]
    # 머리말의 import는 버리고 개념 주석부터 남긴다
    j = src.find("# concepts:")
    return src[j:].rstrip() if j >= 0 else src.rstrip()

want = json.load(open("/hdd_data/yunho/cctmp/barc_overlap.json"))
solo = set(json.load(open("/hdd_data/yunho/cctmp/solo84.json")))
out, skipped = {}, []
for t in want:
    ref = seed_text(t)
    if not ref:
        skipped.append(t); continue
    src = (ROOT / t / "grid_maker.py").read_text()
    i = src.find("def derive_operations")
    j = src.find("\nclass GridMaker", i)
    code = src[i:j if j > 0 else None].rstrip()
    head = ("Task-specific quality requirement: this task's maker already solves it. "
            "Its derive_operations is:\n\n```python\n" + code + "\n```\n\n"
            "Here is an independent solution to the same task, written by someone "
            "else against a different library:\n\n```python\n" + ref + "\n```")
    out[t] = {"feedback": head + TAIL}

json.dump(out, open("/hdd_data/yunho/cctmp/barc_feedback.json", "w"), indent=1)
pilot = [t for t in out if t in solo]
json.dump(pilot, open("/hdd_data/yunho/cctmp/barc_pilot.json", "w"))
rest = [t for t in out if t not in solo]
json.dump(rest, open("/hdd_data/yunho/cctmp/barc_rest.json", "w"))
print(f"wrote {len(out)} tasks (skipped {len(skipped)}); pilot {len(pilot)}, rest {len(rest)}")
print("pilot:", " ".join(sorted(pilot)))
