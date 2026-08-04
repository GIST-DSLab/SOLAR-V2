#!/usr/bin/env python3
"""
Turn critique_makers_llm.py verdicts into a --task_feedback_file for regeneration.

The critic reports WHY a maker's process is unsound; gen_rearc_makers_llm.py already
knows how to put per-task reviewer feedback into the user prompt (as an "acceptance
requirement") and to gate on forbidden ops. This just bridges the two.

Op indices are dropped on the way through: they refer to the previous attempt's op
list, which the regeneration prompt never shows, so they would be noise. The evidence
text survives — it describes what went wrong in terms that stand on their own.

Usage:
    python critique_to_feedback.py --critique /hdd_data/yunho/critique_v6.json \
        --out feedback_from_v6.json

    # then, into a NEW maker dir (never overwrite the audited one):
    python gen_rearc_makers_llm.py --prompt_version v3 \
        --task_feedback_file feedback_from_v6.json \
        --output_subdir arc-from-rearc-v7 \
        --tasks $(python critique_to_feedback.py --critique ... --print_tasks) \
        --overwrite --write_only_valid --attempts 3 --save_log --parallel 4
"""
import argparse
import json
from collections import Counter
from pathlib import Path

# Mirrors FINDING_CODES in critique_makers_llm.py — restated here so the feedback
# explains the code to a generator LLM that never saw the critic's prompt.
CODE_MEANING = {
    "ANSWER_RECONSTRUCTION":     "the rule/parameters used to produce O were taken from O rather than measured from I (drawing after inferring the rule from I is fine; drawing without measuring it is not)",
    "RASTER_ORDER_PAINT":        "cells were edited in top-left raster order rather than by object or region",
    "OBJECT_SCATTER":            "edits to a single object were split across non-adjacent stretches of the trajectory",
    "REDUNDANT_CYCLE":           "a sub-sequence returned the grid to a state it had already reached",
    "BOOKKEEPING_OP":            "an op was emitted only to help the Python side compute something",
    "OVERPAINT_WIDER_THAN_DIFF": "cells that already held their target value were painted again",
    "INPLACE_OP_SWEEP":          "an in-place Flip/Rotate caught pre-existing content it should not have transformed",
    "WRONG_CLIPBOARD_SOURCE":    "CopyO was used where CopyI was correct, or vice versa",
    "HARDCODED_CONSTANT":        "a value true of only some instances was hardcoded instead of measured from I/O",
    "DSL_MIMICRY":               "the op sequence transliterated the verifier DSL instead of deriving from I/O",
    "MOVE_MISUSE":               "Move was used for something that is not a translation",
    "INFO_DESTROY_THEN_REVIVE":  "information destroyed by Crop/Resize/erase was then re-read from the input",
    "LEARNABILITY_GAP":          "a structural variant reachable in test was absent from the examples",
    "WRONG_RULE":                "the trajectory implemented a different rule that happened to match O on this episode instead of the task's actual rule",
    "UNIDIOMATIC_OP":            "an op was used against its idiom (Move for a non-translation, Copy/Paste for a non-replication, Resize as a convenience, flood-fill for arbitrary cells)",
    "INCONSISTENT_WORKSPACE":    "where the task reshapes the grid, the build location was not consistent (should be top-left, or resize-first)",
    # emitted by probe_originals.py, not by the LLM critic
    "FAILS_ORIGINAL_PAIR":       "the trajectory does not reproduce the task's own original ARC pairs, though the vendored verifier does — so the rule it implements holds on the instances generate() samples but not on the task as written",
    "UPSTREAM_PAIR_UNVERIFIED":  "the trajectory does not reproduce some original ARC pairs, and neither does the vendored RE-ARC verifier — treat the original pairs, not the verifier, as the authority here",
    "CONCEPT_NOT_LEGIBLE":       "the answer and rule are right but the ops hide the rule — a valid but opaque route (growing-bbox doubling, or redraw where an object-unit Move of the object's exact shape is the concept). Re-derive so the concept is visible: select the object's true shape and Move it, or stamp the base unit at each period offset",
}

# Findings that name a specific op family can also become a hard validation gate,
# not just prose the LLM may ignore.
CODE_FORBIDS = {
    "ANSWER_RECONSTRUCTION": {"label": "ResetGrid (blind repaint)", "ops": [32]},
}

parser = argparse.ArgumentParser()
parser.add_argument("--critique", type=str, required=True,
                    help="critique_makers_llm.py --out JSON")
parser.add_argument("--out", type=str, default=None,
                    help="Write the feedback JSON here")
parser.add_argument("--min_severity", type=str, default="medium",
                    choices=["low", "medium", "high"],
                    help="Drop findings below this severity (default medium)")
parser.add_argument("--verdicts", nargs="+", default=["FAIL", "REVISE"],
                    help="Verdicts to regenerate (default FAIL REVISE)")
parser.add_argument("--forbid_ops", action="store_true",
                    help="Also emit forbid_ops gates where a code implies one")
parser.add_argument("--print_tasks", action="store_true",
                    help="Print only the selected task IDs, space-separated")
args = parser.parse_args()

RANK = {"low": 0, "medium": 1, "high": 2}


def build_feedback(rec: dict) -> str:
    findings = [f for f in rec.get("findings", [])
                if RANK[f["severity"]] >= RANK[args.min_severity]]
    if not findings:
        return ""

    findings.sort(key=lambda f: -RANK[f["severity"]])
    # probe_originals.py findings mean the previous attempt got the grid WRONG on
    # the authors' own pairs, so the stock "right answer, bad process" opening
    # would tell the generator the opposite of what happened.
    wrong_answer = {"FAILS_ORIGINAL_PAIR", "UPSTREAM_PAIR_UNVERIFIED"}
    if any(f["code"] in wrong_answer for f in findings):
        lines = [
            "Your previous attempt at this task was replayed on the original ARC pairs "
            "that ship with the task — not on instances its own generate() produced — "
            "and it did not reproduce them. Passing on self-generated instances is not "
            "sufficient: the original pairs define the task.",
            "",
            "Findings on the previous attempt:",
        ]
    else:
        lines = [
            "An independent reviewer audited your previous attempt at this task. It produced "
            "the CORRECT answer, but the solution process was rejected as unsound. Producing "
            "the right grid is not sufficient — the trajectory must derive it from I.",
            "",
            "Reviewer findings on the previous attempt:",
        ]
    for i, f in enumerate(findings, 1):
        meaning = CODE_MEANING.get(f["code"], "")
        lines.append(f"{i}. {f['code']} ({f['severity']}) — {meaning}.")
        lines.append(f"   Reviewer: {f['evidence'].strip()}")
    if any(f["code"] in wrong_answer for f in findings):
        lines += [
            "",
            "Work out for yourself what the original pairs require that your previous "
            "attempt did not handle, and where the gap is: the rule your operations "
            "implement, the instances your generate() samples, or both. Your new "
            "attempt must reproduce every original pair above AND keep generating "
            "self-consistent instances.",
        ]
    else:
        lines += [
            "",
            "Do not repeat these. Derive the rule from I and O first, then emit the ops that "
            "carry out that rule on whole objects or regions.",
        ]
    return "\n".join(lines)


def main():
    recs = json.loads(Path(args.critique).read_text(encoding="utf-8"))
    wanted = set(args.verdicts)

    feedback = {}
    dropped_no_finding = []
    for r in recs:
        if "error" in r or r.get("verdict") not in wanted:
            continue
        text = build_feedback(r)
        if not text:
            # Flagged, but every finding was below --min_severity: nothing concrete
            # to tell the generator, so regenerating would be a blind retry.
            dropped_no_finding.append(r["task_id"])
            continue
        entry = {"feedback": text}
        if args.forbid_ops:
            gates = [CODE_FORBIDS[f["code"]] for f in r["findings"]
                     if f["code"] in CODE_FORBIDS]
            if gates:
                entry["forbid_ops"] = list({g["label"]: g for g in gates}.values())
        feedback[r["task_id"]] = entry

    if args.print_tasks:
        print(" ".join(sorted(feedback)))
        return

    codes = Counter(f["code"] for r in recs if "error" not in r
                    and r.get("verdict") in wanted
                    for f in r.get("findings", [])
                    if RANK[f["severity"]] >= RANK[args.min_severity])

    print(f"critique     : {args.critique}")
    print(f"verdicts     : {' '.join(sorted(wanted))}")
    print(f"min severity : {args.min_severity}")
    print(f"selected     : {len(feedback)} task(s)")
    if dropped_no_finding:
        print(f"skipped      : {len(dropped_no_finding)} flagged but no finding at "
              f">= {args.min_severity} ({' '.join(dropped_no_finding[:8])}"
              f"{' ...' if len(dropped_no_finding) > 8 else ''})")
    print("\nfindings carried into feedback:")
    for code, n in codes.most_common():
        print(f"  {n:>4}  {code}")

    if args.out:
        Path(args.out).write_text(json.dumps(feedback, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    else:
        print("\n(no --out given; nothing written)")


if __name__ == "__main__":
    main()
