"""What `pick` is allowed to promote on.

The six routes taken out in 628fda7 all came through this function. Each was a
candidate that tied its incumbent on every measured axis and carried the
verifier's operation family where the incumbent did not, so `route` -- the last
filter -- decided alone, and a transpose spelled with two flips beat a crop.
The cases below are that promotion, written down so it cannot come back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from choose_maker import pick  # noqa: E402


def s(**kw):
    d = {"loaded": True, "solve": 1.0, "route": 0.0, "copy": 0.0,
         "idle": 0.0, "dep": 0.0, "spare": 0.0}
    d.update(kw)
    return d


def test_route_alone_does_not_promote():
    """662c240a: crop incumbent, two-flip transpose candidate, all else equal."""
    scores = {"arc-agi-1": s(route=0.0), "redo-reflect": s(route=1.0)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo-reflect"])
    assert win == "arc-agi-1"


def test_a_candidate_that_solves_more_still_wins():
    scores = {"arc-agi-1": s(solve=0.6), "redo": s(solve=1.0, route=0.0)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "redo"


def test_a_candidate_that_stops_reading_the_answer_still_wins():
    scores = {"arc-agi-1": s(dep=1.0), "redo": s(dep=0.0)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "redo"


def test_a_spare_operation_disqualifies_a_candidate():
    """An op the route reaches O without is not paid for by solving more."""
    scores = {"arc-agi-1": s(solve=0.9), "redo": s(solve=1.0, spare=0.5)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "arc-agi-1"


def test_an_unmeasured_spare_is_not_a_disqualification():
    """Routes too long to ablate are reported, not rejected."""
    scores = {"arc-agi-1": s(solve=0.9), "redo": s(solve=1.0, spare=None)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "redo"


def test_a_spare_incumbent_is_held_rather_than_swapped():
    """995c5fa3: the incumbent's leading resize is spare, the candidate's route
    is clean and six operations longer. Being clean of the one fault the
    incumbent has does not make a candidate better; the fault is a repair."""
    scores = {"arc-agi-1": s(spare=0.67), "redo-reflect": s(route=1.0)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo-reflect"])
    assert win is None


def test_a_spare_incumbent_still_loses_on_another_axis():
    scores = {"arc-agi-1": s(spare=0.67, solve=0.5), "redo": s(solve=1.0)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "redo"


def test_a_minority_spare_rate_is_the_draw_not_the_route():
    """A colour laid on cells one instance already had is not a fault."""
    scores = {"arc-agi-1": s(solve=0.9), "redo": s(solve=1.0, spare=0.33)}
    win, _ = pick(scores, "arc-agi-1", ["arc-agi-1", "redo"])
    assert win == "redo"
