"""
Tests for _update_fk_waived — FK rejection cap logic.

When the agent submits finish_catalogue without a JOIN metric for an FK pair
and keeps doing so after being told, it gets stuck in a rejection loop.
The cap waives enforcement for any pair rejected N consecutive times, letting
the run complete rather than looping forever.

_update_fk_waived is a pure helper: it increments per-pair rejection counts
and moves any pair that hits the cap into the waived set.
"""
from __future__ import annotations

from schematica.agent import _FK_REJECTION_CAP, _update_fk_waived


# ── cap constant ──────────────────────────────────────────────────────────────

def test_fk_rejection_cap_is_positive_int():
    assert isinstance(_FK_REJECTION_CAP, int)
    assert _FK_REJECTION_CAP >= 1


# ── count increments ──────────────────────────────────────────────────────────

def test_first_rejection_increments_count_to_one():
    counts, _ = _update_fk_waived([("orders", "customers")], {}, set())

    assert counts[frozenset({"orders", "customers"})] == 1


def test_second_rejection_increments_count_to_two():
    prior = {frozenset({"orders", "customers"}): 1}

    counts, _ = _update_fk_waived([("orders", "customers")], prior, set())

    assert counts[frozenset({"orders", "customers"})] == 2


def test_multiple_pairs_tracked_independently():
    counts, _ = _update_fk_waived(
        [("a", "b"), ("c", "d")], {frozenset({"a", "b"}): 1}, set(), cap=5
    )

    assert counts[frozenset({"a", "b"})] == 2
    assert counts[frozenset({"c", "d"})] == 1


# ── waiving behaviour ─────────────────────────────────────────────────────────

def test_pair_not_waived_before_cap_is_reached():
    counts = {frozenset({"a", "b"}): _FK_REJECTION_CAP - 1}

    # One more rejection — reaches the cap exactly
    _, waived = _update_fk_waived([("a", "b")], counts, set())

    assert frozenset({"a", "b"}) in waived


def test_pair_not_waived_one_below_cap():
    # cap=3: after 2 rejections the pair must still be enforced
    counts = {frozenset({"a", "b"}): 1}

    _, waived = _update_fk_waived([("a", "b")], counts, set(), cap=3)

    assert frozenset({"a", "b"}) not in waived


def test_already_waived_pairs_remain_waived():
    existing = {frozenset({"a", "b"})}

    _, waived = _update_fk_waived([("a", "b")], {}, existing, cap=99)

    assert frozenset({"a", "b"}) in waived


def test_only_capped_pair_is_waived_not_others():
    # a↔b hits the cap; c↔d does not
    counts = {frozenset({"a", "b"}): _FK_REJECTION_CAP - 1}

    _, waived = _update_fk_waived([("a", "b"), ("c", "d")], counts, set())

    assert frozenset({"a", "b"}) in waived
    assert frozenset({"c", "d"}) not in waived


# ── direction independence ─────────────────────────────────────────────────────

def test_pair_direction_does_not_affect_count_key():
    # ("a", "b") and ("b", "a") must hit the same counter
    counts, _ = _update_fk_waived([("a", "b")], {}, set(), cap=99)
    counts, _ = _update_fk_waived([("b", "a")], counts, set(), cap=99)

    assert counts[frozenset({"a", "b"})] == 2
