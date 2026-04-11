"""
Tests for granularity values accepted by MeasurableMetric.

"tick" is the correct value for un-aggregated, row-level observations.
"event" was the old (semantically wrong) name and must no longer be accepted.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schematica.catalogue import MeasurableMetric


VALID_BASE = {
    "name":        "sensor_readings",
    "description": "Magnitude of sensor events over time",
    "sql":         "SELECT occurred_at, magnitude FROM events ORDER BY occurred_at",
    "time_range":  {"start": "2024-01-01", "end": "2024-12-31"},
    "unit":        "count",
    "tables_used": ["events"],
    "confidence":  "high",
    "agent_notes": "Derived from events.magnitude column",
}


def test_tick_is_accepted_as_granularity():
    metric = MeasurableMetric(**{**VALID_BASE, "granularity": "tick"})

    assert metric.granularity == "tick"


def test_event_is_rejected_as_granularity():
    with pytest.raises(ValidationError):
        MeasurableMetric(**{**VALID_BASE, "granularity": "event"})


def test_all_time_period_granularities_are_still_accepted():
    for granularity in ("daily", "weekly", "monthly", "quarterly", "annual"):
        metric = MeasurableMetric(**{**VALID_BASE, "granularity": granularity})
        assert metric.granularity == granularity
