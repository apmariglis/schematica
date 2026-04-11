"""
Session-level test configuration.

Sets required environment variables before any module is imported, so that
helpers in schematica.agent (which calls _require_env at module level) can be
imported without a real .env file present.
"""
import os

# Provide all values required by agent.py's module-level _require_env() calls so
# that pure helper functions can be imported without a real .env file present.
os.environ.setdefault("SC_MODEL",              "test/dummy-model")
os.environ.setdefault("SC_MAX_ROWS",           "5")
os.environ.setdefault("SC_MAX_CHARS",          "500")
os.environ.setdefault("SC_BUDGET_BASE",        "10")
os.environ.setdefault("SC_BUDGET_MULTIPLIER",  "3")
os.environ.setdefault("SC_BUDGET_CAP",         "50")
os.environ.setdefault("SC_MIN_ITER_FLOOR",     "3")
os.environ.setdefault("SC_MIN_ITER_DIVISOR",   "2")
os.environ.setdefault("SC_REFINEMENT_BUDGET",  "15")
os.environ.setdefault("SC_MAX_OUTPUT_TOKENS",  "32768")
os.environ.setdefault("SC_CACHE",              "false")
