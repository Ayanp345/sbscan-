"""Benchmarking: run the scanner over the Trojan Zoo and score it against ground truth."""
from .harness import EntryResult, EvalReport, naive_baseline_units, run_zoo_eval

__all__ = ["EntryResult", "EvalReport", "run_zoo_eval", "naive_baseline_units"]
