"""Shared code for the studies: timing and accuracy helpers, result saving, and validated layers.

Importable from any folder once scripts/setup_env.sh has run (it puts the repository root on the
virtual environment's path). Code in lab.layers was promoted from a study; each module names its study.
"""
from lab.bench import bench, max_rel_err, peak_memory_gb, rel_err, reset_peak_memory, seed_everything, sync
from lab.results import env_info, save_results

__all__ = ["bench", "env_info", "max_rel_err", "peak_memory_gb", "rel_err", "reset_peak_memory",
           "save_results", "seed_everything", "sync"]
