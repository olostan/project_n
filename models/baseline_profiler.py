"""
Project N: Personal Baseline Profiler (Phase 0).
Calculates Child N's personal bioacoustic baseline statistics across clean historical
episodes to configure the Acute Distress Anomaly Detector.
"""

from typing import Any

import numpy as np


class PersonalBaselineProfiler:
    """
    Computes Child N's personal bioacoustic and kinematic baseline statistics
    from verified clean episodes (no somatic distress, no acute discomfort).
    """

    MIN_EPISODES: int = 10
    MIN_DAYS: int = 5

    @classmethod
    def compute_baseline(
        cls,
        episodes_data: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Computes baseline statistics from a sequence of verified episode dictionaries.

        Each episode dictionary should contain:
        - 'f0_mean_hz': float
        - 'cpp_db': float
        - 'captured_date': str (YYYY-MM-DD)

        Returns:
            Dictionary of baseline statistics.
        """
        if not episodes_data:
            return {
                "calibrated_episodes_count": 0.0,
                "calibrated_days_count": 0.0,
                "f0_mean_hz": 0.0,
                "f0_std_hz": 0.0,
                "f0_upper_limit_hz": 450.0,
                "cpp_mean_db": 0.0,
                "cpp_std_db": 0.0,
                "cpp_lower_limit_db": 4.0,
            }

        f0_values = [
            float(ep["f0_mean_hz"])
            for ep in episodes_data
            if ep.get("f0_mean_hz") is not None and ep["f0_mean_hz"] > 0
        ]
        cpp_values = [float(ep["cpp_db"]) for ep in episodes_data if ep.get("cpp_db") is not None]
        unique_days = {ep["captured_date"] for ep in episodes_data if ep.get("captured_date")}

        episodes_count = float(len(episodes_data))
        days_count = float(len(unique_days))

        f0_mean = float(np.mean(f0_values)) if f0_values else 260.0
        f0_std = float(np.std(f0_values)) if len(f0_values) > 1 else 30.0
        # Upper threshold: Mean + 2.5 standard deviations (capturing acute vocal excursions)
        f0_upper = float(max(f0_mean + (2.5 * f0_std), 400.0))

        cpp_mean = float(np.mean(cpp_values)) if cpp_values else 8.5
        cpp_std = float(np.std(cpp_values)) if len(cpp_values) > 1 else 1.5
        # Lower threshold: Mean - 2.0 standard deviations (capturing severe vocal strain/dysphonia)
        cpp_lower = float(max(cpp_mean - (2.0 * cpp_std), 3.0))

        return {
            "calibrated_episodes_count": episodes_count,
            "calibrated_days_count": days_count,
            "f0_mean_hz": f0_mean,
            "f0_std_hz": f0_std,
            "f0_upper_limit_hz": f0_upper,
            "cpp_mean_db": cpp_mean,
            "cpp_std_db": cpp_std,
            "cpp_lower_limit_db": cpp_lower,
        }
