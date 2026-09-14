"""
Project N: Acute Distress Anomaly Detector.
Automated acoustic and kinematic anomaly screener executing on raw sensory frames
relative to calibrated personal baseline. Triage First: prompts physical comfort check.
"""

from typing import Any


class AcuteDistressAnomalyDetector:
    """
    Automated acoustic and kinematic anomaly screener executing on raw sensory frames.
    Screens for acute acoustic excursions and flinching/guarding kinematics relative to
    the child's calibrated personal baseline. Requires a minimum personal baseline of >=10
    verified clean episodes across >=5 days before activating to prevent alarm fatigue.
    Configured signal-deviation trigger only; does not diagnose or exclude somatic pain, illness,
    or distress. Prompts caregiver to conduct a physical comfort check.
    """

    MIN_CALIBRATED_EPISODES: int = 10
    MIN_CALIBRATED_DAYS: int = 5

    @classmethod
    def evaluate(
        cls,
        measured_features: dict[str, Any],
        baseline_stats: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates measured acoustic and kinematic features against calibrated baseline.

        Args:
            measured_features: Dictionary containing measured features (f0_mean_hz, cpp_db, etc.)
            baseline_stats: Optional dictionary containing child personal baseline limits.

        Returns:
            Dictionary detailing screener availability, anomaly status, and prompts.
        """
        # Fail-closed / Cold-Start gate: require calibrated baseline statistics
        if not baseline_stats:
            return {
                "screener_available": False,
                "distress_anomaly": False,
                "screener_status": "uncalibrated_collecting_baseline",
                "reason": (
                    f"Personal baseline collecting (requires >={cls.MIN_CALIBRATED_EPISODES} clean "
                    f"episodes across >={cls.MIN_CALIBRATED_DAYS} days). Anomaly screening paused to prevent alarm fatigue."
                ),
                "recommendation": (
                    "Personal bioacoustic baseline uncalibrated. Automated anomaly screening paused. "
                    "Use timeline diary and check physical comfort manually if Child N appears unsettled."
                ),
            }

        episodes_count = int(baseline_stats.get("calibrated_episodes_count", 0))
        days_count = int(baseline_stats.get("calibrated_days_count", 0))

        if episodes_count < cls.MIN_CALIBRATED_EPISODES or days_count < cls.MIN_CALIBRATED_DAYS:
            return {
                "screener_available": False,
                "distress_anomaly": False,
                "screener_status": "uncalibrated_collecting_baseline",
                "reason": (
                    f"Personal baseline collecting ({episodes_count}/{cls.MIN_CALIBRATED_EPISODES} episodes, "
                    f"{days_count}/{cls.MIN_CALIBRATED_DAYS} days). Anomaly screening paused."
                ),
                "recommendation": (
                    "Personal bioacoustic baseline collecting. Automated anomaly screening paused. "
                    "Check physical comfort manually if Child N appears unsettled."
                ),
            }

        f0_mean = measured_features.get("f0_mean_hz")
        cpp_val = measured_features.get("cpp_db")
        flinch_guarding = bool(measured_features.get("acute_guarding_detected", False))

        f0_thresh = float(baseline_stats.get("f0_upper_limit_hz", 450.0))
        cpp_thresh = float(baseline_stats.get("cpp_lower_limit_db", 4.0))

        f0_spike = bool(isinstance(f0_mean, int | float) and f0_mean > f0_thresh)
        cpp_strain = bool(isinstance(cpp_val, int | float) and cpp_val < cpp_thresh)

        is_anomaly = bool(f0_spike or cpp_strain or flinch_guarding)
        return {
            "screener_available": True,
            "screener_status": "calibrated_active",
            "distress_anomaly": is_anomaly,
            "indicators": {
                "f0_spike": f0_spike,
                "cpp_strain": cpp_strain,
                "flinch_guarding": flinch_guarding,
            },
            "recommendation": (
                "PHYSICAL COMFORT CHECK SUGGESTED: Bioacoustic or kinematic signals deviate from calibrated baseline. "
                "Prompt caregiver to check physical comfort, hydration, temperature, or sensory environment. "
                "Behavioral and sensory interpretations are suppressed."
                if is_anomaly
                else "No configured signal deviation was detected. This does not assess or exclude pain, illness, or distress."
            ),
        }
