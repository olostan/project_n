"""
Project N: Four-Layer Output and Parent View Synthesis (Invariant 6 & 8).
Constructs canonical L1-L4 layers, grounded dyadic suggestions, and dual-perspective
Parent View text strictly observing child assent, agency, and epistemic humility.
"""

from typing import Any

from models.contracts import CONTROLLED_ACTIONS


def derive_dyadic_suggestions(
    retrieved_candidates: list[dict[str, Any]],
    distress_triggered: bool,
    is_acceptable: bool,
) -> dict[str, Any]:
    """Derives grounded co-regulatory suggestions from historical confirmed resolutions."""
    if not is_acceptable or distress_triggered or not retrieved_candidates:
        return {
            "suggested_actions": ["open_observation"],
            "supporting_episode_ids": [],
            "rationale": (
                "Medical comfort check takes priority."
                if distress_triggered
                else "Signal quality insufficient for behavioral matching."
                if not is_acceptable
                else "No historical confirmed precedents found in memory. Open observation recommended."
            ),
        }

    action_map: dict[str, list[str]] = {}
    for cand in retrieved_candidates:
        act = cand.get("action_offered")
        ep_id = cand.get("episode_id", "")
        if act and act in CONTROLLED_ACTIONS:
            action_map.setdefault(act, []).append(ep_id)

    if action_map:
        ranked = sorted(action_map.keys(), key=lambda a: len(action_map[a]), reverse=True)
        top_act = ranked[0]
        return {
            "suggested_actions": ranked[:2],
            "supporting_episode_ids": action_map[top_act],
            "rationale": f"Derived from {len(action_map[top_act])} prior confirmed episode(s).",
        }

    return {
        "suggested_actions": ["open_observation"],
        "supporting_episode_ids": [],
        "rationale": "Historical episodes lack confirmed resolutions.",
    }


def synthesize_parent_view_text(
    mean_f0: float,
    mean_rhythm: float,
    antecedent_id: str,
    top_action: str,
    distress_triggered: bool,
    is_acceptable: bool,
) -> str:
    """Synthesizes accessible, non-diagnostic Parent View summary in warm everyday English."""
    if not is_acceptable:
        return (
            "Sensory signal quality was lower than standard (audio/video gate breach). "
            "Please check in with Child N directly via gentle, open observation."
        )
    if distress_triggered:
        return (
            "Child N's vocal pitch or physical guarding is noticeably elevated compared to usual patterns. "
            "Please pause activities and offer a pediatrician-approved comfort check and water."
        )
    action_label = top_action.replace("_", " ")
    return (
        f"Child N's vocal tone ({mean_f0:.0f} Hz) and motion rhythm ({mean_rhythm:.1f} Hz) "
        f"were observed during {antecedent_id}. Suggested co-regulatory option: {action_label}."
    )
