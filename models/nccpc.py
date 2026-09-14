"""
Project N: Canonical Psychometric Definitions for the Non-Communicating Children's Pain Checklist.
Authoritative item sets, subscales, and scoring policies for NCCPC-PV and NCCPC-R.
Primary Sources:
- Breau et al. (2002), Anesthesiology, 96(3):528-535. (NCCPC-PV, 27 items, 6 subscales)
- Breau et al. (2002), Pain, 99(1-2):349-357. (NCCPC-R, 30 items, 7 subscales)
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

# Response scale: 0 = Not at all, 1 = Just a little, 2 = Fairly often, 3 = Very often, "NA" = Not applicable
ALLOWED_RESPONSES: Final[set[int | str]] = {0, 1, 2, 3, "NA", "na", "Na"}

# ==============================================================================
# Canonical NCCPC-PV Items (27 Items, 6 Subscales)
# ==============================================================================
NCCPC_PV_VOCAL_ITEMS: Final[list[tuple[str, str]]] = [
    ("moaning_whining_whimpering", "Moaning, whining, whimpering (fairly soft)"),
    ("crying", "Crying (moderately loud)"),
    ("screaming_yelling", "Screaming/yelling (very loud)"),
    ("pain_sound_word", "A specific sound or word for pain (e.g., a cry, 'ouch')"),
]

NCCPC_PV_SOCIAL_ITEMS: Final[list[tuple[str, str]]] = [
    ("not_cooperating_cranky", "Not cooperating, cranky"),
    ("less_interaction_withdrawn", "Less interaction with others, withdrawn"),
    ("seeking_comfort_closeness", "Seeking comfort or physical closeness"),
    ("difficult_to_comfort_please", "Being difficult to comfort or please"),
]

NCCPC_PV_FACIAL_ITEMS: Final[list[tuple[str, str]]] = [
    ("furrowed_brow", "A furrowed brow"),
    (
        "change_in_eyes_squinching",
        "A change in eyes, including squinching of eyes, eyes open wide, eyes frowning",
    ),
    ("turning_mouth_down", "Turning down of mouth, not smiling"),
    (
        "lips_puckering_tight_pouting_quivering",
        "Lips puckering up, tight, pouting, or quivering",
    ),
    (
        "clenching_teeth_chewing_thrusting_tongue",
        "Clenching or grinding teeth, chewing, or thrusting tongue out",
    ),
]

NCCPC_PV_ACTIVITY_ITEMS: Final[list[tuple[str, str]]] = [
    ("not_moving_less_active_quiet", "Not moving, less active, quiet"),
    ("jumping_agitated_fidgety", "Jumping around, agitated, fidgety"),
]

NCCPC_PV_BODY_LIMBS_ITEMS: Final[list[tuple[str, str]]] = [
    ("floppy", "Floppy"),
    ("stiff_spastic_rigid_tense", "Stiff, spastic, rigid, tense"),
    (
        "gesturing_touching_hurt_part",
        "Gesturing to or touching a part of the body that seems to hurt",
    ),
    ("protecting_favoring_guarding", "Protecting, favoring, or guarding part of the body"),
    ("flinching_moving_away", "Flinching or moving body part away"),
    (
        "moving_body_specifically",
        "Moving body in a specific way (e.g., head back, arms/legs out)",
    ),
]

NCCPC_PV_PHYSIOLOGICAL_ITEMS: Final[list[tuple[str, str]]] = [
    ("shivering", "Shivering"),
    ("change_in_color_pallor", "Change in color, pallor"),
    ("sweating_perspiring", "Sweating, perspiring"),
    ("tears", "Tears"),
    ("sharp_breath_gasping", "Sharp intake of breath, gasping"),
    ("breath_holding", "Breath holding"),
]

# Additional Subscale for NCCPC-R (30 Items Total)
NCCPC_R_EATING_SLEEPING_ITEMS: Final[list[tuple[str, str]]] = [
    ("eating_less_not_interested", "Eating less, not interested in food"),
    ("increased_sleep", "Increase in sleep"),
    ("decreased_sleep", "Decrease in sleep"),
]

# Subscale groupings
NCCPC_PV_SUBSCALES: Final[dict[str, list[tuple[str, str]]]] = {
    "vocal": NCCPC_PV_VOCAL_ITEMS,  # 4
    "social": NCCPC_PV_SOCIAL_ITEMS,  # 4
    "facial": NCCPC_PV_FACIAL_ITEMS,  # 5
    "activity": NCCPC_PV_ACTIVITY_ITEMS,  # 2
    "body_and_limbs": NCCPC_PV_BODY_LIMBS_ITEMS,  # 6
    "physiological": NCCPC_PV_PHYSIOLOGICAL_ITEMS,  # 6
}

NCCPC_R_SUBSCALES: Final[dict[str, list[tuple[str, str]]]] = {
    **NCCPC_PV_SUBSCALES,
    "eating_and_sleeping": NCCPC_R_EATING_SLEEPING_ITEMS,  # 3
}

# Sets of canonical item keys
CANONICAL_PV_ITEMS: Final[set[str]] = {
    item_id for subscale in NCCPC_PV_SUBSCALES.values() for item_id, _ in subscale
}

CANONICAL_R_ITEMS: Final[set[str]] = {
    item_id for subscale in NCCPC_R_SUBSCALES.values() for item_id, _ in subscale
}

PV_CUTOFF_SCORE: Final[int] = 11  # Sensitivity 0.88, Specificity 0.81 (Breau et al., 2002)
R_CUTOFF_SCORE: Final[int] = 7  # Sensitivity 0.84, Specificity 0.77 (Breau et al., 2002)


@dataclass(frozen=True)
class NCCPCScoreResult:
    instrument: str  # 'nccpc_pv' or 'nccpc_r'
    score: int
    max_possible_score: int
    cutoff_threshold: int
    cutoff_breached: bool
    na_count: int
    subscale_scores: dict[str, int]
    high_missingness_advisory: bool
    is_indeterminate: (
        bool  # True if na_count > threshold, meaning the assessment cannot rule out pain
    )


def score_nccpc_pv(
    items: Mapping[str, int | str],
    strict_keys: bool = True,
) -> NCCPCScoreResult:
    """
    Scores the 27-item NCCPC-PV instrument.
    Accepts only exact integers 0, 1, 2, 3, or 'NA'. Rejects booleans and floats.
    'NA' responses contribute 0 to the raw sum while tracking na_count.
    """
    if strict_keys:
        provided_keys = set(items.keys())
        if provided_keys != CANONICAL_PV_ITEMS:
            missing = CANONICAL_PV_ITEMS - provided_keys
            extra = provided_keys - CANONICAL_PV_ITEMS
            raise ValueError(
                f"NCCPC-PV items mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

    total_score = 0
    na_count = 0
    subscale_scores: dict[str, int] = {}

    for subscale_name, subscale_items in NCCPC_PV_SUBSCALES.items():
        subscale_total = 0
        for item_id, _ in subscale_items:
            val = items.get(item_id, 0)
            if isinstance(val, str) and val.strip().upper() == "NA":
                na_count += 1
            elif isinstance(val, int) and not isinstance(val, bool):
                if not 0 <= val <= 3:
                    raise ValueError(f"Item '{item_id}' score {val} out of bounds [0, 3].")
                subscale_total += val
            else:
                raise ValueError(
                    f"Invalid response '{val}' (type {type(val).__name__}) for item '{item_id}'. "
                    f"Must be an integer in [0, 3] or 'NA'."
                )
        subscale_scores[subscale_name] = subscale_total
        total_score += subscale_total

    is_indeterminate = na_count > 5
    return NCCPCScoreResult(
        instrument="nccpc_pv",
        score=total_score,
        max_possible_score=81,
        cutoff_threshold=PV_CUTOFF_SCORE,
        cutoff_breached=total_score >= PV_CUTOFF_SCORE,
        na_count=na_count,
        subscale_scores=subscale_scores,
        high_missingness_advisory=is_indeterminate,
        is_indeterminate=is_indeterminate,
    )


def score_nccpc_r(
    items: Mapping[str, int | str],
    strict_keys: bool = True,
) -> NCCPCScoreResult:
    """
    Scores the 30-item NCCPC-R instrument.
    Accepts only exact integers 0, 1, 2, 3, or 'NA'. Rejects booleans and floats.
    'NA' responses contribute 0 to the raw sum while tracking na_count.
    """
    if strict_keys:
        provided_keys = set(items.keys())
        if provided_keys != CANONICAL_R_ITEMS:
            missing = CANONICAL_R_ITEMS - provided_keys
            extra = provided_keys - CANONICAL_R_ITEMS
            raise ValueError(
                f"NCCPC-R items mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

    total_score = 0
    na_count = 0
    subscale_scores: dict[str, int] = {}

    for subscale_name, subscale_items in NCCPC_R_SUBSCALES.items():
        subscale_total = 0
        for item_id, _ in subscale_items:
            val = items.get(item_id, 0)
            if isinstance(val, str) and val.strip().upper() == "NA":
                na_count += 1
            elif isinstance(val, int) and not isinstance(val, bool):
                if not 0 <= val <= 3:
                    raise ValueError(f"Item '{item_id}' score {val} out of bounds [0, 3].")
                subscale_total += val
            else:
                raise ValueError(
                    f"Invalid response '{val}' (type {type(val).__name__}) for item '{item_id}'. "
                    f"Must be an integer in [0, 3] or 'NA'."
                )
        subscale_scores[subscale_name] = subscale_total
        total_score += subscale_total

    is_indeterminate = na_count > 6
    return NCCPCScoreResult(
        instrument="nccpc_r",
        score=total_score,
        max_possible_score=90,
        cutoff_threshold=R_CUTOFF_SCORE,
        cutoff_breached=total_score >= R_CUTOFF_SCORE,
        na_count=na_count,
        subscale_scores=subscale_scores,
        high_missingness_advisory=is_indeterminate,
        is_indeterminate=is_indeterminate,
    )
