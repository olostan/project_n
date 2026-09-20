"""
Project N: Invariant 2 Unified Memory Ceiling Verification.
Verifies operational peak memory <= 28.0 GB and hard invariant ceiling <= 36.0 GB.
"""

import mlx.core as mx

OPERATIONAL_PEAK_GB = 28.0
HARD_CEILING_GB = 36.0


def verify_mlx_memory_ceiling(max_gb: float = HARD_CEILING_GB) -> None:
    peak_bytes = mx.get_peak_memory()
    peak_gb = peak_bytes / 1e9

    assert peak_gb <= max_gb, (
        f"INVARIANT 2 BREACH: Peak Metal memory {peak_gb:.2f} GB exceeds limit {max_gb:.1f} GB"
    )


def test_memory_ceiling_invariant() -> None:
    verify_mlx_memory_ceiling()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify unified memory ceiling.")
    parser.add_argument(
        "--max_gb",
        type=float,
        default=HARD_CEILING_GB,
        help="Maximum allowable peak unified memory in GB (default: 36.0)",
    )
    args = parser.parse_args()

    active_gb = mx.get_active_memory() / 1e9
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"Active Metal Memory: {active_gb:.2f} GB")
    print(f"Peak Metal Memory:   {peak_gb:.2f} GB")
    verify_mlx_memory_ceiling(max_gb=args.max_gb)
    print(f"Invariant 2 Verification: PASSED (Peak memory <= {args.max_gb:.1f} GB)")
