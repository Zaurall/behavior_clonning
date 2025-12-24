#!/usr/bin/env python3
"""
Comprehensive analysis of PGTO-generated trajectory data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

DATA_DIR = Path("data/pgto")
TARGET_SEGMENTS = 5000


@dataclass
class SegmentStats:
    """Statistics for a single segment."""

    segment_id: str
    restart_costs: np.ndarray  # [R]
    mean_cost: float
    std_cost: float
    best_cost: float
    worst_cost: float

    # Trajectory data
    actions: list[np.ndarray]  # [R] x [T]
    targets: np.ndarray  # [T]
    current_lataccels: list[np.ndarray]  # [R] x [T]


def load_segment_stats(pt_path: Path) -> SegmentStats:
    """Load statistics from a single .pt file."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    num_restarts = data["num_restarts"]
    costs = []
    actions = []
    current_lataccels = []
    targets = None

    for i in range(num_restarts):
        cost = data[f"restart_{i}_cost"]
        if isinstance(cost, torch.Tensor):
            cost = cost.item()
        costs.append(cost)

        traj = data[f"restart_{i}_trajectory"]
        actions.append(traj["actions"].numpy())
        current_lataccels.append(traj["current_lataccel"].numpy())
        if targets is None:
            targets = traj["targets"].numpy()

    costs = np.array(costs)

    return SegmentStats(
        segment_id=data["segment_id"],
        restart_costs=costs,
        mean_cost=costs.mean(),
        std_cost=costs.std(),
        best_cost=costs.min(),
        worst_cost=costs.max(),
        actions=actions,
        targets=targets,
        current_lataccels=current_lataccels,
    )


def load_all_segments(data_dir: Path) -> list[SegmentStats]:
    """Load all segment statistics from directory."""
    pt_files = sorted(data_dir.glob("*.pt"))

    segments = []
    for pt_file in tqdm(pt_files, desc="Loading segments"):
        try:
            stats = load_segment_stats(pt_file)
            segments.append(stats)
        except Exception as e:
            print(f"Warning: Failed to load {pt_file}: {e}")

    return segments


def compute_progress_estimate(data_dir: Path) -> dict:
    """Estimate progress and time to completion."""
    pt_files = sorted(data_dir.glob("*.pt"))

    if len(pt_files) < 2:
        return {"completed": len(pt_files), "target": TARGET_SEGMENTS}

    mtimes = [f.stat().st_mtime for f in pt_files]
    first_time = min(mtimes)
    last_time = max(mtimes)
    elapsed_hours = (last_time - first_time) / 3600

    if elapsed_hours > 0:
        rate_per_hour = len(pt_files) / elapsed_hours
        remaining = TARGET_SEGMENTS - len(pt_files)
        hours_remaining = (
            remaining / rate_per_hour if rate_per_hour > 0 else float("inf")
        )
    else:
        rate_per_hour = 0
        hours_remaining = float("inf")

    return {
        "completed": len(pt_files),
        "target": TARGET_SEGMENTS,
        "pct_complete": len(pt_files) / TARGET_SEGMENTS * 100,
        "elapsed_hours": elapsed_hours,
        "rate_per_hour": rate_per_hour,
        "remaining": TARGET_SEGMENTS - len(pt_files),
        "hours_remaining": hours_remaining,
        "days_remaining": hours_remaining / 24,
    }


def print_report(segments: list[SegmentStats], progress: dict):
    """Print comprehensive analysis report."""

    if not segments:
        print("No segments found!")
        return

    # Gather all costs
    all_restart_costs = np.concatenate([s.restart_costs for s in segments])
    segment_means = np.array([s.mean_cost for s in segments])
    segment_bests = np.array([s.best_cost for s in segments])

    # Sort segments by mean cost for display
    sorted_segments = sorted(segments, key=lambda s: s.mean_cost)

    print("=" * 80)
    print("PGTO DATA ANALYSIS REPORT")
    print("=" * 80)

    # Progress
    print(f"\n{'═' * 30} PROGRESS {'═' * 30}")
    print(
        f"  Segments completed:    {progress['completed']:,} / {progress['target']:,} ({progress.get('pct_complete', 0):.1f}%)"
    )
    if progress.get("rate_per_hour", 0) > 0:
        print(f"  Processing rate:       {progress['rate_per_hour']:.1f} segments/hour")
        print(f"  Elapsed time:          {progress['elapsed_hours']:.1f} hours")
        if progress["hours_remaining"] < float("inf"):
            print(
                f"  Estimated remaining:   {progress['hours_remaining']:.1f} hours ({progress['days_remaining']:.1f} days)"
            )

    # Expected evaluation - THE KEY NUMBER
    expected_score = segment_means.mean()
    expected_sem = segment_means.std() / np.sqrt(len(segments))
    oracle_score = segment_bests.mean()

    print(f"\n{'═' * 30} EXPECTED EVALUATION {'═' * 30}")
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print(
        f"  │  EXPECTED EVAL SCORE:  {expected_score:6.2f} ± {expected_sem:.2f} (SEM)              │"
    )
    print(
        f"  │  Oracle (best restart): {oracle_score:6.2f}                            │"
    )
    print(
        f"  │  Restart selection gap: {expected_score - oracle_score:6.2f}                            │"
    )
    print("  └──────────────────────────────────────────────────────────────┘")

    # ALL SEGMENTS TABLE - crucial for understanding
    print(f"\n{'═' * 30} ALL SEGMENTS {'═' * 30}")
    print(f"  {'Segment':<10} {'Mean':>8} {'Std':>8} {'Best':>8} {'Worst':>8} {'R':>4}")
    print(f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 4}")
    for s in sorted_segments:
        print(
            f"  {s.segment_id:<10} {s.mean_cost:>8.1f} {s.std_cost:>8.1f} {s.best_cost:>8.1f} {s.worst_cost:>8.1f} {len(s.restart_costs):>4}"
        )
    print(f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 4}")
    print(
        f"  {'MEAN':<10} {segment_means.mean():>8.1f} {np.mean([s.std_cost for s in segments]):>8.1f} {segment_bests.mean():>8.1f} {np.mean([s.worst_cost for s in segments]):>8.1f}"
    )

    # Cost distribution
    print(f"\n{'═' * 30} COST DISTRIBUTION {'═' * 30}")
    print(f"  All restarts (n={len(all_restart_costs)}):")
    print(f"    Min:              {all_restart_costs.min():>8.1f}")
    print(f"    5th percentile:   {np.percentile(all_restart_costs, 5):>8.1f}")
    print(f"    25th percentile:  {np.percentile(all_restart_costs, 25):>8.1f}")
    print(f"    Median:           {np.median(all_restart_costs):>8.1f}")
    print(f"    75th percentile:  {np.percentile(all_restart_costs, 75):>8.1f}")
    print(f"    95th percentile:  {np.percentile(all_restart_costs, 95):>8.1f}")
    print(f"    Max:              {all_restart_costs.max():>8.1f}")

    # Variance analysis
    within_stds = np.array([s.std_cost for s in segments])
    within_ranges = np.array([s.worst_cost - s.best_cost for s in segments])

    print(f"\n{'═' * 30} VARIANCE ANALYSIS {'═' * 30}")
    print("  Within-segment (physics stochasticity):")
    print(f"    Mean std across segments:   {within_stds.mean():.2f}")
    print(f"    Mean restart range:         {within_ranges.mean():.2f}")
    print(f"    Max restart range:          {within_ranges.max():.2f}")
    print("  Between-segment (difficulty):")
    print(f"    Std of segment means:       {segment_means.std():.2f}")
    print(
        f"  Ratio (between/within):       {segment_means.std() / within_stds.mean():.2f}x"
    )

    # Action analysis
    all_actions = np.concatenate([a for s in segments for a in s.actions])
    all_action_deltas = np.concatenate(
        [np.diff(a) for s in segments for a in s.actions]
    )

    print(f"\n{'═' * 30} ACTION ANALYSIS {'═' * 30}")
    print("  Distribution:")
    print(f"    Mean:   {all_actions.mean():>7.3f}")
    print(f"    Std:    {all_actions.std():>7.3f}")
    print(f"    Min:    {all_actions.min():>7.3f}")
    print(f"    Max:    {all_actions.max():>7.3f}")
    print("  Clipping:")
    print(f"    At min (-2): {(all_actions <= -1.99).mean() * 100:.2f}%")
    print(f"    At max (+2): {(all_actions >= 1.99).mean() * 100:.2f}%")
    print("  Smoothness:")
    print(f"    Mean |Δaction|: {np.abs(all_action_deltas).mean():.4f}")
    print(f"    Max |Δaction|:  {np.abs(all_action_deltas).max():.4f}")

    # Tracking analysis
    all_tracking_errors = []
    for s in segments:
        targets = s.targets
        for lataccels in s.current_lataccels:
            if len(lataccels) > 1:
                # Error at t: how well did we track target[t-1] with result lataccel[t]
                errors = targets[:-1] - lataccels[1:]
                all_tracking_errors.extend(errors)

    all_tracking_errors = np.array(all_tracking_errors)

    print(f"\n{'═' * 30} TRACKING ANALYSIS {'═' * 30}")
    print("  Tracking error (target - actual):")
    print(f"    Mean:      {all_tracking_errors.mean():>8.4f}")
    print(f"    Mean |e|:  {np.abs(all_tracking_errors).mean():>8.4f}")
    print(f"    MSE:       {(all_tracking_errors**2).mean():>8.4f}")
    print(f"    Std:       {all_tracking_errors.std():>8.4f}")

    # Data quality
    num_restarts = sum(len(s.restart_costs) for s in segments)
    any_nan = any(np.any(np.isnan(s.restart_costs)) for s in segments)
    any_inf = any(np.any(np.isinf(s.restart_costs)) for s in segments)

    print(f"\n{'═' * 30} DATA QUALITY {'═' * 30}")
    print(f"  Total segments:        {len(segments)}")
    print(f"  Total restarts:        {num_restarts}")
    print(f"  Restarts per segment:  {num_restarts / len(segments):.1f}")
    print(f"  Any NaN costs:         {any_nan}")
    print(f"  Any Inf costs:         {any_inf}")

    # High cost segments (potential issues)
    high_cost = [s for s in segments if s.mean_cost > 150]
    if high_cost:
        print(f"\n  ⚠️  High cost segments (>150): {[s.segment_id for s in high_cost]}")

    high_variance = [s for s in segments if s.std_cost > 30]
    if high_variance:
        print(
            f"  ⚠️  High variance segments (std>30): {[s.segment_id for s in high_variance]}"
        )

    # Leaderboard context
    print(f"\n{'═' * 30} LEADERBOARD CONTEXT {'═' * 30}")
    leaderboard = [
        ("bheijden (1st, PPO)", 45.76),
        ("ellenjxu (2nd)", 48.08),
        ("TheConverseEngineer (3rd, tube MPC)", 48.47),
        ("Your CMA-ES baseline", 52.5),
    ]

    print(f"  Your expected PGTO score: {expected_score:.2f}")
    print()
    for name, score in leaderboard:
        diff = expected_score - score
        better = "✓ BEATING" if diff < 0 else "  behind"
        print(f"  {better} {name:35s} {score:>6.2f}  (diff: {diff:+.2f})")

    print()
    print("=" * 80)


def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist")
        return

    print(f"Loading data from {DATA_DIR}...")
    progress = compute_progress_estimate(DATA_DIR)
    segments = load_all_segments(DATA_DIR)

    if segments:
        print(f"Loaded {len(segments)} segments\n")

    print_report(segments, progress)


if __name__ == "__main__":
    main()
