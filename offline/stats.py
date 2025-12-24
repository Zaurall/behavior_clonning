#!/usr/bin/env python3
"""
Comprehensive analysis of PGTO-generated trajectory data.

Provides expected evaluation scores, cost breakdowns, trajectory analysis,
and identifies potential issues in the generated data.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class SegmentStats:
    """Statistics for a single segment."""

    segment_id: str
    restart_costs: np.ndarray  # [R]
    mean_cost: float
    std_cost: float
    best_cost: float
    worst_cost: float

    # Per-restart trajectory data (optional, for detailed analysis)
    actions: Optional[list[np.ndarray]] = None  # [R] x [T]
    targets: Optional[np.ndarray] = None  # [T]
    current_lataccels: Optional[list[np.ndarray]] = None  # [R] x [T]


def load_segment_stats(pt_path: Path, load_trajectories: bool = False) -> SegmentStats:
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

        if load_trajectories:
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
        actions=actions if load_trajectories else None,
        targets=targets if load_trajectories else None,
        current_lataccels=current_lataccels if load_trajectories else None,
    )


def load_all_segments(
    data_dir: Path,
    load_trajectories: bool = False,
    max_segments: Optional[int] = None,
) -> list[SegmentStats]:
    """Load all segment statistics from directory."""
    pt_files = sorted(data_dir.glob("*.pt"))
    if max_segments:
        pt_files = pt_files[:max_segments]

    segments = []
    for pt_file in tqdm(pt_files, desc="Loading segments"):
        try:
            stats = load_segment_stats(pt_file, load_trajectories=load_trajectories)
            segments.append(stats)
        except Exception as e:
            print(f"Warning: Failed to load {pt_file}: {e}")

    return segments


def compute_expected_eval(segments: list[SegmentStats]) -> dict:
    """
    Compute expected evaluation metrics.

    The key insight: all restarts use the same PGTO policy, so variance
    between restarts is due to physics stochasticity, not policy differences.
    The mean across restarts estimates expected performance.
    """
    # Per-segment means (expected cost per segment)
    segment_means = np.array([s.mean_cost for s in segments])

    # Per-segment bests (if we could pick best restart)
    segment_bests = np.array([s.best_cost for s in segments])

    # All individual restart costs
    all_costs = np.concatenate([s.restart_costs for s in segments])

    return {
        # PRIMARY METRIC: Expected eval score
        "expected_eval_score": segment_means.mean(),
        "expected_eval_std": segment_means.std(),
        "expected_eval_sem": segment_means.std() / np.sqrt(len(segments)),
        # If we could magically pick best restart each time
        "oracle_best_restart_score": segment_bests.mean(),
        # Gap shows potential value of restart selection
        "restart_selection_gap": segment_means.mean() - segment_bests.mean(),
        # Distribution stats
        "cost_median": np.median(all_costs),
        "cost_25th": np.percentile(all_costs, 25),
        "cost_75th": np.percentile(all_costs, 75),
        "cost_5th": np.percentile(all_costs, 5),
        "cost_95th": np.percentile(all_costs, 95),
        "cost_min": all_costs.min(),
        "cost_max": all_costs.max(),
    }


def compute_variance_analysis(segments: list[SegmentStats]) -> dict:
    """Analyze variance within and between segments."""
    within_segment_stds = np.array([s.std_cost for s in segments])
    within_segment_ranges = np.array([s.worst_cost - s.best_cost for s in segments])
    segment_means = np.array([s.mean_cost for s in segments])

    return {
        # Within-segment variance (physics stochasticity)
        "mean_within_segment_std": within_segment_stds.mean(),
        "median_within_segment_std": np.median(within_segment_stds),
        "max_within_segment_std": within_segment_stds.max(),
        # Range between best and worst restart
        "mean_restart_range": within_segment_ranges.mean(),
        "max_restart_range": within_segment_ranges.max(),
        # Between-segment variance (segment difficulty)
        "between_segment_std": segment_means.std(),
        # Coefficient of variation (normalized spread)
        "within_cv": within_segment_stds.mean() / segment_means.mean(),
        "between_cv": segment_means.std() / segment_means.mean(),
    }


def compute_difficulty_analysis(segments: list[SegmentStats], n_show: int = 10) -> dict:
    """Identify easy and hard segments."""
    sorted_by_mean = sorted(segments, key=lambda s: s.mean_cost)

    easiest = [(s.segment_id, s.mean_cost, s.std_cost) for s in sorted_by_mean[:n_show]]
    hardest = [
        (s.segment_id, s.mean_cost, s.std_cost) for s in sorted_by_mean[-n_show:]
    ]

    # High variance segments (unpredictable)
    sorted_by_std = sorted(segments, key=lambda s: s.std_cost, reverse=True)
    most_variable = [
        (s.segment_id, s.mean_cost, s.std_cost) for s in sorted_by_std[:n_show]
    ]

    means = np.array([s.mean_cost for s in segments])

    return {
        "easiest_segments": easiest,
        "hardest_segments": hardest,
        "most_variable_segments": most_variable,
        "num_under_30": (means < 30).sum(),
        "num_under_40": (means < 40).sum(),
        "num_under_50": (means < 50).sum(),
        "num_over_100": (means > 100).sum(),
        "num_over_200": (means > 200).sum(),
    }


def compute_action_analysis(segments: list[SegmentStats]) -> dict:
    """Analyze action distributions and smoothness."""
    if not segments or segments[0].actions is None:
        return {"error": "No trajectory data loaded"}

    all_actions = []
    all_action_deltas = []
    action_stds_per_segment = []

    for seg in segments:
        for actions in seg.actions:
            all_actions.extend(actions)
            deltas = np.diff(actions)
            all_action_deltas.extend(deltas)
            action_stds_per_segment.append(actions.std())

    all_actions = np.array(all_actions)
    all_action_deltas = np.array(all_action_deltas)

    return {
        # Action distribution
        "action_mean": all_actions.mean(),
        "action_std": all_actions.std(),
        "action_min": all_actions.min(),
        "action_max": all_actions.max(),
        "action_median": np.median(all_actions),
        # Clipping analysis
        "pct_at_min_clip": (all_actions <= -1.99).mean() * 100,
        "pct_at_max_clip": (all_actions >= 1.99).mean() * 100,
        # Smoothness
        "action_delta_mean": np.abs(all_action_deltas).mean(),
        "action_delta_std": all_action_deltas.std(),
        "action_delta_max": np.abs(all_action_deltas).max(),
        # Per-segment action variance
        "mean_action_std_per_segment": np.mean(action_stds_per_segment),
    }


def compute_tracking_analysis(segments: list[SegmentStats]) -> dict:
    """Analyze tracking error patterns."""
    if not segments or segments[0].current_lataccels is None:
        return {"error": "No trajectory data loaded"}

    # We'll look at error = target - current_lataccel
    # Note: current_lataccel[t] is state BEFORE action at t, so
    # current_lataccel[t+1] is result after action at t

    all_errors = []
    error_by_timestep = []

    for seg in segments:
        targets = seg.targets
        for lataccels in seg.current_lataccels:
            # Shifted error: compare target[t] to result[t+1]
            if len(lataccels) > 1:
                errors = targets[:-1] - lataccels[1:]
                all_errors.extend(errors)

                # Track by timestep for temporal analysis
                if len(error_by_timestep) == 0:
                    error_by_timestep = [[] for _ in range(len(errors))]
                for t, e in enumerate(errors):
                    if t < len(error_by_timestep):
                        error_by_timestep[t].append(e)

    all_errors = np.array(all_errors)

    # Temporal pattern: MSE by timestep
    mse_by_timestep = [
        np.mean(np.array(errs) ** 2) if errs else 0 for errs in error_by_timestep
    ]

    # Find worst periods
    window = 20
    if len(mse_by_timestep) >= window:
        rolling_mse = np.convolve(
            mse_by_timestep, np.ones(window) / window, mode="valid"
        )
        worst_window_start = np.argmax(rolling_mse)
    else:
        worst_window_start = 0
        rolling_mse = mse_by_timestep

    return {
        "tracking_error_mean": all_errors.mean(),
        "tracking_error_std": all_errors.std(),
        "tracking_error_abs_mean": np.abs(all_errors).mean(),
        "tracking_mse": (all_errors**2).mean(),
        # Temporal patterns
        "mse_first_50_steps": np.mean(mse_by_timestep[:50])
        if len(mse_by_timestep) >= 50
        else np.nan,
        "mse_last_50_steps": np.mean(mse_by_timestep[-50:])
        if len(mse_by_timestep) >= 50
        else np.nan,
        "worst_window_start": worst_window_start,
        "worst_window_mse": rolling_mse[worst_window_start]
        if len(rolling_mse) > 0
        else np.nan,
    }


def compute_data_quality_checks(segments: list[SegmentStats]) -> dict:
    """Check for potential data issues."""
    issues = []

    # Check for extreme costs
    for seg in segments:
        if seg.mean_cost > 500:
            issues.append(
                f"Very high cost: {seg.segment_id} (mean={seg.mean_cost:.1f})"
            )
        if seg.std_cost > 100:
            issues.append(
                f"Very high variance: {seg.segment_id} (std={seg.std_cost:.1f})"
            )
        if np.any(np.isnan(seg.restart_costs)) or np.any(np.isinf(seg.restart_costs)):
            issues.append(f"NaN/Inf cost: {seg.segment_id}")

    all_costs = np.concatenate([s.restart_costs for s in segments])

    return {
        "num_segments": len(segments),
        "num_restarts_total": len(all_costs),
        "any_nan_costs": np.any(np.isnan(all_costs)),
        "any_inf_costs": np.any(np.isinf(all_costs)),
        "num_issues": len(issues),
        "issues": issues[:20],  # Cap at 20
    }


def compute_progress_estimate(data_dir: Path, target_segments: int = 5000) -> dict:
    """Estimate progress and time to completion."""
    pt_files = sorted(data_dir.glob("*.pt"))

    if len(pt_files) < 2:
        return {"completed": len(pt_files), "target": target_segments}

    # Get file modification times
    mtimes = [f.stat().st_mtime for f in pt_files]
    first_time = min(mtimes)
    last_time = max(mtimes)
    elapsed_hours = (last_time - first_time) / 3600

    if elapsed_hours > 0:
        rate_per_hour = len(pt_files) / elapsed_hours
        remaining = target_segments - len(pt_files)
        hours_remaining = (
            remaining / rate_per_hour if rate_per_hour > 0 else float("inf")
        )
    else:
        rate_per_hour = 0
        hours_remaining = float("inf")

    return {
        "completed": len(pt_files),
        "target": target_segments,
        "pct_complete": len(pt_files) / target_segments * 100,
        "elapsed_hours": elapsed_hours,
        "rate_per_hour": rate_per_hour,
        "remaining": target_segments - len(pt_files),
        "hours_remaining": hours_remaining,
        "days_remaining": hours_remaining / 24,
    }


def print_report(
    segments: list[SegmentStats],
    eval_stats: dict,
    variance_stats: dict,
    difficulty_stats: dict,
    action_stats: dict,
    tracking_stats: dict,
    quality_stats: dict,
    progress_stats: dict,
):
    """Print comprehensive analysis report."""

    print("=" * 80)
    print("PGTO DATA ANALYSIS REPORT")
    print("=" * 80)

    # Progress
    print(f"\n{'═' * 30} PROGRESS {'═' * 30}")
    print(
        f"  Segments completed:    {progress_stats['completed']:,} / {progress_stats['target']:,} ({progress_stats.get('pct_complete', 0):.1f}%)"
    )
    if progress_stats.get("rate_per_hour", 0) > 0:
        print(
            f"  Processing rate:       {progress_stats['rate_per_hour']:.1f} segments/hour"
        )
        print(f"  Elapsed time:          {progress_stats['elapsed_hours']:.1f} hours")
        if progress_stats["hours_remaining"] < float("inf"):
            print(
                f"  Estimated remaining:   {progress_stats['hours_remaining']:.1f} hours ({progress_stats['days_remaining']:.1f} days)"
            )

    # Expected evaluation
    print(f"\n{'═' * 30} EXPECTED EVALUATION {'═' * 30}")
    print("  ┌────────────────────────────────────────────────────────┐")
    print(
        f"  │  EXPECTED EVAL SCORE:  {eval_stats['expected_eval_score']:.2f} ± {eval_stats['expected_eval_sem']:.2f} (SEM)  │"
    )
    print("  └────────────────────────────────────────────────────────┘")
    print("  (This estimates the score if BC perfectly learns PGTO policy)")
    print()
    print(f"  Oracle (best restart):   {eval_stats['oracle_best_restart_score']:.2f}")
    print(f"  Restart selection gap:   {eval_stats['restart_selection_gap']:.2f}")
    print()
    print("  Cost distribution:")
    print(f"    5th percentile:   {eval_stats['cost_5th']:.1f}")
    print(f"    25th percentile:  {eval_stats['cost_25th']:.1f}")
    print(f"    Median:           {eval_stats['cost_median']:.1f}")
    print(f"    75th percentile:  {eval_stats['cost_75th']:.1f}")
    print(f"    95th percentile:  {eval_stats['cost_95th']:.1f}")
    print(
        f"    Min / Max:        {eval_stats['cost_min']:.1f} / {eval_stats['cost_max']:.1f}"
    )

    # Variance analysis
    print(f"\n{'═' * 30} VARIANCE ANALYSIS {'═' * 30}")
    print("  Within-segment (physics stochasticity):")
    print(f"    Mean std:           {variance_stats['mean_within_segment_std']:.2f}")
    print(f"    Mean restart range: {variance_stats['mean_restart_range']:.2f}")
    print(f"    Max restart range:  {variance_stats['max_restart_range']:.2f}")
    print()
    print("  Between-segment (difficulty variation):")
    print(f"    Std of segment means: {variance_stats['between_segment_std']:.2f}")
    print()
    print("  Coefficient of variation:")
    print(f"    Within-segment CV:  {variance_stats['within_cv']:.3f}")
    print(f"    Between-segment CV: {variance_stats['between_cv']:.3f}")

    # Difficulty breakdown
    print(f"\n{'═' * 30} SEGMENT DIFFICULTY {'═' * 30}")
    print("  Cost buckets:")
    print(f"    Under 30:    {difficulty_stats['num_under_30']:4d} segments")
    print(f"    Under 40:    {difficulty_stats['num_under_40']:4d} segments")
    print(f"    Under 50:    {difficulty_stats['num_under_50']:4d} segments")
    print(f"    Over 100:    {difficulty_stats['num_over_100']:4d} segments")
    print(f"    Over 200:    {difficulty_stats['num_over_200']:4d} segments")
    print()
    print("  Easiest segments:")
    for seg_id, mean, std in difficulty_stats["easiest_segments"][:5]:
        print(f"    {seg_id}: {mean:.1f} ± {std:.1f}")
    print()
    print("  Hardest segments:")
    for seg_id, mean, std in difficulty_stats["hardest_segments"][:5]:
        print(f"    {seg_id}: {mean:.1f} ± {std:.1f}")
    print()
    print("  Most variable segments (high physics uncertainty):")
    for seg_id, mean, std in difficulty_stats["most_variable_segments"][:5]:
        print(f"    {seg_id}: {mean:.1f} ± {std:.1f}")

    # Action analysis
    print(f"\n{'═' * 30} ACTION ANALYSIS {'═' * 30}")
    if "error" not in action_stats:
        print("  Distribution:")
        print(f"    Mean:   {action_stats['action_mean']:.3f}")
        print(f"    Std:    {action_stats['action_std']:.3f}")
        print(
            f"    Range:  [{action_stats['action_min']:.3f}, {action_stats['action_max']:.3f}]"
        )
        print()
        print("  Clipping (hitting limits):")
        print(f"    At min (-2): {action_stats['pct_at_min_clip']:.2f}%")
        print(f"    At max (+2): {action_stats['pct_at_max_clip']:.2f}%")
        print()
        print("  Smoothness:")
        print(f"    Mean |Δaction|: {action_stats['action_delta_mean']:.4f}")
        print(f"    Max |Δaction|:  {action_stats['action_delta_max']:.4f}")
    else:
        print(f"  {action_stats['error']}")

    # Tracking analysis
    print(f"\n{'═' * 30} TRACKING ANALYSIS {'═' * 30}")
    if "error" not in tracking_stats:
        print("  Overall:")
        print(f"    Mean error:     {tracking_stats['tracking_error_mean']:.4f}")
        print(f"    Mean |error|:   {tracking_stats['tracking_error_abs_mean']:.4f}")
        print(f"    MSE:            {tracking_stats['tracking_mse']:.4f}")
        print()
        print("  Temporal pattern:")
        print(f"    MSE first 50 steps:  {tracking_stats['mse_first_50_steps']:.4f}")
        print(f"    MSE last 50 steps:   {tracking_stats['mse_last_50_steps']:.4f}")
        if not np.isnan(tracking_stats["worst_window_mse"]):
            print(
                f"    Worst 20-step window: steps {tracking_stats['worst_window_start']}-{tracking_stats['worst_window_start'] + 20} (MSE={tracking_stats['worst_window_mse']:.4f})"
            )
    else:
        print(f"  {tracking_stats['error']}")

    # Data quality
    print(f"\n{'═' * 30} DATA QUALITY {'═' * 30}")
    print(f"  Total segments:   {quality_stats['num_segments']}")
    print(f"  Total restarts:   {quality_stats['num_restarts_total']}")
    print(f"  NaN costs:        {quality_stats['any_nan_costs']}")
    print(f"  Inf costs:        {quality_stats['any_inf_costs']}")
    print(f"  Issues found:     {quality_stats['num_issues']}")
    if quality_stats["issues"]:
        print("\n  Issues:")
        for issue in quality_stats["issues"][:10]:
            print(f"    - {issue}")

    # Leaderboard context
    print(f"\n{'═' * 30} LEADERBOARD CONTEXT {'═' * 30}")
    expected = eval_stats["expected_eval_score"]
    print(f"  Your expected:  {expected:.2f}")
    print()
    leaderboard = [
        ("bheijden (1st)", 45.76),
        ("ellenjxu (2nd)", 48.08),
        ("TheConverseEngineer (3rd)", 48.47),
        ("CMA-ES baseline", 52.5),
    ]
    for name, score in leaderboard:
        diff = expected - score
        indicator = "✓" if diff < 0 else " "
        print(f"  {indicator} {name:30s}: {score:.2f}  (you: {diff:+.2f})")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze PGTO generated data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pgto"),
        help="Directory containing .pt files",
    )
    parser.add_argument(
        "--load-trajectories",
        action="store_true",
        help="Load full trajectory data for detailed analysis (slower)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Max segments to analyze (for quick checks)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist")
        return

    print(f"Loading data from {args.data_dir}...")

    # Progress estimate (before loading)
    progress_stats = compute_progress_estimate(args.data_dir)

    # Load segment data
    segments = load_all_segments(
        args.data_dir,
        load_trajectories=args.load_trajectories,
        max_segments=args.max_segments,
    )

    if not segments:
        print("No segments found!")
        return

    print(f"Loaded {len(segments)} segments, analyzing...\n")

    # Compute all statistics
    eval_stats = compute_expected_eval(segments)
    variance_stats = compute_variance_analysis(segments)
    difficulty_stats = compute_difficulty_analysis(segments)
    action_stats = compute_action_analysis(segments)
    tracking_stats = compute_tracking_analysis(segments)
    quality_stats = compute_data_quality_checks(segments)

    # Print report
    print_report(
        segments=segments,
        eval_stats=eval_stats,
        variance_stats=variance_stats,
        difficulty_stats=difficulty_stats,
        action_stats=action_stats,
        tracking_stats=tracking_stats,
        quality_stats=quality_stats,
        progress_stats=progress_stats,
    )


if __name__ == "__main__":
    main()
