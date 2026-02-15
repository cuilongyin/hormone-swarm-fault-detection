#!/usr/bin/env python3
"""Scalability Analysis for Fault Detection Methods.

This module provides a comprehensive framework for evaluating how fault detection
methods scale with swarm size. It implements a rigorous experimental protocol with
statistical validation across multiple random seeds.

The experiment systematically tests performance degradation (or improvement) as the
number of agents increases from 30 to 120, providing insights into the scalability
characteristics of each detection approach.

Example
-------
To run the complete scalability experiment:

    $ python test_scalability.py

This will generate a CSV file with results and print a comprehensive summary.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulator import ScalingMode, SimulationConfig
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator
from src.methods.threshold import ThresholdDetector
from src.methods.voting_simplified import SimpleVotingSimulator

# Experimental parameters
SWARM_SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
NUM_SEEDS = 10  # Statistical validity through repeated trials
FAULT_RATE = 0.25
RUN_TIME = 20.0
CONFIDENCE_LEVEL = 0.95


# Quick test configuration (uncomment for rapid iteration)
# NUM_SEEDS = 3
# RUN_TIME = 10.0


@dataclass
class ScalabilityMetrics:
    """Performance metrics for a single experimental run.

    Attributes
    ----------
    swarm_size : int
        Number of agents in the swarm.
    method : str
        Name of the fault detection method.
    seed : int
        Random seed used for this run.
    task_completion : float
        Overall task completion rate [0, 1].
    precision : float
        Precision of fault detection [0, 1].
    recall : float
        Recall of fault detection [0, 1].
    cascade_prevention : float
        Fraction of cascade damage prevented [0, 1].
    response_time : float
        Mean time to detect and quarantine faults (seconds).
    healthy_preserved : float
        Fraction of healthy agents that remained unimpaired [0, 1].
    quarantine_efficiency : float
        Ratio of true positives to total quarantines [0, 1].
    false_positive_rate : float
        Rate of healthy agents incorrectly quarantined [0, 1].
    agents_saved : int
        Number of agents protected from cascade damage.
    simulation_time : float
        Wall-clock time for simulation execution (seconds).
    density : float
        Agent density (agents per square meter).
    """

    swarm_size: int
    method: str
    seed: int
    task_completion: float
    precision: float
    recall: float
    cascade_prevention: float
    response_time: float
    healthy_preserved: float
    quarantine_efficiency: float
    false_positive_rate: float
    agents_saved: int
    simulation_time: float
    density: float


@dataclass
class AggregatedMetrics:
    """Statistical summary across multiple experimental runs.

    Provides means, confidence intervals, and success rates for all metrics,
    aggregated across multiple random seeds to ensure statistical validity.

    Attributes
    ----------
    swarm_size : int
        Number of agents tested.
    method : str
        Detection method name.
    task_completion_mean : float
        Mean task completion rate.
    task_completion_ci : tuple of float
        95% confidence interval for task completion.
    precision_mean : float
        Mean precision of fault detection.
    precision_ci : tuple of float
        95% confidence interval for precision.
    recall_mean : float
        Mean recall of fault detection.
    recall_ci : tuple of float
        95% confidence interval for recall.
    cascade_prevention_mean : float
        Mean cascade prevention rate.
    cascade_prevention_ci : tuple of float
        95% confidence interval for cascade prevention.
    response_time_mean : float
        Mean detection response time.
    response_time_ci : tuple of float
        95% confidence interval for response time.
    success_rate : float
        Fraction of runs that completed successfully [0, 1].
    runtime_mean : float
        Mean wall-clock execution time (seconds).
    raw_results : list of ScalabilityMetrics
        Individual run results for detailed analysis.
    """

    swarm_size: int
    method: str
    task_completion_mean: float
    task_completion_ci: Tuple[float, float]
    precision_mean: float
    precision_ci: Tuple[float, float]
    recall_mean: float
    recall_ci: Tuple[float, float]
    cascade_prevention_mean: float
    cascade_prevention_ci: Tuple[float, float]
    response_time_mean: float
    response_time_ci: Tuple[float, float]
    success_rate: float
    runtime_mean: float
    raw_results: List[ScalabilityMetrics] = field(default_factory=list)


def compute_confidence_interval(
        values: List[float], confidence: float = CONFIDENCE_LEVEL
) -> Tuple[float, float]:
    """Compute confidence interval using Student's t-distribution.

    Parameters
    ----------
    values : list of float
        Sample values for which to compute the interval.
    confidence : float, optional
        Confidence level (default: 0.95 for 95% CI).

    Returns
    -------
    lower_bound : float
        Lower bound of the confidence interval.
    upper_bound : float
        Upper bound of the confidence interval.

    Notes
    -----
    Uses the t-distribution to account for small sample sizes. For samples
    with n > 30, this approximates the normal distribution.
    """
    # Filter invalid values
    valid_values = [v for v in values if v is not None and not np.isnan(v)]

    if len(valid_values) == 0:
        return (0.0, 0.0)

    if len(valid_values) == 1:
        return (valid_values[0], valid_values[0])

    mean = np.mean(valid_values)
    std_error = np.std(valid_values, ddof=1) / np.sqrt(len(valid_values))

    # Student's t critical value
    degrees_of_freedom = len(valid_values) - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)

    margin = t_critical * std_error

    return (mean - margin, mean + margin)


async def run_single_experiment(
        swarm_size: int, method: str, seed: int
) -> Optional[ScalabilityMetrics]:
    """Execute a single scalability experiment.

    Parameters
    ----------
    swarm_size : int
        Number of agents in the swarm.
    method : str
        Detection method name ('Baseline', 'Threshold', 'Voting', 'Hormone').
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    metrics : ScalabilityMetrics or None
        Collected metrics, or None if the experiment failed.

    Notes
    -----
    The baseline method is handled specially as it has no fault detection
    capability. For other methods, a baseline reference is generated to
    compute cascade prevention metrics.
    """
    try:
        # Compute density for reference
        arena_size = 100.0
        density = swarm_size / (arena_size ** 2)

        start_time = time.time()

        # Baseline has no detection capability
        if method == "Baseline":
            config = SimulationConfig(
                num_agents=swarm_size,
                fault_rate=FAULT_RATE,
                run_time=RUN_TIME,
                scaling_mode=ScalingMode.FIXED_ARENA,
                enable_quarantine=False,
                random_seed=seed,
            )

            simulator = BaselineSimulator(config)
            metrics = await simulator.run_simulation()

            detection_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "avg_response_time": float("inf"),
                "tp": 0,
                "fp": 0,
            }

            cascade_prevention = 0.0

        else:
            # Create configuration with quarantine enabled
            config = SimulationConfig(
                num_agents=swarm_size,
                fault_rate=FAULT_RATE,
                run_time=RUN_TIME,
                scaling_mode=ScalingMode.FIXED_ARENA,
                enable_quarantine=True,
                random_seed=seed,
            )

            # Generate baseline reference for cascade prevention computation
            baseline_config = SimulationConfig(
                num_agents=swarm_size,
                fault_rate=FAULT_RATE,
                run_time=RUN_TIME,
                scaling_mode=ScalingMode.FIXED_ARENA,
                enable_quarantine=False,
                random_seed=seed,
            )

            baseline_sim = BaselineSimulator(baseline_config)
            await baseline_sim.run_simulation()
            baseline_tracker = baseline_sim.metrics_tracker

            # Run the detection method
            if method == "Voting":
                simulator = SimpleVotingSimulator(config)
                metrics = await simulator.run_simulation(baseline_tracker)
                detection_metrics = simulator.get_detection_metrics()

            elif method == "Hormone":
                simulator = LocalHormoneSimulator(config)
                metrics = await simulator.run_simulation(baseline_tracker)
                detection_metrics = simulator.get_quarantine_event_metrics()

            elif method == "Threshold":
                simulator = ThresholdDetector(config)
                metrics = await simulator.run_simulation(baseline_tracker)
                detection_metrics = simulator.get_detection_metrics()

            else:
                raise ValueError(f"Unknown detection method: {method}")

            cascade_prevention = metrics.cascade_prevention

        simulation_time = time.time() - start_time

        # Extract detection performance
        precision = detection_metrics.get("precision", 0.0)
        recall = detection_metrics.get("recall", 0.0)

        # Compute false positive rate
        false_positives = detection_metrics.get("fp", 0)
        total_healthy = metrics.num_healthy
        false_positive_rate = (
            false_positives / total_healthy if total_healthy > 0 else 0.0
        )

        return ScalabilityMetrics(
            swarm_size=swarm_size,
            method=method,
            seed=seed,
            task_completion=metrics.task_completion,
            precision=precision,
            recall=recall,
            cascade_prevention=cascade_prevention,
            response_time=metrics.avg_response_time,
            healthy_preserved=metrics.healthy_preservation_rate,
            quarantine_efficiency=metrics.quarantine_efficiency,
            false_positive_rate=false_positive_rate,
            agents_saved=getattr(metrics, "agents_saved", 0),
            simulation_time=simulation_time,
            density=density,
        )

    except Exception as e:
        print(f"    Error: {str(e)}")
        return None


def aggregate_experimental_results(
        swarm_size: int, method: str, results: List[ScalabilityMetrics]
) -> AggregatedMetrics:
    """Aggregate multiple experimental runs into statistical summary.

    Parameters
    ----------
    swarm_size : int
        Number of agents tested.
    method : str
        Detection method name.
    results : list of ScalabilityMetrics
        Individual run results to aggregate.

    Returns
    -------
    aggregated : AggregatedMetrics
        Statistical summary with means and confidence intervals.

    Notes
    -----
    Invalid or infinite values (e.g., response times for baseline) are
    filtered before computing statistics.
    """
    if not results:
        return AggregatedMetrics(
            swarm_size=swarm_size,
            method=method,
            task_completion_mean=0.0,
            task_completion_ci=(0.0, 0.0),
            precision_mean=0.0,
            precision_ci=(0.0, 0.0),
            recall_mean=0.0,
            recall_ci=(0.0, 0.0),
            cascade_prevention_mean=0.0,
            cascade_prevention_ci=(0.0, 0.0),
            response_time_mean=float("inf"),
            response_time_ci=(float("inf"), float("inf")),
            success_rate=0.0,
            runtime_mean=0.0,
            raw_results=[],
        )

    # Extract metric values
    task_completions = [r.task_completion for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    cascade_preventions = [r.cascade_prevention for r in results]
    response_times = [r.response_time for r in results if r.response_time < float("inf")]
    runtimes = [r.simulation_time for r in results]

    # Compute statistics
    return AggregatedMetrics(
        swarm_size=swarm_size,
        method=method,
        task_completion_mean=np.mean(task_completions),
        task_completion_ci=compute_confidence_interval(task_completions),
        precision_mean=np.mean(precisions),
        precision_ci=compute_confidence_interval(precisions),
        recall_mean=np.mean(recalls),
        recall_ci=compute_confidence_interval(recalls),
        cascade_prevention_mean=np.mean(cascade_preventions),
        cascade_prevention_ci=compute_confidence_interval(cascade_preventions),
        response_time_mean=np.mean(response_times) if response_times else float("inf"),
        response_time_ci=(
            compute_confidence_interval(response_times)
            if response_times
            else (float("inf"), float("inf"))
        ),
        success_rate=len(results) / NUM_SEEDS,
        runtime_mean=np.mean(runtimes),
        raw_results=results,
    )


async def run_scalability_suite() -> Dict[str, List[AggregatedMetrics]]:
    """Execute the complete scalability experiment suite.

    Returns
    -------
    results : dict of str to list of AggregatedMetrics
        Results for each method across all swarm sizes.

    Notes
    -----
    Runs all combinations of swarm sizes, methods, and random seeds.
    Progress is reported to stdout during execution.
    """
    print("\n" + "=" * 80)
    print("SCALABILITY EXPERIMENT SUITE")
    print("Evaluating fault detection methods across swarm sizes 30-120")
    print("=" * 80)

    methods = ["Baseline", "Threshold", "Voting", "Hormone"]
    all_results = {method: [] for method in methods}

    for swarm_size in SWARM_SIZES:
        print(f"\n{'─' * 80}")
        print(f"Swarm size: {swarm_size} agents")
        print(f"Density: {swarm_size / (100 * 100):.4f} agents/m²")
        print(f"{'─' * 80}")

        for method in methods:
            print(f"\n  {method} method:")
            method_results = []

            for seed in range(NUM_SEEDS):
                print(f"    Run {seed + 1}/{NUM_SEEDS} ... ", end="", flush=True)

                result = await run_single_experiment(swarm_size, method, seed)

                if result is not None:
                    method_results.append(result)
                    print(f"✓ TCR={result.task_completion:.3f}")
                else:
                    print("✗ Failed")

            if method_results:
                aggregated = aggregate_experimental_results(
                    swarm_size, method, method_results
                )
                all_results[method].append(aggregated)

                # Print summary statistics
                tcr_mean = aggregated.task_completion_mean
                tcr_margin = aggregated.task_completion_ci[1] - tcr_mean
                print(f"    → Mean TCR = {tcr_mean:.3f} ± {tcr_margin:.3f}")
            else:
                print(f"    → All runs failed")

    return all_results


def print_results_table(results: Dict[str, List[AggregatedMetrics]]) -> None:
    """Print formatted table of experimental results.

    Parameters
    ----------
    results : dict of str to list of AggregatedMetrics
        Results for each method across all swarm sizes.
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Table header
    print(
        f"\n{'Size':>5} │ {'Method':>9} │ {'TCR':>8} │ {'±CI':>6} │ "
        f"{'Prec':>6} │ {'Recall':>6} │ {'Cascade↓':>9} │ {'Response':>9}"
    )
    print("─" * 90)

    for size in SWARM_SIZES:
        for method in ["Baseline", "Threshold", "Voting", "Hormone"]:
            method_results = results.get(method, [])
            size_result = next((r for r in method_results if r.swarm_size == size), None)

            if size_result:
                tcr = size_result.task_completion_mean
                tcr_margin = size_result.task_completion_ci[1] - tcr
                prec = size_result.precision_mean
                recall = size_result.recall_mean
                cascade = size_result.cascade_prevention_mean
                response = size_result.response_time_mean

                response_str = f"{response:.2f}s" if response < float("inf") else "∞"

                print(
                    f"{size:>5} │ {method:>9} │ {tcr:>8.3f} │ {tcr_margin:>6.3f} │ "
                    f"{prec:>6.2f} │ {recall:>6.2f} │ {cascade:>9.2f} │ {response_str:>9}"
                )
            else:
                print(f"{size:>5} │ {method:>9} │ {'FAILED':>8} │ {'':>6} │ {'':>6} │ {'':>6} │ {'':>9} │ {'':>9}")


def analyze_scaling_trends(results: Dict[str, List[AggregatedMetrics]]) -> None:
    """Analyze and print scaling behavior trends.

    Parameters
    ----------
    results : dict of str to list of AggregatedMetrics
        Results for each method across all swarm sizes.
    """
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    print("\n1. Task Completion Rate Trends:")
    for method in ["Baseline", "Threshold", "Voting", "Hormone"]:
        method_results = results.get(method, [])

        if len(method_results) >= 2:
            first_tcr = method_results[0].task_completion_mean
            last_tcr = method_results[-1].task_completion_mean
            delta = last_tcr - first_tcr
            percent_change = (delta / first_tcr * 100) if first_tcr > 0 else 0

            print(
                f"   {method:>9}: {first_tcr:.3f} → {last_tcr:.3f} "
                f"({delta:+.3f}, {percent_change:+.1f}%)"
            )
        else:
            print(f"   {method:>9}: Insufficient data")

    print("\n2. Performance at Maximum Scale (120 agents):")
    for method in ["Baseline", "Threshold", "Voting", "Hormone"]:
        method_results = results.get(method, [])
        large_result = next((r for r in method_results if r.swarm_size == 120), None)

        if large_result:
            print(
                f"   {method:>9}: TCR={large_result.task_completion_mean:.3f}, "
                f"Precision={large_result.precision_mean:.2f}, "
                f"Recall={large_result.recall_mean:.2f}"
            )
        else:
            print(f"   {method:>9}: No data available")

    print("\n3. Statistical Confidence:")
    for method in ["Baseline", "Threshold", "Voting", "Hormone"]:
        method_results = results.get(method, [])

        if method_results:
            avg_margin = np.mean(
                [r.task_completion_ci[1] - r.task_completion_mean for r in method_results]
            )
            avg_success = np.mean([r.success_rate for r in method_results])
            print(
                f"   {method:>9}: Average CI margin=±{avg_margin:.3f}, "
                f"Success rate={avg_success:.0%}"
            )


def export_to_csv(
        results: Dict[str, List[AggregatedMetrics]], filename: str = "scalability_results.csv"
) -> None:
    """Export experimental results to CSV format.

    Parameters
    ----------
    results : dict of str to list of AggregatedMetrics
        Results to export.
    filename : str, optional
        Output filename (default: 'scalability_results.csv').

    Notes
    -----
    CSV is saved to the 'results' directory in the project root.
    """
    import csv

    # Ensure results directory exists
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename

    try:
        with open(filepath, "w", newline="") as csvfile:
            fieldnames = [
                "swarm_size",
                "method",
                "task_completion_mean",
                "task_completion_ci_lower",
                "task_completion_ci_upper",
                "precision_mean",
                "recall_mean",
                "cascade_prevention_mean",
                "response_time_mean",
                "success_rate",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for method, results_list in results.items():
                for result in results_list:
                    writer.writerow(
                        {
                            "swarm_size": result.swarm_size,
                            "method": method,
                            "task_completion_mean": result.task_completion_mean,
                            "task_completion_ci_lower": result.task_completion_ci[0],
                            "task_completion_ci_upper": result.task_completion_ci[1],
                            "precision_mean": result.precision_mean,
                            "recall_mean": result.recall_mean,
                            "cascade_prevention_mean": result.cascade_prevention_mean,
                            "response_time_mean": result.response_time_mean,
                            "success_rate": result.success_rate,
                        }
                    )

        print(f"\nResults exported to {filepath}")

    except Exception as e:
        print(f"\nCSV export failed: {e}")


async def main():
    """Execute the scalability experiment and generate results."""
    print("Scalability Experiment")
    print(f"Configuration: {len(SWARM_SIZES)} swarm sizes × 4 methods × {NUM_SEEDS} seeds")
    print(f"Total runs: {len(SWARM_SIZES) * 4 * NUM_SEEDS}")

    start_time = time.time()

    # Run experiments
    results = await run_scalability_suite()

    # Analyze and report
    print_results_table(results)
    analyze_scaling_trends(results)

    # Export data
    export_to_csv(results)

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.1f} seconds")
    print("Experiment complete.\n")

    return results


if __name__ == "__main__":
    # Ensure scipy is available for confidence intervals
    try:
        import scipy.stats
    except ImportError:
        print("Installing scipy for statistical analysis...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], check=True)

    asyncio.run(main())