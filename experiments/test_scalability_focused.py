"""Scalability analysis with variable seed allocation.

This module implements a focused scalability experiment that allocates
computational resources efficiently based on where performance differences
are most pronounced.

Seed Allocation Strategy
------------------------
- Sizes 30-70: 3 seeds (performance convergence)
- Sizes 80-120: 10 seeds (maximum separation)

This approach follows reviewer feedback to concentrate statistical power
where it matters most while reducing unnecessary computation.

Usage
-----
From the experiments/ directory:
    $ python test_scalability_focused.py
"""

import asyncio
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import SimulationConfig, ScalingMode
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator
from src.methods.threshold import ThresholdDetector
from src.methods.voting_simplified import SimpleVotingSimulator


# Experiment configuration
SWARM_SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
FAULT_RATE = 0.25
RUN_TIME = 20.0
ARENA_SIZE = 100.0


def compute_num_seeds(swarm_size: int) -> int:
    """Determine number of seeds based on swarm size.

    Parameters
    ----------
    swarm_size : int
        Number of agents in swarm.

    Returns
    -------
    int
        Number of random seeds to use.

    Notes
    -----
    Allocation strategy prioritizes computational resources where
    method separation is largest (sizes 80-120).
    """
    return 10 if swarm_size >= 80 else 3


def compute_standard_error(values: List[float]) -> float:
    """Compute standard error of the mean.

    Parameters
    ----------
    values : list of float
        Sample values.

    Returns
    -------
    float
        Standard error (SEM = std / sqrt(n)).

    Notes
    -----
    Returns 0.0 for insufficient sample size (n < 2).
    """
    if not values or len(values) < 2:
        return 0.0

    clean_values = [v for v in values if v is not None and not np.isnan(v)]
    if len(clean_values) < 2:
        return 0.0

    return np.std(clean_values, ddof=1) / np.sqrt(len(clean_values))


@dataclass
class ScalabilityResult:
    """Results from a single scalability evaluation.

    Attributes
    ----------
    swarm_size : int
        Number of agents.
    method : str
        Detection method identifier.
    seed : int
        Random seed used.
    task_completion : float
        Task completion rate.
    precision : float
        Detection precision.
    recall : float
        Detection recall.
    cascade_prevention : float
        Cascade prevention score.
    response_time : float
        Average detection response time (seconds).
    healthy_preserved : float
        Proportion of healthy agents preserved.
    quarantine_efficiency : float
        Quarantine efficiency score.
    false_positive_rate : float
        Rate of false positive detections.
    simulation_time : float
        Wall-clock execution time (seconds).
    density : float
        Agent density (agents/m^2).
    """
    swarm_size: int
    method: str
    seed: int

    # Performance metrics
    task_completion: float
    precision: float
    recall: float
    cascade_prevention: float
    response_time: float

    # Additional metrics
    healthy_preserved: float
    quarantine_efficiency: float
    false_positive_rate: float

    # Runtime information
    simulation_time: float
    density: float


@dataclass
class AggregatedResults:
    """Aggregated statistics across multiple seeds.

    Attributes
    ----------
    swarm_size : int
        Number of agents.
    method : str
        Detection method identifier.
    task_completion_mean : float
        Mean task completion rate.
    task_completion_se : float
        Standard error of task completion.
    precision_mean : float
        Mean precision.
    precision_se : float
        Standard error of precision.
    recall_mean : float
        Mean recall.
    recall_se : float
        Standard error of recall.
    cascade_prevention_mean : float
        Mean cascade prevention score.
    cascade_prevention_se : float
        Standard error of cascade prevention.
    response_time_mean : float
        Mean response time (seconds).
    response_time_se : float
        Standard error of response time.
    success_rate : float
        Proportion of successful runs.
    runtime_mean : float
        Mean wall-clock runtime (seconds).
    n_runs : int
        Number of successful runs.
    raw_results : list of ScalabilityResult
        Individual run results.
    """
    swarm_size: int
    method: str

    # Primary metrics with standard errors
    task_completion_mean: float
    task_completion_se: float
    precision_mean: float
    precision_se: float
    recall_mean: float
    recall_se: float
    cascade_prevention_mean: float
    cascade_prevention_se: float
    response_time_mean: float
    response_time_se: float

    # Meta-information
    success_rate: float
    runtime_mean: float
    n_runs: int

    raw_results: List[ScalabilityResult] = field(default_factory=list)


class ScalabilityExperiment:
    """Orchestrates focused scalability experiments.

    This class manages the execution, aggregation, and export of
    scalability experiments with variable seed allocation.

    Parameters
    ----------
    output_dir : Path
        Directory for results export.

    Attributes
    ----------
    methods : list of str
        Detection methods to evaluate.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.methods = ["Baseline", "Threshold", "Voting", "Hormone"]

    async def evaluate_single_configuration(
        self,
        swarm_size: int,
        method: str,
        seed: int
    ) -> Optional[ScalabilityResult]:
        """Evaluate a single swarm size, method, seed combination.

        Parameters
        ----------
        swarm_size : int
            Number of agents in swarm.
        method : str
            Detection method identifier.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        ScalabilityResult or None
            Results if successful, None if evaluation failed.
        """
        try:
            density = swarm_size / (ARENA_SIZE * ARENA_SIZE)
            start_time = time.time()

            if method == "Baseline":
                result = await self._evaluate_baseline(swarm_size, seed, density)
            else:
                result = await self._evaluate_detection_method(
                    swarm_size, method, seed, density
                )

            result.simulation_time = time.time() - start_time
            return result

        except Exception as e:
            print(f"  [FAIL] Error: {method} (size={swarm_size}, seed={seed}): {e}")
            return None

    async def _evaluate_baseline(
        self,
        swarm_size: int,
        seed: int,
        density: float
    ) -> ScalabilityResult:
        """Evaluate baseline method (no detection)."""
        config = SimulationConfig(
            num_agents=swarm_size,
            fault_rate=FAULT_RATE,
            run_time=RUN_TIME,
            scaling_mode=ScalingMode.FIXED_ARENA,
            enable_quarantine=False,
            random_seed=seed
        )

        simulator = BaselineSimulator(config)
        metrics = await simulator.run_simulation()

        return ScalabilityResult(
            swarm_size=swarm_size,
            method="Baseline",
            seed=seed,
            task_completion=metrics.task_completion,
            precision=0.0,
            recall=0.0,
            cascade_prevention=0.0,
            response_time=float('inf'),
            healthy_preserved=metrics.healthy_preservation_rate,
            quarantine_efficiency=metrics.quarantine_efficiency,
            false_positive_rate=0.0,
            simulation_time=0.0,  # Set by caller
            density=density
        )

    async def _evaluate_detection_method(
        self,
        swarm_size: int,
        method: str,
        seed: int,
        density: float
    ) -> ScalabilityResult:
        """Evaluate detection method with baseline comparison."""
        # Configure detection method
        config = SimulationConfig(
            num_agents=swarm_size,
            fault_rate=FAULT_RATE,
            run_time=RUN_TIME,
            scaling_mode=ScalingMode.FIXED_ARENA,
            enable_quarantine=True,
            random_seed=seed
        )

        # Run baseline for reference
        baseline_config = SimulationConfig(
            num_agents=swarm_size,
            fault_rate=FAULT_RATE,
            run_time=RUN_TIME,
            scaling_mode=ScalingMode.FIXED_ARENA,
            enable_quarantine=False,
            random_seed=seed
        )

        baseline_sim = BaselineSimulator(baseline_config)
        await baseline_sim.run_simulation()
        baseline_tracker = baseline_sim.metrics_tracker

        # Run detection method
        simulator = self._create_simulator(method, config)

        if method == "Hormone":
            metrics = await simulator.run_simulation(baseline_metrics=baseline_tracker)
            detection_metrics = simulator.get_quarantine_event_metrics()
        else:
            metrics = await simulator.run_simulation(baseline_tracker)
            detection_metrics = simulator.get_detection_metrics()

        # Extract metrics
        precision = detection_metrics.get('precision', 0.0)
        recall = detection_metrics.get('recall', 0.0)
        fp = detection_metrics.get('fp', 0)

        total_healthy = metrics.num_healthy
        false_positive_rate = fp / total_healthy if total_healthy > 0 else 0.0

        return ScalabilityResult(
            swarm_size=swarm_size,
            method=method,
            seed=seed,
            task_completion=metrics.task_completion,
            precision=precision,
            recall=recall,
            cascade_prevention=metrics.cascade_prevention,
            response_time=metrics.avg_response_time,
            healthy_preserved=metrics.healthy_preservation_rate,
            quarantine_efficiency=metrics.quarantine_efficiency,
            false_positive_rate=false_positive_rate,
            simulation_time=0.0,  # Set by caller
            density=density
        )

    def _create_simulator(self, method: str, config: SimulationConfig):
        """Factory method for simulator instantiation.

        Parameters
        ----------
        method : str
            Method identifier.
        config : SimulationConfig
            Simulation configuration.

        Returns
        -------
        Simulator
            Configured simulator instance.

        Raises
        ------
        ValueError
            If method is not recognized.
        """
        simulators = {
            'Voting': SimpleVotingSimulator,
            'Hormone': LocalHormoneSimulator,
            'Threshold': ThresholdDetector
        }

        if method not in simulators:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid options: {list(simulators.keys())}"
            )

        return simulators[method](config)

    async def run_full_experiment(self) -> Dict[str, List[AggregatedResults]]:
        """Execute complete scalability experiment.

        Returns
        -------
        dict
            Aggregated results for each method across all swarm sizes.
        """
        print("\n" + "=" * 70)
        print("Focused Scalability Experiment")
        print("=" * 70)
        print("\nVariable seed allocation:")
        print("  - Sizes 30-70:  3 seeds")
        print("  - Sizes 80-120: 10 seeds")

        all_results = {method: [] for method in self.methods}

        total_runs = sum(
            compute_num_seeds(size) * len(self.methods)
            for size in SWARM_SIZES
        )
        completed_runs = 0

        for swarm_size in SWARM_SIZES:
            num_seeds = compute_num_seeds(swarm_size)
            density = swarm_size / (ARENA_SIZE * ARENA_SIZE)

            print(f"\n{'-' * 70}")
            print(f"Swarm size: {swarm_size:3d} agents ({num_seeds} seeds)")
            print(f"Density:    {density:.4f} agents/m^2")
            print(f"{'-' * 70}")

            for method in self.methods:
                method_results = []

                for seed in range(num_seeds):
                    result = await self.evaluate_single_configuration(
                        swarm_size, method, seed
                    )

                    if result is not None:
                        method_results.append(result)

                    completed_runs += 1
                    progress = (completed_runs / total_runs) * 100

                    if result is not None:
                        print(
                            f"  {method:9s} seed {seed+1:2d}/{num_seeds:2d}: "
                            f"TCR={result.task_completion:.3f} "
                            f"[{completed_runs:3d}/{total_runs:3d}, {progress:5.1f}%]"
                        )

                # Aggregate results
                if method_results:
                    aggregated = self._aggregate_results(
                        swarm_size, method, method_results
                    )
                    all_results[method].append(aggregated)

        return all_results

    def _aggregate_results(
        self,
        swarm_size: int,
        method: str,
        results: List[ScalabilityResult]
    ) -> AggregatedResults:
        """Aggregate results across seeds.

        Parameters
        ----------
        swarm_size : int
            Number of agents.
        method : str
            Method identifier.
        results : list of ScalabilityResult
            Individual run results.

        Returns
        -------
        AggregatedResults
            Aggregated statistics.
        """
        if not results:
            return self._create_empty_aggregation(swarm_size, method)

        # Extract metric arrays
        task_completions = [r.task_completion for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        cascade_preventions = [r.cascade_prevention for r in results]
        response_times = [
            r.response_time for r in results
            if r.response_time < float('inf')
        ]
        runtimes = [r.simulation_time for r in results]

        return AggregatedResults(
            swarm_size=swarm_size,
            method=method,
            task_completion_mean=np.mean(task_completions),
            task_completion_se=compute_standard_error(task_completions),
            precision_mean=np.mean(precisions),
            precision_se=compute_standard_error(precisions),
            recall_mean=np.mean(recalls),
            recall_se=compute_standard_error(recalls),
            cascade_prevention_mean=np.mean(cascade_preventions),
            cascade_prevention_se=compute_standard_error(cascade_preventions),
            response_time_mean=np.mean(response_times) if response_times else float('inf'),
            response_time_se=compute_standard_error(response_times) if response_times else 0.0,
            success_rate=len(results) / compute_num_seeds(swarm_size),
            runtime_mean=np.mean(runtimes),
            n_runs=len(results),
            raw_results=results
        )

    @staticmethod
    def _create_empty_aggregation(swarm_size: int, method: str) -> AggregatedResults:
        """Create empty aggregation for failed runs."""
        return AggregatedResults(
            swarm_size=swarm_size,
            method=method,
            task_completion_mean=0.0,
            task_completion_se=0.0,
            precision_mean=0.0,
            precision_se=0.0,
            recall_mean=0.0,
            recall_se=0.0,
            cascade_prevention_mean=0.0,
            cascade_prevention_se=0.0,
            response_time_mean=float('inf'),
            response_time_se=0.0,
            success_rate=0.0,
            runtime_mean=0.0,
            n_runs=0,
            raw_results=[]
        )

    def print_summary(self, all_results: Dict[str, List[AggregatedResults]]):
        """Print formatted summary table.

        Parameters
        ----------
        all_results : dict
            Results dictionary from run_full_experiment.
        """
        print("\n" + "=" * 70)
        print("Scalability Results Summary (mean +/- SE)")
        print("=" * 70)

        # Header
        print(
            f"\n{'Size':>4} | {'Method':>9} | {'N':>2} | "
            f"{'TCR':>8} | {'+/- SE':>6} | "
            f"{'Prec':>6} | {'Recall':>6} | "
            f"{'CCP':>8} | {'Response':>8}"
        )
        print("-" * 95)

        # Data rows
        for size in SWARM_SIZES:
            for method in self.methods:
                method_results = all_results.get(method, [])
                size_result = next(
                    (r for r in method_results if r.swarm_size == size),
                    None
                )

                if size_result:
                    response_str = (
                        f"{size_result.response_time_mean:.1f}s"
                        if size_result.response_time_mean < float('inf')
                        else "inf"
                    )

                    print(
                        f"{size:>4} | {method:>9} | {size_result.n_runs:>2} | "
                        f"{size_result.task_completion_mean:>7.3f} | "
                        f"{size_result.task_completion_se:>5.3f} | "
                        f"{size_result.precision_mean:>5.2f} | "
                        f"{size_result.recall_mean:>6.2f} | "
                        f"{size_result.cascade_prevention_mean:>7.2f} | "
                        f"{response_str:>8}"
                    )

    def export_csv(
        self,
        all_results: Dict[str, List[AggregatedResults]],
        filename: str = "scalability_focused.csv"
    ) -> Path:
        """Export results to CSV format.

        Parameters
        ----------
        all_results : dict
            Results dictionary from run_full_experiment.
        filename : str, optional
            Output filename.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        filepath = self.output_dir / filename

        fieldnames = [
            'swarm_size', 'method', 'n_runs',
            'task_completion_mean', 'task_completion_se',
            'precision_mean', 'precision_se',
            'recall_mean', 'recall_se',
            'cascade_prevention_mean', 'cascade_prevention_se',
            'response_time_mean', 'response_time_se'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for method, results_list in all_results.items():
                for result in results_list:
                    writer.writerow({
                        'swarm_size': result.swarm_size,
                        'method': method,
                        'n_runs': result.n_runs,
                        'task_completion_mean': result.task_completion_mean,
                        'task_completion_se': result.task_completion_se,
                        'precision_mean': result.precision_mean,
                        'precision_se': result.precision_se,
                        'recall_mean': result.recall_mean,
                        'recall_se': result.recall_se,
                        'cascade_prevention_mean': result.cascade_prevention_mean,
                        'cascade_prevention_se': result.cascade_prevention_se,
                        'response_time_mean': result.response_time_mean,
                        'response_time_se': result.response_time_se
                    })

        return filepath


async def main():
    """Execute focused scalability experiment."""
    # Setup
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    experiment = ScalabilityExperiment(results_dir)

    # Estimate runtime
    num_runs_30_70 = 5 * len(experiment.methods) * 3
    num_runs_80_120 = 5 * len(experiment.methods) * 10
    total_runs = num_runs_30_70 + num_runs_80_120

    print("\n" + "=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"\nTotal runs:     {total_runs}")
    print(f"Estimated time: {total_runs * 0.5:.0f}-{total_runs * 1:.0f} minutes")

    # Execute experiment
    start_time = time.time()
    all_results = await experiment.run_full_experiment()
    elapsed_time = time.time() - start_time

    # Present results
    experiment.print_summary(all_results)

    # Export results
    csv_path = experiment.export_csv(all_results)

    print(f"\n{'=' * 70}")
    print("Experiment Complete")
    print(f"{'=' * 70}")
    print(f"\nElapsed time: {elapsed_time / 60:.1f} minutes")
    print(f"Results exported to: {csv_path}\n")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())