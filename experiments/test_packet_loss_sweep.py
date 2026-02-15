"""Packet loss sweep for communication degradation analysis.

This module evaluates the resilience of the hormone-based detection method
to communication failures. By systematically varying packet loss rates,
we quantify the relative importance of internal vs external hormone signals.

Loss Rate Schedule
------------------
0%, 30%, 50%, 70%, 90%, 100%

The experiment reveals whether the hybrid architecture (internal + external
hormones) provides graceful degradation under communication stress.

Usage
-----
From the experiments/ directory:
    $ python test_packet_loss_sweep.py

Notes
-----
Baseline method performance is unaffected by packet loss as it requires
no inter-agent communication. This serves as a communication-independent
reference point.

"""

import asyncio
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import SimulationConfig, ScalingMode
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator

# Experiment configuration
LOSS_RATES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
NUM_SEEDS = 10
NUM_AGENTS = 120  # Large swarm for clear effects
FAULT_RATE = 0.3
RUN_TIME = 20.0


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
class PacketLossResult:
    """Results from a single packet loss evaluation.

    Attributes
    ----------
    loss_rate : float
        Packet loss rate (0.0 to 1.0).
    seed : int
        Random seed used.
    task_completion : float
        Task completion rate.
    cascade_prevention : float
        Cascade prevention score.
    precision : float
        Detection precision.
    recall : float
        Detection recall.
    response_time : float
        Average detection response time (seconds).
    quarantine_efficiency : float
        Quarantine efficiency score.
    avg_internal_hormone : float
        Mean internal hormone level for quarantined agents.
    avg_external_hormone : float
        Mean external hormone level for quarantined agents.
    external_fraction : float
        Proportion of hormone signal from external sources (0.0 to 1.0).
    simulation_time : float
        Wall-clock execution time (seconds).
    """
    loss_rate: float
    seed: int

    # Primary metrics
    task_completion: float
    cascade_prevention: float
    precision: float

    # Detection metrics
    recall: float
    response_time: float
    quarantine_efficiency: float

    # Hormone breakdown
    avg_internal_hormone: float
    avg_external_hormone: float
    external_fraction: float

    # Runtime information
    simulation_time: float


@dataclass
class AggregatedPacketLossResults:
    """Aggregated statistics for a specific loss rate.

    Attributes
    ----------
    loss_rate : float
        Packet loss rate.
    n_runs : int
        Number of successful runs.
    tc_mean : float
        Mean task completion rate.
    tc_se : float
        Standard error of task completion.
    ccp_mean : float
        Mean cascade prevention score.
    ccp_se : float
        Standard error of cascade prevention.
    precision_mean : float
        Mean detection precision.
    precision_se : float
        Standard error of precision.
    recall_mean : float
        Mean detection recall.
    external_fraction_mean : float
        Mean proportion of external hormone signal.
    external_fraction_se : float
        Standard error of external fraction.
    raw_results : list of PacketLossResult
        Individual run results.
    """
    loss_rate: float
    n_runs: int

    # Primary metrics with standard errors
    tc_mean: float
    tc_se: float
    ccp_mean: float
    ccp_se: float
    precision_mean: float
    precision_se: float

    # Additional metrics
    recall_mean: float
    external_fraction_mean: float
    external_fraction_se: float

    raw_results: List[PacketLossResult]


class PacketLossSweep:
    """Orchestrates packet loss sweep experiments.

    This class manages systematic evaluation of communication degradation
    effects on hormone-based detection.

    Parameters
    ----------
    output_dir : Path
        Directory for results export.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_single_configuration(
            self,
            loss_rate: float,
            seed: int
    ) -> Optional[PacketLossResult]:
        """Evaluate hormone method at specific packet loss rate.

        Parameters
        ----------
        loss_rate : float
            Packet loss rate (0.0 to 1.0).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        PacketLossResult or None
            Results if successful, None if evaluation failed.
        """
        try:
            # Run baseline for reference (no packet loss effect)
            baseline_config = SimulationConfig(
                num_agents=NUM_AGENTS,
                fault_rate=FAULT_RATE,
                run_time=RUN_TIME,
                enable_quarantine=False,
                packet_loss_rate=0.0,  # Baseline unaffected
                random_seed=seed,
                scaling_mode=ScalingMode.FIXED_ARENA
            )

            baseline_sim = BaselineSimulator(baseline_config)
            await baseline_sim.run_simulation()
            baseline_tracker = baseline_sim.metrics_tracker

            # Run hormone method with specified packet loss
            hormone_config = SimulationConfig(
                num_agents=NUM_AGENTS,
                fault_rate=FAULT_RATE,
                run_time=RUN_TIME,
                enable_quarantine=True,
                packet_loss_rate=loss_rate,
                random_seed=seed,
                scaling_mode=ScalingMode.FIXED_ARENA
            )

            start_time = time.time()
            simulator = LocalHormoneSimulator(hormone_config)
            metrics = await simulator.run_simulation(baseline_metrics=baseline_tracker)
            simulation_time = time.time() - start_time

            # Extract detection metrics
            detection_metrics = simulator.get_quarantine_event_metrics()

            # Compute hormone breakdown
            hormone_breakdown = self._compute_hormone_breakdown(simulator)

            return PacketLossResult(
                loss_rate=loss_rate,
                seed=seed,
                task_completion=metrics.task_completion,
                cascade_prevention=metrics.cascade_prevention,
                precision=detection_metrics.get('precision', 0.0),
                recall=detection_metrics.get('recall', 0.0),
                response_time=metrics.avg_response_time,
                quarantine_efficiency=metrics.quarantine_efficiency,
                avg_internal_hormone=hormone_breakdown['avg_internal'],
                avg_external_hormone=hormone_breakdown['avg_external'],
                external_fraction=hormone_breakdown['external_fraction'],
                simulation_time=simulation_time
            )

        except Exception as e:
            print(f"  [FAIL] Error at loss={loss_rate:.0%}, seed={seed}: {e}")
            return None

    @staticmethod
    def _compute_hormone_breakdown(simulator) -> Dict[str, float]:
        """Compute internal vs external hormone contributions.

        Parameters
        ----------
        simulator : LocalHormoneSimulator
            Completed simulator instance.

        Returns
        -------
        dict
            Hormone breakdown statistics.
        """
        quarantined_agents = [a for a in simulator.agents if a.is_quarantined]

        if not quarantined_agents:
            return {
                'avg_internal': 0.0,
                'avg_external': 0.0,
                'external_fraction': 0.0
            }

        internal_hormones = [a.internal_hormone for a in quarantined_agents]
        external_hormones = [a.external_hormone for a in quarantined_agents]

        avg_internal = np.mean(internal_hormones)
        avg_external = np.mean(external_hormones)

        total_hormone = avg_internal + avg_external
        external_fraction = (
            avg_external / total_hormone if total_hormone > 0 else 0.0
        )

        return {
            'avg_internal': avg_internal,
            'avg_external': avg_external,
            'external_fraction': external_fraction
        }

    async def run_full_sweep(self) -> Dict[float, AggregatedPacketLossResults]:
        """Execute complete packet loss sweep.

        Returns
        -------
        dict
            Aggregated results for each loss rate.
        """
        print("\n" + "=" * 70)
        print("Packet Loss Sweep Experiment")
        print("=" * 70)
        print("\nTesting communication degradation:")
        print(f"  Loss rates: {[f'{r:.0%}' for r in LOSS_RATES]}")
        print(f"  Seeds per rate: {NUM_SEEDS}")

        all_results = {}
        total_runs = len(LOSS_RATES) * NUM_SEEDS
        completed_runs = 0

        for loss_rate in LOSS_RATES:
            print(f"\n{'-' * 70}")
            print(f"Loss rate: {loss_rate:4.0%} ({NUM_SEEDS} seeds)")
            print(f"{'-' * 70}")

            run_results = []

            for seed in range(NUM_SEEDS):
                result = await self.evaluate_single_configuration(loss_rate, seed)

                if result is not None:
                    run_results.append(result)
                    print(
                        f"  Seed {seed + 1:2d}/{NUM_SEEDS:2d}: "
                        f"TC={result.task_completion:.3f}, "
                        f"CCP={result.cascade_prevention:.3f}, "
                        f"Ext={result.external_fraction:.1%}"
                    )

                completed_runs += 1
                progress = (completed_runs / total_runs) * 100
                print(f"  Progress: {completed_runs}/{total_runs} ({progress:.1f}%)")

            # Aggregate results
            if run_results:
                aggregated = self._aggregate_results(loss_rate, run_results)
                all_results[loss_rate] = aggregated

                print(f"\n  Summary:")
                print(f"    TC:  {aggregated.tc_mean:.3f} +/- {aggregated.tc_se:.3f}")
                print(f"    CCP: {aggregated.ccp_mean:.3f} +/- {aggregated.ccp_se:.3f}")
                print(f"    Precision: {aggregated.precision_mean:.3f} +/- {aggregated.precision_se:.3f}")
                print(f"    External signal: {aggregated.external_fraction_mean:.1%}")

        return all_results

    def _aggregate_results(
            self,
            loss_rate: float,
            results: List[PacketLossResult]
    ) -> AggregatedPacketLossResults:
        """Aggregate results for a specific loss rate.

        Parameters
        ----------
        loss_rate : float
            Packet loss rate.
        results : list of PacketLossResult
            Individual run results.

        Returns
        -------
        AggregatedPacketLossResults
            Aggregated statistics.
        """
        if not results:
            return self._create_empty_aggregation(loss_rate)

        # Extract metric arrays
        tcs = [r.task_completion for r in results]
        ccps = [r.cascade_prevention for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        external_fractions = [r.external_fraction for r in results]

        return AggregatedPacketLossResults(
            loss_rate=loss_rate,
            n_runs=len(results),
            tc_mean=np.mean(tcs),
            tc_se=compute_standard_error(tcs),
            ccp_mean=np.mean(ccps),
            ccp_se=compute_standard_error(ccps),
            precision_mean=np.mean(precisions),
            precision_se=compute_standard_error(precisions),
            recall_mean=np.mean(recalls),
            external_fraction_mean=np.mean(external_fractions),
            external_fraction_se=compute_standard_error(external_fractions),
            raw_results=results
        )

    @staticmethod
    def _create_empty_aggregation(loss_rate: float) -> AggregatedPacketLossResults:
        """Create empty aggregation for failed runs."""
        return AggregatedPacketLossResults(
            loss_rate=loss_rate,
            n_runs=0,
            tc_mean=0.0,
            tc_se=0.0,
            ccp_mean=0.0,
            ccp_se=0.0,
            precision_mean=0.0,
            precision_se=0.0,
            recall_mean=0.0,
            external_fraction_mean=0.0,
            external_fraction_se=0.0,
            raw_results=[]
        )

    def print_summary(self, all_results: Dict[float, AggregatedPacketLossResults]):
        """Print formatted summary table.

        Parameters
        ----------
        all_results : dict
            Results dictionary from run_full_sweep.
        """
        print("\n" + "=" * 70)
        print("Packet Loss Results Summary")
        print("=" * 70)

        # Header
        print(
            f"\n{'Loss':>6} | {'N':>2} | "
            f"{'TC (mean+/-SE)':>18} | "
            f"{'CCP (mean+/-SE)':>18} | "
            f"{'Prec (mean+/-SE)':>18} | "
            f"{'Ext%':>6}"
        )
        print("-" * 95)

        # Data rows
        for loss_rate in LOSS_RATES:
            result = all_results.get(loss_rate)
            if result:
                print(
                    f"{loss_rate:>5.0%} | {result.n_runs:>2} | "
                    f"{result.tc_mean:>7.3f} +/- {result.tc_se:>5.3f} | "
                    f"{result.ccp_mean:>7.3f} +/- {result.ccp_se:>5.3f} | "
                    f"{result.precision_mean:>7.3f} +/- {result.precision_se:>5.3f} | "
                    f"{result.external_fraction_mean:>5.1%}"
                )

    def print_degradation_analysis(
            self,
            all_results: Dict[float, AggregatedPacketLossResults]
    ):
        """Print degradation analysis.

        Parameters
        ----------
        all_results : dict
            Results dictionary from run_full_sweep.
        """
        print("\n" + "=" * 70)
        print("Degradation Analysis")
        print("=" * 70)

        baseline_result = all_results.get(0.0)
        worst_result = all_results.get(1.0)

        if not (baseline_result and worst_result):
            print("\nInsufficient data for degradation analysis.")
            return

        # Compute degradation
        tc_drop = (
                (baseline_result.tc_mean - worst_result.tc_mean) /
                baseline_result.tc_mean * 100
        )
        ccp_drop = (
                (baseline_result.ccp_mean - worst_result.ccp_mean) /
                baseline_result.ccp_mean * 100
        )
        ext_drop = (
                baseline_result.external_fraction_mean -
                worst_result.external_fraction_mean
        )

        print(f"\nPerformance change (0% -> 100% packet loss):")
        print(f"  Task completion drop:    {tc_drop:>6.1f}%")
        print(f"  Cascade prevention drop: {ccp_drop:>6.1f}%")
        print(f"  External signal drop:    {ext_drop:>6.1%} (absolute)")

        avg_drop = (tc_drop + ccp_drop) / 2
        print(f"\nOverall performance degradation: {avg_drop:.1f}%")

        # Interpretation
        if avg_drop < 10:
            interpretation = (
                "-> Minimal degradation: System highly robust to communication loss"
            )
        elif avg_drop < 25:
            interpretation = (
                "-> Moderate degradation: Hybrid architecture provides resilience"
            )
        else:
            interpretation = (
                "-> Significant degradation: Communication clearly beneficial"
            )

        print(interpretation)

    def export_csv(
            self,
            all_results: Dict[float, AggregatedPacketLossResults],
            filename: str = "packet_loss_sweep.csv"
    ) -> Path:
        """Export results to CSV format.

        Parameters
        ----------
        all_results : dict
            Results dictionary from run_full_sweep.
        filename : str, optional
            Output filename.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        filepath = self.output_dir / filename

        fieldnames = [
            'loss_rate', 'n_runs',
            'tc_mean', 'tc_se',
            'ccp_mean', 'ccp_se',
            'precision_mean', 'precision_se',
            'recall_mean',
            'external_fraction_mean', 'external_fraction_se'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for loss_rate, result in sorted(all_results.items()):
                writer.writerow({
                    'loss_rate': result.loss_rate,
                    'n_runs': result.n_runs,
                    'tc_mean': result.tc_mean,
                    'tc_se': result.tc_se,
                    'ccp_mean': result.ccp_mean,
                    'ccp_se': result.ccp_se,
                    'precision_mean': result.precision_mean,
                    'precision_se': result.precision_se,
                    'recall_mean': result.recall_mean,
                    'external_fraction_mean': result.external_fraction_mean,
                    'external_fraction_se': result.external_fraction_se
                })

        return filepath


async def main():
    """Execute packet loss sweep experiment."""
    # Setup
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    experiment = PacketLossSweep(results_dir)

    # Estimate runtime
    total_runs = len(LOSS_RATES) * NUM_SEEDS

    print("\n" + "=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"\nLoss rates:     {[f'{r:.0%}' for r in LOSS_RATES]}")
    print(f"Seeds per rate: {NUM_SEEDS}")
    print(f"Total runs:     {total_runs}")
    print(f"Estimated time: {total_runs * 0.5:.0f}-{total_runs * 1:.0f} minutes")

    # Execute experiment
    start_time = time.time()
    all_results = await experiment.run_full_sweep()
    elapsed_time = time.time() - start_time

    # Present results
    experiment.print_summary(all_results)
    experiment.print_degradation_analysis(all_results)

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