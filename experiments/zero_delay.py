"""Zero-delay experiment for temporal advantage analysis.

This module tests whether the hormone-based detection method maintains its
superiority when the temporal gap between fault onset and cascade effects
is eliminated.

The experiment compares performance across two timing regimes:
    - Baseline: 0.8-1.5s gap (natural timing)
    - Zero-delay: 0.0s gap (no temporal advantage)

Usage
-----
From the experiments/ directory:
    $ python zero_delay.py
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.simulator import SimulationConfig, ScalingMode
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator
from src.methods.threshold import ThresholdDetector
from src.methods.voting_simplified import SimpleVotingSimulator


# Experiment configuration
DEFAULT_SEEDS = [42, 43, 44]
BASELINE_TIMING = (0.8, 1.5)
ZERO_DELAY_TIMING = (0.0, 0.0)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for zero-delay experiment.

    Parameters
    ----------
    num_agents : int
        Number of agents in the swarm.
    fault_rate : float
        Proportion of faulty agents (0.0 to 1.0).
    run_time : float
        Simulation duration in seconds.
    """
    num_agents: int = 60
    fault_rate: float = 0.30
    run_time: float = 30.0


@dataclass
class MethodResult:
    """Results from a single method evaluation.

    Attributes
    ----------
    tcr : float
        Task completion rate.
    ccp : float
        Cascade prevention score.
    detection : dict
        Detection metrics (precision, recall, TP, FP).
    source_class : dict
        Source classification accuracy (hormone method only).
    """
    tcr: float
    ccp: float
    detection: Dict[str, float]
    source_class: Dict[str, float]


class ZeroDelayExperiment:
    """Orchestrates zero-delay timing experiments.

    This class manages the execution and analysis of experiments comparing
    detection methods with and without temporal advantages.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    output_dir : Path
        Directory for saving results.

    Attributes
    ----------
    methods : list of str
        Detection methods to evaluate.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.methods = ['baseline', 'hormone', 'threshold', 'voting']

    async def run_single_evaluation(
        self,
        method: str,
        timing_gap: Tuple[float, float],
        seed: int
    ) -> MethodResult:
        """Evaluate a single method with specified timing.

        Parameters
        ----------
        method : str
            Method name: {'baseline', 'hormone', 'threshold', 'voting'}.
        timing_gap : tuple of float
            (min_gap, max_gap) in seconds.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        MethodResult
            Evaluation results.
        """
        # Configure simulation
        sim_config = SimulationConfig(
            num_agents=self.config.num_agents,
            fault_rate=self.config.fault_rate,
            run_time=self.config.run_time,
            scaling_mode=ScalingMode.FIXED_ARENA,
            enable_quarantine=(method != "baseline"),
            packet_loss_rate=0.0,
            sensor_noise_level=0.0,
            random_seed=seed
        )

        # Add timing configuration
        sim_config.timing_regime = timing_gap
        sim_config.severity_distribution = {
            "minor": 0.60,
            "moderate": 0.30,
            "severe": 0.10
        }

        # Instantiate simulator
        simulator = self._create_simulator(method, sim_config)

        # Execute simulation
        metrics = await simulator.run_simulation()

        # Extract detection metrics
        detection = self._extract_detection_metrics(simulator)

        # Extract source classification (hormone only)
        source_class = self._extract_source_classification(simulator)

        return MethodResult(
            tcr=metrics.task_completion,
            ccp=metrics.cascade_prevention,
            detection=detection,
            source_class=source_class
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
            'baseline': BaselineSimulator,
            'hormone': LocalHormoneSimulator,
            'threshold': ThresholdDetector,
            'voting': SimpleVotingSimulator
        }

        if method not in simulators:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid options: {list(simulators.keys())}"
            )

        return simulators[method](config)

    @staticmethod
    def _extract_detection_metrics(simulator) -> Dict[str, float]:
        """Extract detection performance metrics.

        Parameters
        ----------
        simulator : Simulator
            Completed simulator instance.

        Returns
        -------
        dict
            Detection metrics (precision, recall, TP, FP).
        """
        if not hasattr(simulator, 'detection_stats'):
            return {}

        stats = simulator.detection_stats
        tp = stats.get('tp', 0)
        fp = stats.get('fp', 0)
        fn = stats.get('fn', 0)

        return {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'tp': tp,
            'fp': fp
        }

    @staticmethod
    def _extract_source_classification(simulator) -> Dict[str, float]:
        """Extract source classification metrics (hormone method).

        Parameters
        ----------
        simulator : Simulator
            Completed simulator instance.

        Returns
        -------
        dict
            Source classification metrics.
        """
        if not hasattr(simulator, 'quarantine_event_log'):
            return {}

        events = simulator.quarantine_event_log
        if not events:
            return {}

        source_like = sum(1 for e in events if e.get('class') == 'source-like')
        victim_like = sum(1 for e in events if e.get('class') == 'victim-like')

        correct = sum(
            1 for e in events
            if (e.get('class') == 'source-like' and e.get('label_at_t')) or
               (e.get('class') == 'victim-like' and not e.get('label_at_t'))
        )

        return {
            'source_like': source_like,
            'victim_like': victim_like,
            'accuracy': correct / len(events)
        }

    async def run_comparison(
        self,
        timing_regime: Tuple[float, float],
        seeds: List[int]
    ) -> Dict[str, List[MethodResult]]:
        """Execute full comparison across methods.

        Parameters
        ----------
        timing_regime : tuple of float
            (min_gap, max_gap) timing configuration.
        seeds : list of int
            Random seeds for repeated trials.

        Returns
        -------
        dict
            Results for each method across all seeds.
        """
        results = {method: [] for method in self.methods}

        for seed_idx, seed in enumerate(seeds, 1):
            for method in self.methods:
                result = await self.run_single_evaluation(
                    method, timing_regime, seed
                )
                results[method].append(result)

        return results

    def compute_summary_statistics(
        self,
        results: Dict[str, List[MethodResult]]
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Compute mean and std for each metric.

        Parameters
        ----------
        results : dict
            Results dictionary from run_comparison.

        Returns
        -------
        dict
            Summary statistics: {method: {metric: (mean, std)}}.
        """
        summaries = {}

        for method, method_results in results.items():
            if not method_results:
                continue

            tcr_vals = [r.tcr for r in method_results]
            ccp_vals = [r.ccp for r in method_results]

            summary = {
                'tcr': (np.mean(tcr_vals), np.std(tcr_vals)),
                'ccp': (np.mean(ccp_vals), np.std(ccp_vals))
            }

            if method != 'baseline':
                prec_vals = [r.detection['precision'] for r in method_results]
                rec_vals = [r.detection['recall'] for r in method_results]
                summary['precision'] = (np.mean(prec_vals), np.std(prec_vals))
                summary['recall'] = (np.mean(rec_vals), np.std(rec_vals))

            if method == 'hormone':
                acc_vals = [
                    r.source_class['accuracy'] for r in method_results
                    if r.source_class
                ]
                if acc_vals:
                    summary['source_accuracy'] = (np.mean(acc_vals), np.std(acc_vals))

            summaries[method] = summary

        return summaries

    def export_results(
        self,
        baseline_results: Dict[str, List[MethodResult]],
        zero_delay_results: Dict[str, List[MethodResult]]
    ) -> Tuple[Path, Path]:
        """Export results to CSV and summary files.

        Parameters
        ----------
        baseline_results : dict
            Results from baseline timing.
        zero_delay_results : dict
            Results from zero-delay timing.

        Returns
        -------
        tuple of Path
            Paths to (CSV file, summary file).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export CSV
        csv_path = self._export_csv(
            baseline_results, zero_delay_results, timestamp
        )

        # Export summary
        summary_path = self._export_summary(
            baseline_results, zero_delay_results, timestamp
        )

        return csv_path, summary_path

    def _export_csv(
        self,
        baseline_results: Dict[str, List[MethodResult]],
        zero_delay_results: Dict[str, List[MethodResult]],
        timestamp: str
    ) -> Path:
        """Export results to CSV format."""
        csv_path = self.output_dir / f"zero_delay_{timestamp}.csv"

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("method,timing_condition,tcr_mean,tcr_std,"
                   "ccp_mean,ccp_std,precision_mean,precision_std\n")

            for method in self.methods:
                self._write_method_csv_rows(
                    f, method, baseline_results, zero_delay_results
                )

        return csv_path

    def _write_method_csv_rows(
        self,
        file,
        method: str,
        baseline_results: Dict[str, List[MethodResult]],
        zero_delay_results: Dict[str, List[MethodResult]]
    ):
        """Write CSV rows for a single method."""
        for label, results in [
            ("baseline_timing", baseline_results),
            ("zero_delay", zero_delay_results)
        ]:
            method_results = results[method]

            tcr_vals = [r.tcr for r in method_results]
            ccp_vals = [r.ccp for r in method_results]

            if method != 'baseline':
                prec_vals = [r.detection['precision'] for r in method_results]
                prec_str = f"{np.mean(prec_vals):.3f},{np.std(prec_vals):.3f}"
            else:
                prec_str = "NA,NA"

            file.write(
                f"{method},{label},"
                f"{np.mean(tcr_vals):.3f},{np.std(tcr_vals):.3f},"
                f"{np.mean(ccp_vals):.3f},{np.std(ccp_vals):.3f},"
                f"{prec_str}\n"
            )

    def _export_summary(
        self,
        baseline_results: Dict[str, List[MethodResult]],
        zero_delay_results: Dict[str, List[MethodResult]],
        timestamp: str
    ) -> Path:
        """Export human-readable summary."""
        summary_path = self.output_dir / f"zero_delay_summary_{timestamp}.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Zero-Delay Experiment Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Seeds tested: {len(baseline_results['baseline'])}\n\n")

            self._write_condition_summary(f, "Baseline Timing", baseline_results)
            self._write_condition_summary(f, "Zero-Delay", zero_delay_results)
            self._write_key_findings(f, zero_delay_results)

        return summary_path

    def _write_condition_summary(
        self,
        file,
        label: str,
        results: Dict[str, List[MethodResult]]
    ):
        """Write summary for a timing condition."""
        file.write(f"\n{label}\n")
        file.write("-" * 40 + "\n")

        for method in self.methods:
            tcr = np.mean([r.tcr for r in results[method]])
            file.write(f"{method:12s}: TCR = {tcr:.3f}\n")

    def _write_key_findings(
        self,
        file,
        zero_delay_results: Dict[str, List[MethodResult]]
    ):
        """Write key findings section."""
        file.write("\n\nKey Findings\n")
        file.write("-" * 40 + "\n")

        h_tcr = np.mean([r.tcr for r in zero_delay_results['hormone']])
        t_tcr = np.mean([r.tcr for r in zero_delay_results['threshold']])
        v_tcr = np.mean([r.tcr for r in zero_delay_results['voting']])

        if h_tcr > max(t_tcr, v_tcr):
            file.write(
                "Hormone method maintains superiority without temporal advantage\n"
            )
            file.write(f"  Hormone:   {h_tcr:.3f}\n")
            file.write(f"  Threshold: {t_tcr:.3f}\n")
            file.write(f"  Voting:    {v_tcr:.3f}\n")
        else:
            file.write(
                "Hormone method does not maintain superiority with zero-delay\n"
            )


def print_summary(
    label: str,
    summaries: Dict[str, Dict[str, Tuple[float, float]]]
):
    """Print formatted summary statistics.

    Parameters
    ----------
    label : str
        Summary label.
    summaries : dict
        Summary statistics from compute_summary_statistics.
    """
    print(f"\n{'=' * 70}")
    print(f"Summary: {label}")
    print(f"{'=' * 70}\n")

    for method in ['baseline', 'hormone', 'threshold', 'voting']:
        if method not in summaries:
            continue

        print(f"{method.upper()}:")

        summary = summaries[method]
        for metric, (mean, std) in summary.items():
            print(f"  {metric:20s}: {mean:.3f} +/- {std:.3f}")

        print()


def print_comparison(
    baseline_summaries: Dict[str, Dict[str, Tuple[float, float]]],
    zero_delay_summaries: Dict[str, Dict[str, Tuple[float, float]]]
):
    """Print comparative analysis.

    Parameters
    ----------
    baseline_summaries : dict
        Baseline timing summaries.
    zero_delay_summaries : dict
        Zero-delay summaries.
    """
    print(f"\n{'=' * 70}")
    print("Comparison: Baseline vs Zero-Delay")
    print(f"{'=' * 70}\n")

    for method in ['hormone', 'threshold', 'voting']:
        print(f"{method.upper()}:")

        baseline = baseline_summaries[method]
        zero_delay = zero_delay_summaries[method]

        for metric in ['tcr', 'ccp', 'precision']:
            if metric not in baseline or metric not in zero_delay:
                continue

            b_mean, _ = baseline[metric]
            z_mean, _ = zero_delay[metric]
            pct_change = ((z_mean - b_mean) / b_mean * 100) if b_mean > 0 else 0

            print(f"  {metric:12s}: {b_mean:.3f} -> {z_mean:.3f} ({pct_change:+.1f}%)")

        print()


async def main():
    """Execute zero-delay experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Zero-delay temporal advantage analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help='random seeds for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='output directory for results'
    )
    args = parser.parse_args()

    # Configure experiment
    config = ExperimentConfig()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results"

    experiment = ZeroDelayExperiment(config, output_dir)

    print("\n" + "=" * 70)
    print("Zero-Delay Experiment")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print(f"Output: {output_dir}")

    # Run baseline timing
    print("\n\nBaseline Timing (0.8-1.5s gap)")
    print("-" * 70)
    baseline_results = await experiment.run_comparison(BASELINE_TIMING, args.seeds)
    baseline_summaries = experiment.compute_summary_statistics(baseline_results)

    # Run zero-delay timing
    print("\n\nZero-Delay (0.0s gap)")
    print("-" * 70)
    zero_delay_results = await experiment.run_comparison(ZERO_DELAY_TIMING, args.seeds)
    zero_delay_summaries = experiment.compute_summary_statistics(zero_delay_results)

    # Print results
    print_summary("Baseline Timing", baseline_summaries)
    print_summary("Zero-Delay", zero_delay_summaries)
    print_comparison(baseline_summaries, zero_delay_summaries)

    # Export results
    csv_path, summary_path = experiment.export_results(
        baseline_results, zero_delay_results
    )

    print("\n" + "=" * 70)
    print("Results Exported")
    print("=" * 70)
    print(f"\nCSV:     {csv_path}")
    print(f"Summary: {summary_path}\n")


if __name__ == "__main__":
    asyncio.run(main())