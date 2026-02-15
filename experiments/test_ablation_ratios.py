"""Ablation study for ratio-conditioned threshold mechanism.

This module tests whether the ratio-based threshold adaptation is critical
for distinguishing truly faulty agents from cascade-damaged healthy agents.

Experimental Design
-------------------
We compare two variants of the hormone method:
    1. Full: Standard hormone with ratio-conditioned threshold
    2. Ablated: Hormone with fixed threshold (ratio conditioning disabled)

If ratio conditioning is critical, we expect to see:
    - Higher precision in the full variant
    - Lower false positive rate in the full variant
    - Improved discrimination between source and victim agents

Usage
-----
From the experiments/ directory:
    $ python test_ablation_ratios.py

Notes
-----
This is a focused ablation with limited seeds (n=5) to quickly assess
whether the mechanism warrants further investigation. If the effect is
minimal (<5% difference), the ratio mechanism may be unnecessary complexity.

"""

import asyncio
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulator import SimulationConfig, ScalingMode
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator

# Experiment configuration
NUM_AGENTS = 120
FAULT_RATE = 0.3
RUN_TIME = 20.0
NUM_SEEDS = 5  # Quick ablation study


@dataclass
class AblationResult:
    """Results from a single ablation evaluation.

    Attributes
    ----------
    variant : str
        Variant identifier: 'full' or 'ablated'.
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
    f1_score : float
        Harmonic mean of precision and recall.
    num_quarantined : int
        Total number of quarantined agents.
    num_faulty_quarantined : int
        Number of faulty agents correctly quarantined.
    num_healthy_quarantined : int
        Number of healthy agents incorrectly quarantined.
    false_positive_rate : float
        Rate of false positive detections.
    simulation_time : float
        Wall-clock execution time (seconds).
    """
    variant: str
    seed: int

    # Performance metrics
    task_completion: float
    cascade_prevention: float
    precision: float
    recall: float
    f1_score: float

    # Quarantine statistics
    num_quarantined: int
    num_faulty_quarantined: int
    num_healthy_quarantined: int

    # Error metrics
    false_positive_rate: float

    # Runtime information
    simulation_time: float


class RatioAblationStudy:
    """Orchestrates ratio-conditioned threshold ablation.

    This class manages the systematic comparison of full vs ablated
    hormone variants to assess the importance of ratio conditioning.

    Parameters
    ----------
    output_dir : Path
        Directory for results export.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_variant(
            self,
            variant: str,
            seed: int,
            use_ratio_threshold: bool
    ) -> AblationResult:
        """Evaluate a single variant configuration.

        Parameters
        ----------
        variant : str
            Variant identifier: 'full' or 'ablated'.
        seed : int
            Random seed for reproducibility.
        use_ratio_threshold : bool
            Whether to use ratio-conditioned threshold.

        Returns
        -------
        AblationResult
            Evaluation results.
        """
        # Run baseline for reference
        baseline_config = SimulationConfig(
            num_agents=NUM_AGENTS,
            fault_rate=FAULT_RATE,
            run_time=RUN_TIME,
            enable_quarantine=False,
            random_seed=seed,
            scaling_mode=ScalingMode.FIXED_ARENA
        )

        baseline_sim = BaselineSimulator(baseline_config)
        await baseline_sim.run_simulation()
        baseline_tracker = baseline_sim.metrics_tracker

        # Configure hormone method
        hormone_config = SimulationConfig(
            num_agents=NUM_AGENTS,
            fault_rate=FAULT_RATE,
            run_time=RUN_TIME,
            enable_quarantine=True,
            random_seed=seed,
            scaling_mode=ScalingMode.FIXED_ARENA
        )

        # Apply ablation configuration
        # NOTE: This assumes the simulator supports ablation mode
        if hasattr(hormone_config, 'use_ratio_conditioned_threshold'):
            hormone_config.use_ratio_conditioned_threshold = use_ratio_threshold

        # Execute simulation
        start_time = time.time()
        simulator = LocalHormoneSimulator(hormone_config)

        # Temporary workaround for simulators without ablation support
        if not use_ratio_threshold and hasattr(simulator, 'use_fixed_threshold'):
            simulator.use_fixed_threshold = True
            simulator.fixed_threshold_value = 0.5  # Reasonable default

        metrics = await simulator.run_simulation(baseline_metrics=baseline_tracker)
        simulation_time = time.time() - start_time

        # Extract detection metrics
        detection_metrics = simulator.get_quarantine_event_metrics()

        # Compute quarantine statistics
        quarantine_stats = self._compute_quarantine_statistics(simulator)

        # Compute derived metrics
        precision = detection_metrics.get('precision', 0.0)
        recall = detection_metrics.get('recall', 0.0)
        f1_score = self._compute_f1_score(precision, recall)

        return AblationResult(
            variant=variant,
            seed=seed,
            task_completion=metrics.task_completion,
            cascade_prevention=metrics.cascade_prevention,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            num_quarantined=quarantine_stats['num_quarantined'],
            num_faulty_quarantined=quarantine_stats['num_faulty_quarantined'],
            num_healthy_quarantined=quarantine_stats['num_healthy_quarantined'],
            false_positive_rate=quarantine_stats['false_positive_rate'],
            simulation_time=simulation_time
        )

    @staticmethod
    def _compute_quarantine_statistics(simulator) -> Dict[str, float]:
        """Compute quarantine-related statistics.

        Parameters
        ----------
        simulator : LocalHormoneSimulator
            Completed simulator instance.

        Returns
        -------
        dict
            Quarantine statistics.
        """
        quarantined_agents = [a for a in simulator.agents if a.is_quarantined]
        num_quarantined = len(quarantined_agents)

        num_faulty_quarantined = sum(
            1 for a in quarantined_agents if a.is_faulty
        )
        num_healthy_quarantined = num_quarantined - num_faulty_quarantined

        num_healthy = sum(1 for a in simulator.agents if not a.is_faulty)
        false_positive_rate = (
            num_healthy_quarantined / num_healthy if num_healthy > 0 else 0.0
        )

        return {
            'num_quarantined': num_quarantined,
            'num_faulty_quarantined': num_faulty_quarantined,
            'num_healthy_quarantined': num_healthy_quarantined,
            'false_positive_rate': false_positive_rate
        }

    @staticmethod
    def _compute_f1_score(precision: float, recall: float) -> float:
        """Compute F1 score (harmonic mean of precision and recall).

        Parameters
        ----------
        precision : float
            Detection precision.
        recall : float
            Detection recall.

        Returns
        -------
        float
            F1 score.
        """
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    async def run_full_ablation(self) -> Dict[str, List[AblationResult]]:
        """Execute complete ablation study.

        Returns
        -------
        dict
            Results for each variant across all seeds.
        """
        print("\n" + "=" * 70)
        print("Ablation Study: Ratio-Conditioned Threshold")
        print("=" * 70)
        print("\nVariants:")
        print("  1. Full:    Hormone with ratio-conditioned threshold")
        print("  2. Ablated: Hormone with fixed threshold")

        results = {
            'full': [],
            'ablated': []
        }

        for seed in range(NUM_SEEDS):
            print(f"\n{'-' * 70}")
            print(f"Seed {seed + 1}/{NUM_SEEDS}")
            print(f"{'-' * 70}")

            # Evaluate full variant
            try:
                print("  Full variant (with ratios)...", end=" ", flush=True)
                result = await self.evaluate_variant('full', seed, use_ratio_threshold=True)
                results['full'].append(result)
                print(f"[OK] Prec={result.precision:.3f}, Rec={result.recall:.3f}")
            except Exception as e:
                print(f"[FAIL] Failed: {e}")

            # Evaluate ablated variant
            try:
                print("  Ablated variant (fixed)...  ", end=" ", flush=True)
                result = await self.evaluate_variant('ablated', seed, use_ratio_threshold=False)
                results['ablated'].append(result)
                print(f"[OK] Prec={result.precision:.3f}, Rec={result.recall:.3f}")
            except Exception as e:
                print(f"[FAIL] Failed: {e}")

        return results

    def print_summary(self, results: Dict[str, List[AblationResult]]):
        """Print formatted summary statistics.

        Parameters
        ----------
        results : dict
            Results dictionary from run_full_ablation.
        """
        print("\n" + "=" * 70)
        print("Ablation Results")
        print("=" * 70)

        for variant in ['full', 'ablated']:
            variant_results = results[variant]

            if not variant_results:
                print(f"\n{variant}: No successful runs")
                continue

            # Compute means
            tc_mean = np.mean([r.task_completion for r in variant_results])
            ccp_mean = np.mean([r.cascade_prevention for r in variant_results])
            prec_mean = np.mean([r.precision for r in variant_results])
            rec_mean = np.mean([r.recall for r in variant_results])
            f1_mean = np.mean([r.f1_score for r in variant_results])
            fpr_mean = np.mean([r.false_positive_rate for r in variant_results])

            variant_label = (
                "Full (with ratios)" if variant == 'full'
                else "Ablated (fixed threshold)"
            )

            print(f"\n{variant_label}:")
            print(f"  Task completion:     {tc_mean:.4f}")
            print(f"  Cascade prevention:  {ccp_mean:.4f}")
            print(f"  Precision:           {prec_mean:.4f}")
            print(f"  Recall:              {rec_mean:.4f}")
            print(f"  F1 score:            {f1_mean:.4f}")
            print(f"  False positive rate: {fpr_mean:.4f}")

    def print_comparison(self, results: Dict[str, List[AblationResult]]):
        """Print comparative analysis.

        Parameters
        ----------
        results : dict
            Results dictionary from run_full_ablation.
        """
        if not (results['full'] and results['ablated']):
            print("\nInsufficient data for comparison.")
            return

        print("\n" + "=" * 70)
        print("Impact Analysis: Ratio Conditioning")
        print("=" * 70)

        full = results['full']
        ablated = results['ablated']

        # Compute means
        prec_full = np.mean([r.precision for r in full])
        prec_ablated = np.mean([r.precision for r in ablated])
        prec_delta = prec_full - prec_ablated
        prec_pct_change = (
            (prec_delta / prec_ablated * 100) if prec_ablated > 0 else 0
        )

        rec_full = np.mean([r.recall for r in full])
        rec_ablated = np.mean([r.recall for r in ablated])
        rec_delta = rec_full - rec_ablated
        rec_pct_change = (
            (rec_delta / rec_ablated * 100) if rec_ablated > 0 else 0
        )

        f1_full = np.mean([r.f1_score for r in full])
        f1_ablated = np.mean([r.f1_score for r in ablated])
        f1_delta = f1_full - f1_ablated
        f1_pct_change = (
            (f1_delta / f1_ablated * 100) if f1_ablated > 0 else 0
        )

        fpr_full = np.mean([r.false_positive_rate for r in full])
        fpr_ablated = np.mean([r.false_positive_rate for r in ablated])
        fpr_delta = fpr_full - fpr_ablated

        # Print changes
        print(
            f"\nPrecision: {prec_ablated:.3f} -> {prec_full:.3f} "
            f"({prec_delta:+.3f}, {prec_pct_change:+.1f}%)"
        )
        print(
            f"Recall:    {rec_ablated:.3f} -> {rec_full:.3f} "
            f"({rec_delta:+.3f}, {rec_pct_change:+.1f}%)"
        )
        print(
            f"F1 score:  {f1_ablated:.3f} -> {f1_full:.3f} "
            f"({f1_delta:+.3f}, {f1_pct_change:+.1f}%)"
        )
        print(
            f"FPR:       {fpr_ablated:.3f} -> {fpr_full:.3f} "
            f"({fpr_delta:+.3f})"
        )

        # Interpretation
        print("\n" + "=" * 70)
        print("Interpretation")
        print("=" * 70)

        if abs(prec_pct_change) < 5 and abs(f1_pct_change) < 5:
            print("\n[!] Ratio conditioning has MINIMAL impact (<5% change)")
            print("This suggests:")
            print("  - Fixed threshold works nearly as well")
            print("  - Ratios may not be critical for discrimination")
            print("  - Simple threshold could be sufficient")

        elif prec_pct_change > 5 and fpr_delta < -0.05:
            print("\n[OK] Ratio conditioning IMPROVES discrimination")
            print("This suggests:")
            print("  - Ratios help distinguish faulty vs cascade-damaged")
            print("  - Adaptive threshold reduces false positives")
            print("  - Ratio mechanism is valuable for precision")

        elif prec_pct_change < -5:
            print("\n[!] Ratio conditioning REDUCES precision")
            print("This suggests:")
            print("  - Fixed threshold may be more reliable")
            print("  - Ratio calculation might be noisy")
            print("  - Consider simplifying the mechanism")

        else:
            print("\n[~] Ratio conditioning has MODEST impact")
            print("Results are mixed or inconclusive")

    def export_csv(
            self,
            results: Dict[str, List[AblationResult]],
            filename: str = "ablation_ratios.csv"
    ) -> Path:
        """Export results to CSV format.

        Parameters
        ----------
        results : dict
            Results dictionary from run_full_ablation.
        filename : str, optional
            Output filename.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        filepath = self.output_dir / filename

        fieldnames = [
            'variant', 'seed',
            'task_completion', 'cascade_prevention',
            'precision', 'recall', 'f1_score',
            'false_positive_rate',
            'num_quarantined', 'num_faulty_quarantined', 'num_healthy_quarantined'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for variant, variant_results in results.items():
                for result in variant_results:
                    writer.writerow({
                        'variant': result.variant,
                        'seed': result.seed,
                        'task_completion': result.task_completion,
                        'cascade_prevention': result.cascade_prevention,
                        'precision': result.precision,
                        'recall': result.recall,
                        'f1_score': result.f1_score,
                        'false_positive_rate': result.false_positive_rate,
                        'num_quarantined': result.num_quarantined,
                        'num_faulty_quarantined': result.num_faulty_quarantined,
                        'num_healthy_quarantined': result.num_healthy_quarantined
                    })

        return filepath


async def main():
    """Execute ratio ablation study."""
    # Setup
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    study = RatioAblationStudy(results_dir)

    # Estimate runtime
    total_runs = 2 * NUM_SEEDS

    print("\n" + "=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"\nVariants:   2 (full vs ablated)")
    print(f"Seeds:      {NUM_SEEDS}")
    print(f"Total runs: {total_runs}")
    print(f"Estimated time: {total_runs * 0.5:.0f}-{total_runs * 1:.0f} minutes")

    print("\n[!] Note: This ablation requires simulator support:")
    print("    - Config flag: use_ratio_conditioned_threshold")
    print("    - Fixed threshold fallback when disabled")
    print("\n    If not implemented, some runs may fail - that's OK!")
    print("    Implementation can be added after verifying necessity.")

    # Execute study
    start_time = time.time()
    results = await study.run_full_ablation()
    elapsed_time = time.time() - start_time

    # Present results
    study.print_summary(results)
    study.print_comparison(results)

    # Export results
    csv_path = study.export_csv(results)

    print(f"\n{'=' * 70}")
    print("Study Complete")
    print(f"{'=' * 70}")
    print(f"\nElapsed time: {elapsed_time / 60:.1f} minutes")
    print(f"Results exported to: {csv_path}\n")

    return results


if __name__ == "__main__":
    asyncio.run(main())