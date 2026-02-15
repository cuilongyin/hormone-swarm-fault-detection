#!/usr/bin/env python3
"""Orchestration framework for swarm fault tolerance experiments.

This module provides a comprehensive experiment runner that manages the
execution of multiple experimental configurations with intelligent caching,
error recovery, and progress tracking. The framework is designed to handle
long-running experimental suites while maintaining data integrity and
providing clear status reporting.

Key features:
    - Smart caching: Skips experiments with existing results
    - Error isolation: One failure does not halt the entire pipeline
    - Progress tracking: Detailed logging and time accounting
    - Automatic figure generation: Post-processing of all results
    - Comprehensive reporting: JSON logs and summary statistics
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    script : Path
        Path to the experiment script.
    outputs : list of str
        Expected output filenames (supports wildcards).
    description : str
        Brief description of the experiment's purpose.
    """
    name: str
    script: Path
    outputs: List[str]
    description: str


@dataclass
class ExperimentResult:
    """Result container for experiment execution.

    Attributes
    ----------
    status : str
        Execution status: 'success', 'skipped', 'failed', or 'crashed'.
    time : float
        Execution time in seconds.
    error : str, optional
        Error message if execution failed.
    outputs_created : bool, optional
        Whether all expected outputs were created.
    """
    status: str
    time: float = 0.0
    error: Optional[str] = None
    outputs_created: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format.

        Returns
        -------
        dict
            Dictionary representation of the result.
        """
        result = {
            'status': self.status,
            'time': self.time
        }
        if self.error is not None:
            result['error'] = self.error
        if self.outputs_created is not None:
            result['outputs_created'] = self.outputs_created
        return result


class PathManager:
    """Manages directory structure for experiments and results.

    This class centralizes path management and ensures all required
    directories exist before experiment execution begins.
    """

    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.experiments_dir = self.root_dir / "experiments"
        self.figures_dir = self.root_dir / "figures"
        self.results_dir = self.root_dir / "results"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        logger.info(f"Directories initialized: {self.root_dir}")


class OutputValidator:
    """Validates existence of experiment output files.

    This class handles both exact filename matching and pattern-based
    matching (wildcards) for flexible output validation.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def file_exists(self, filename_pattern: str) -> bool:
        """Check if a result file exists.

        Parameters
        ----------
        filename_pattern : str
            Filename or pattern (with wildcards) to check.

        Returns
        -------
        bool
            True if file(s) matching the pattern exist.

        Notes
        -----
        Supports both exact matches (e.g., "results.csv") and wildcard
        patterns (e.g., "run_*_data.csv") for flexible file detection.
        """
        if '*' in filename_pattern:
            matches = list(self.results_dir.glob(filename_pattern))
            return len(matches) > 0
        else:
            return (self.results_dir / filename_pattern).exists()

    def check_all_outputs(self, output_files: List[str]) -> tuple[List[str], List[str]]:
        """Check which output files exist and which are missing.

        Parameters
        ----------
        output_files : list of str
            List of expected output filenames.

        Returns
        -------
        existing : list of str
            Files that exist.
        missing : list of str
            Files that are missing.
        """
        existing = []
        missing = []

        for output_file in output_files:
            if self.file_exists(output_file):
                existing.append(output_file)
            else:
                missing.append(output_file)

        return existing, missing


class ExperimentRunner:
    """Manages execution of experimental suite with error handling.

    This class orchestrates the execution of multiple experiments,
    handling caching, error recovery, and result aggregation. It
    maintains detailed timing information and execution logs.
    """

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.validator = OutputValidator(path_manager.results_dir)
        self.results: Dict[str, ExperimentResult] = {}
        self.start_time = time.time()

    @property
    def total_time(self) -> float:
        """Total elapsed time since runner initialization."""
        return time.time() - self.start_time

    async def run_experiment(self, config: ExperimentConfig) -> bool:
        """Execute a single experiment with caching and error handling.

        Parameters
        ----------
        config : ExperimentConfig
            Configuration for the experiment to run.

        Returns
        -------
        bool
            True if experiment succeeded, False if it failed.

        Notes
        -----
        This method implements intelligent caching: if all expected
        outputs already exist, the experiment is skipped. This allows
        for efficient re-running of partial experimental suites.
        """
        logger.info("=" * 80)
        logger.info(f"EXPERIMENT: {config.name}")
        logger.info("=" * 80)
        logger.info(f"Description: {config.description}")

        # Check for existing outputs
        existing_files, missing_files = self.validator.check_all_outputs(config.outputs)

        if not missing_files:
            logger.info(f"All output files already exist:")
            for f in existing_files:
                logger.info(f"  - {f}")
            logger.info(f"Skipping {config.name}")

            self.results[config.name] = ExperimentResult(
                status='skipped',
                time=0.0
            )
            return True

        logger.info(f"Missing files: {missing_files}")
        logger.info(f"Running {config.name}...")

        try:
            start = time.time()

            # Execute experiment as subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(config.script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.path_manager.experiments_dir)
            )

            stdout, stderr = await process.communicate()
            elapsed = time.time() - start

            if process.returncode == 0:
                logger.info(f"{config.name} completed successfully in {elapsed / 60:.1f} minutes")

                # Verify output creation
                _, still_missing = self.validator.check_all_outputs(config.outputs)
                all_created = len(still_missing) == 0

                if all_created:
                    logger.info("All expected outputs created:")
                    for f in config.outputs:
                        if self.validator.file_exists(f):
                            logger.info(f"  - {f}")
                else:
                    logger.warning("Some expected outputs missing:")
                    for f in still_missing:
                        logger.warning(f"  X {f}")

                self.results[config.name] = ExperimentResult(
                    status='success',
                    time=elapsed,
                    outputs_created=all_created
                )
                return True

            else:
                error_msg = stderr.decode() if stderr else 'Unknown error'
                logger.error(f"{config.name} failed with return code {process.returncode}")
                if stderr:
                    logger.error(f"Error output:\n{error_msg}")

                self.results[config.name] = ExperimentResult(
                    status='failed',
                    time=elapsed,
                    error=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"{config.name} crashed with exception: {str(e)}")
            self.results[config.name] = ExperimentResult(
                status='crashed',
                time=time.time() - start,
                error=str(e)
            )
            return False

    def generate_figures(self) -> bool:
        """Execute figure generation script.

        Returns
        -------
        bool
            True if figure generation succeeded, False otherwise.
        """
        logger.info("=" * 80)
        logger.info("GENERATING FIGURES")
        logger.info("=" * 80)

        try:
            script_path = self.path_manager.figures_dir / "generate_figures.py"

            if not script_path.exists():
                logger.warning(f"Figure generation script not found: {script_path}")
                return False

            logger.info("Running generate_figures.py...")
            start = time.time()

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.path_manager.figures_dir),
                capture_output=True,
                text=True
            )

            elapsed = time.time() - start

            if result.returncode == 0:
                logger.info(f"Figures generated successfully in {elapsed:.1f} seconds")
                logger.info(result.stdout)

                self.results['figure_generation'] = ExperimentResult(
                    status='success',
                    time=elapsed
                )
                return True
            else:
                logger.error("Figure generation failed")
                logger.error(result.stderr)

                self.results['figure_generation'] = ExperimentResult(
                    status='failed',
                    time=elapsed,
                    error=result.stderr
                )
                return False

        except Exception as e:
            logger.exception(f"Figure generation crashed: {str(e)}")
            self.results['figure_generation'] = ExperimentResult(
                status='crashed',
                time=0.0,
                error=str(e)
            )
            return False

    def save_execution_log(self) -> Path:
        """Save detailed execution log to JSON file.

        Returns
        -------
        Path
            Path to the saved log file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.path_manager.results_dir / f"run_all_log_{timestamp}.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_hours': self.total_time / 3600,
            'total_time_minutes': self.total_time / 60,
            'experiments': {
                name: result.to_dict()
                for name, result in self.results.items()
            }
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Execution log saved: {log_file}")
        return log_file

    def generate_summary(self) -> None:
        """Print comprehensive execution summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)

        start_dt = datetime.fromtimestamp(self.start_time)
        logger.info(f"Started:  {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {self.total_time / 3600:.2f} hours ({self.total_time / 60:.1f} minutes)")

        # Compute statistics
        success_count = sum(1 for r in self.results.values() if r.status == 'success')
        skipped_count = sum(1 for r in self.results.values() if r.status == 'skipped')
        failed_count = sum(1 for r in self.results.values() if r.status in ['failed', 'crashed'])

        logger.info("\nExperiment Results:")
        logger.info("-" * 80)

        # Status icons
        status_icons = {
            'success': '[OK]',
            'skipped': '[SKIP]',
            'failed': '[FAIL]',
            'crashed': '[CRASH]'
        }

        for exp_name, exp_result in self.results.items():
            icon = status_icons.get(exp_result.status, '[?]')
            status = exp_result.status.upper()
            elapsed_min = exp_result.time / 60

            logger.info(f"  {icon} {exp_name:30s}: {status:8s} ({elapsed_min:5.1f} min)")

        logger.info("\nStatistics:")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Failed:  {failed_count}")
        logger.info(f"  Total:   {len(self.results)}")
        logger.info("=" * 80)


class ExperimentSuite:
    """Defines the complete experimental suite configuration.

    This class serves as a central registry for all experiments
    in the research pipeline, providing clear documentation of
    each experiment's purpose and expected outputs.
    """

    @staticmethod
    def get_experiments(experiments_dir: Path) -> List[ExperimentConfig]:
        """Define all experiments in the suite.

        Parameters
        ----------
        experiments_dir : Path
            Base directory containing experiment scripts.

        Returns
        -------
        list of ExperimentConfig
            Complete list of experiment configurations.
        """
        return [
            ExperimentConfig(
                name='Scalability Test',
                script=experiments_dir / 'test_scalability.py',
                outputs=['scalability_results_before.csv'],
                description='Tests performance across swarm sizes 30-120'
            ),
            ExperimentConfig(
                name='Focused Scalability',
                script=experiments_dir / 'test_scalability_focused.py',
                outputs=['scalability_focused.csv'],
                description='Variable seeds: 3 for sizes 30-70, 10 for sizes 80-120'
            ),
            ExperimentConfig(
                name='Packet Loss Sweep',
                script=experiments_dir / 'test_packet_loss_sweep.py',
                outputs=['packet_loss_sweep.csv'],
                description='Tests communication degradation from 0% to 100% loss'
            ),
            ExperimentConfig(
                name='Extended Packet Loss',
                script=experiments_dir / 'extended_packet_loss.py',
                outputs=['packet_loss_raw.csv', 'packet_loss_aggregated.csv'],
                description='Comprehensive packet loss testing with multiple metrics'
            ),
            ExperimentConfig(
                name='Ablation Study',
                script=experiments_dir / 'test_ablation_ratios.py',
                outputs=['ablation_ratios.csv'],
                description='Tests hormone method with ratio-conditioning disabled'
            ),
            ExperimentConfig(
                name='Zero Delay Experiment',
                script=experiments_dir / 'zero_delay.py',
                outputs=['zero_delay_results_*.csv', 'zero_delay_summary_*.txt'],
                description='Tests if hormone method works without temporal advantage'
            ),
            ExperimentConfig(
                name='Fault Severity Response',
                script=experiments_dir / 'fault_severity_response.py',
                outputs=['fault_severity_response.csv'],
                description='Tests response to different fault severity distributions'
            ),
            ExperimentConfig(
                name='Cascade Timeline',
                script=experiments_dir / 'cascade_timeline_collector.py',
                outputs=['cascade_timeline_fair.csv'],
                description='Collects high-resolution cascade progression data'
            )
        ]


async def main():
    """Main execution function for experimental suite.

    This function orchestrates the complete experimental pipeline:
    1. Initialize directory structure
    2. Load experiment configurations
    3. Execute experiments with caching
    4. Generate figures from results
    5. Save execution logs and summary
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENTAL SUITE ORCHESTRATOR")
    logger.info("=" * 80)

    # Initialize infrastructure
    path_manager = PathManager()
    path_manager.ensure_directories()

    logger.info(f"Root directory: {path_manager.root_dir}")
    logger.info(f"Results directory: {path_manager.results_dir}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("\nPipeline features:")
    logger.info("  - Smart caching: Skip experiments with existing results")
    logger.info("  - Error isolation: Continue on failure")
    logger.info("  - Progress tracking: Detailed logging")
    logger.info("  - Automatic post-processing: Figure generation")
    logger.info("=" * 80 + "\n")

    # Initialize runner
    runner = ExperimentRunner(path_manager)

    # Load experiment suite
    experiments = ExperimentSuite.get_experiments(path_manager.experiments_dir)

    # Execute all experiments
    for config in experiments:
        await runner.run_experiment(config)

    # Generate figures
    runner.generate_figures()

    # Save logs and generate summary
    runner.save_execution_log()
    runner.generate_summary()

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Review results/ folder for CSV files")
    logger.info("  2. Examine figures/ folder for generated plots")
    logger.info("  3. Consult execution log for detailed information")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\n\nExecution interrupted by user")
        logger.info("Partial results may be available in results/ folder")
    except Exception as e:
        logger.exception(f"\n\nPipeline failed with error: {str(e)}")
        import traceback

        traceback.print_exc()