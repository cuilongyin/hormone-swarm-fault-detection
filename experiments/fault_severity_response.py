"""Fault Severity Response Analysis.

This module evaluates how fault detection methods adapt to different severity
distributions. By varying the mix of minor, moderate, and severe faults, we can
assess each method's ability to prioritize threats appropriately.

Three severity profiles are tested:
- Minor-dominant: 70% minor, 20% moderate, 10% severe
- Balanced: 33% each severity level
- Severe-dominant: 70% severe, 20% moderate, 10% minor

The experiment collects high-resolution time-series data (10 Hz) to capture
the dynamic response patterns of each detection method.

Example
-------
To run the severity response experiment:

    $ python fault_severity_response.py

This generates a CSV file with temporal dynamics for each method and severity profile.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Project path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simulator import ScalingMode, SimulationConfig
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator
from src.methods.threshold import ThresholdDetector
from src.methods.voting_simplified import SimpleVotingSimulator


# Severity distribution profiles
SEVERITY_PROFILES = {
    "minor_dominant": {"minor": 0.70, "moderate": 0.20, "severe": 0.10},
    "balanced": {"minor": 0.33, "moderate": 0.34, "severe": 0.33},
    "severe_dominant": {"minor": 0.10, "moderate": 0.20, "severe": 0.70},
}

# Experimental configuration
NUM_AGENTS = 60
FAULT_RATE = 0.20
SIMULATION_TIME = 15.0
SAMPLING_RATE = 10.0  # Hz


async def collect_severity_response_data() -> pd.DataFrame:
    """Execute severity response experiment and collect time-series data.

    Returns
    -------
    results : DataFrame
        Time-series metrics for all method-severity combinations.

    Notes
    -----
    Each combination of severity profile and detection method is run once
    with deterministic seeding for reproducibility. Data is sampled at 10 Hz
    to capture temporal dynamics.
    """
    print("Fault Severity Response Experiment")
    print("=" * 70)
    print(f"Configuration: {NUM_AGENTS} agents, {FAULT_RATE:.0%} fault rate, "
          f"{SIMULATION_TIME}s duration")
    print("=" * 70)

    # Method registry
    methods = {
        "baseline": BaselineSimulator,
        "threshold": ThresholdDetector,
        "voting": SimpleVotingSimulator,
        "hormone": LocalHormoneSimulator,
    }

    # Data collection
    all_results = []
    total_runs = len(SEVERITY_PROFILES) * len(methods)
    current_run = 0

    for severity_name, severity_dist in SEVERITY_PROFILES.items():
        print(f"\n{'─' * 70}")
        print(f"Severity profile: {severity_name}")
        print(f"Distribution: {severity_dist}")
        print(f"{'─' * 70}")

        for method_name, SimulatorClass in methods.items():
            current_run += 1
            print(f"  [{current_run}/{total_runs}] {method_name.title():12s} ... ", end="", flush=True)

            try:
                # Configure simulation
                config = SimulationConfig(
                    num_agents=NUM_AGENTS,
                    fault_rate=FAULT_RATE,
                    run_time=SIMULATION_TIME,
                    enable_quarantine=(method_name != "baseline"),
                    packet_loss_rate=0.00,
                    sensor_noise_level=0.0,
                    random_seed=42 + current_run,
                    scaling_mode=ScalingMode.FIXED_ARENA,
                )
                config.severity_distribution = severity_dist

                # Run simulation with data collection
                simulator = SimulatorClass(config)
                time_series = await collect_temporal_data(
                    simulator, severity_name, method_name
                )

                all_results.extend(time_series)
                print(f"✓ Collected {len(time_series)} samples")

            except Exception as e:
                print(f"✗ Error: {str(e)}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    print(f"\n{'─' * 70}")
    print("Saving results...")
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "fault_severity_response.csv"
    df.to_csv(output_path, index=False)

    print(f"✓ Saved {len(df)} data points to {output_path}")
    print(f"{'─' * 70}")

    return df


async def collect_temporal_data(
    simulator, severity_name: str, method_name: str
) -> List[Dict[str, Any]]:
    """Run simulation with high-frequency data collection.

    Parameters
    ----------
    simulator : BaseSimulator
        Fault detection simulator instance.
    severity_name : str
        Name of the severity distribution profile.
    method_name : str
        Name of the detection method.

    Returns
    -------
    time_series : list of dict
        Collected metrics at each timestep.

    Notes
    -----
    Samples system state every 0.1 seconds (10 Hz) during simulation.
    Metrics include task completion, healthy performance, quarantine count,
    and cascade damage indicators.
    """
    # Initialize simulation
    arena_size = simulator.calculate_arena_size()
    density = simulator.calculate_density(arena_size)

    simulator.create_agents()
    simulator.initialize_fault_detection(density)

    # Sampling configuration
    dt = 1.0 / SAMPLING_RATE
    num_steps = int(simulator.config.run_time / dt)

    time_series = []

    # Main simulation loop with data collection
    for step in range(num_steps + 1):
        current_time = step * dt

        # Collect current state
        metrics = extract_state_metrics(
            simulator, current_time, severity_name, method_name
        )
        time_series.append(metrics)

        # Update simulation (skip last step)
        if step < num_steps:
            simulator._update_agent_states(current_time)
            simulator._update_agent_interactions(density, current_time)

            if simulator.config.enable_quarantine:
                await simulator.update_fault_detection(current_time, density)

            await asyncio.sleep(0.001)  # Yield control

    return time_series


def extract_state_metrics(
    simulator, current_time: float, severity_name: str, method_name: str
) -> Dict[str, Any]:
    """Extract performance metrics from current simulation state.

    Parameters
    ----------
    simulator : BaseSimulator
        Simulator instance.
    current_time : float
        Current simulation time.
    severity_name : str
        Severity profile name.
    method_name : str
        Detection method name.

    Returns
    -------
    metrics : dict
        Collected performance metrics.
    """
    agents = simulator.agents

    # Agent categorization
    healthy_agents = [a for a in agents if not a.is_faulty]
    active_agents = [a for a in agents if not a.is_quarantined]
    active_healthy = [a for a in healthy_agents if not a.is_quarantined]

    # Task completion (system-wide)
    if active_agents:
        task_completion = sum(a.task_rate for a in active_agents) / len(agents)
    else:
        task_completion = 0.0

    # Healthy agent performance
    if active_healthy:
        healthy_performance = np.mean([a.task_rate for a in active_healthy])
    else:
        healthy_performance = 0.0

    # Quarantine count
    num_quarantined = sum(1 for a in agents if a.is_quarantined)

    # Cascade damage (fraction of healthy agents impaired)
    if healthy_agents:
        impaired = sum(1 for a in healthy_agents if a.task_rate < 0.8)
        cascade_damage = impaired / len(healthy_agents)
    else:
        cascade_damage = 0.0

    return {
        "method": method_name,
        "severity": severity_name,
        "time": current_time,
        "task_completion": task_completion,
        "healthy_performance": healthy_performance,
        "quarantine_count": num_quarantined,
        "cascade_damage": cascade_damage,
        "num_agents": len(agents),
        "num_healthy": len(healthy_agents),
        "num_active": len(active_agents),
        "num_faulty": len(agents) - len(healthy_agents),
    }


def analyze_results(df: pd.DataFrame) -> None:
    """Perform summary analysis of collected results.

    Parameters
    ----------
    df : DataFrame
        Collected experimental data.
    """
    print("\n" + "=" * 70)
    print("SUMMARY ANALYSIS")
    print("=" * 70)

    for method in df["method"].unique():
        print(f"\n{method.title()} Method:")
        method_data = df[df["method"] == method]

        for severity in df["severity"].unique():
            severity_data = method_data[method_data["severity"] == severity]

            if len(severity_data) > 0:
                # Extract final state
                final = severity_data.iloc[-1]

                print(f"  {severity}:")
                print(f"    Task completion:    {final['task_completion']:.3f}")
                print(f"    Healthy performance: {final['healthy_performance']:.3f}")
                print(f"    Quarantined:         {final['quarantine_count']:.0f}")
                print(f"    Cascade damage:      {final['cascade_damage']:.3f}")


def main():
    """Execute the fault severity response experiment."""
    print("=" * 70)
    print("FAULT SEVERITY RESPONSE EXPERIMENT")
    print("Evaluating adaptive capabilities across severity distributions")
    print("=" * 70)

    start_time = time.time()

    try:
        # Run experiment
        df = asyncio.run(collect_severity_response_data())

        # Analyze results
        if df is not None and len(df) > 0:
            analyze_results(df)

        elapsed = time.time() - start_time
        print(f"\nExecution time: {elapsed:.1f} seconds")
        print("Experiment complete.\n")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()