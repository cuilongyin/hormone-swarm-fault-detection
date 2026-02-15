#!/usr/bin/env python3
"""Cascade Timeline Data Collection.

This module collects high-resolution temporal data showing how cascade failures
progress over time, with fair tracking that separates active damage from
quarantine costs. This enables rigorous analysis of detection method effectiveness.

The data collected is designed for Figure 3 visualization, showing:
- Active cascade damage (impaired healthy agents still in service)
- Quarantine cost (healthy agents removed from service)
- Combined system impact

Key Features
------------
- 10 Hz sampling for fine-grained temporal resolution
- Separate tracking of damage vs. quarantine effects
- First detection time markers for response analysis
- Detailed agent state tracking for cascade dynamics

Example
-------
To collect cascade timeline data:

    $ python cascade_timeline_collector.py

This generates cascade_timeline_fair.csv with time-series metrics for all methods.
"""

import asyncio
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulator import BaseSimulator, SimulationConfig
from src.methods.baseline import BaselineSimulator
from src.methods.hormone import LocalHormoneSimulator
from src.methods.threshold import ThresholdDetector
from src.methods.voting_simplified import SimpleVotingSimulator

# Experimental parameters
DEFAULT_NUM_AGENTS = 60
DEFAULT_FAULT_RATE = 0.5
DEFAULT_RUN_TIME = 30.0
DEFAULT_SEED = 42
SAMPLING_RATE = 10.0  # Hz


class TimelineCollector:
    """Wrapper for collecting cascade progression data during simulation.

    This class wraps a fault detection simulator to collect detailed temporal
    data during execution, including cascade metrics and quarantine events.

    Parameters
    ----------
    simulator : BaseSimulator
        The fault detection simulator to instrument.

    Attributes
    ----------
    simulator : BaseSimulator
        Wrapped simulator instance.
    timeline_data : list of dict
        Collected temporal metrics.
    first_quarantine_time : float or None
        Time of first quarantine action.
    quarantine_events : list of dict
        Record of all quarantine events.
    """

    def __init__(self, simulator: BaseSimulator):
        self.simulator = simulator
        self.timeline_data = []
        self.first_quarantine_time = None
        self.quarantine_events = []

    async def run_with_timeline_collection(self) -> Tuple[List[Dict], Optional[float], List[Dict]]:
        """Execute simulation with timeline data collection.

        Returns
        -------
        timeline_data : list of dict
            Time-series metrics sampled at 10 Hz.
        first_quarantine_time : float or None
            Time of first quarantine action, or None if no quarantines occurred.
        quarantine_events : list of dict
            Detailed records of all quarantine events.

        Notes
        -----
        This method runs the full simulation loop while collecting metrics
        at each timestep. It tracks both cascade damage (active impairment)
        and quarantine costs separately for fair comparison.
        """
        config = self.simulator.config
        arena_size = self.simulator.calculate_arena_size()
        density = self.simulator.calculate_density(arena_size)

        # Initialize simulation
        self.simulator.create_agents()
        self.simulator.initialize_fault_detection(density)

        # Simulation loop with data collection
        dt = 1.0 / SAMPLING_RATE
        num_steps = int(config.run_time * SAMPLING_RATE)

        for step in range(num_steps):
            current_time = step * dt

            # Update agent states
            self.simulator._update_agent_states(current_time)
            self.simulator._update_agent_interactions(density, current_time)

            # Track quarantines before detection update
            num_quarantined_before = sum(1 for a in self.simulator.agents if a.is_quarantined)

            # Update fault detection
            if config.enable_quarantine:
                await self.simulator.update_fault_detection(current_time, density)

            # Check for new quarantines
            num_quarantined_after = sum(1 for a in self.simulator.agents if a.is_quarantined)

            if num_quarantined_after > num_quarantined_before:
                if self.first_quarantine_time is None:
                    self.first_quarantine_time = current_time

                # Record new quarantine events
                self._record_quarantine_events(current_time)

            # Collect cascade metrics
            metrics = self._extract_cascade_metrics(current_time)
            self.timeline_data.append(metrics)

            await asyncio.sleep(0.001)  # Yield control

        return self.timeline_data, self.first_quarantine_time, self.quarantine_events

    def _record_quarantine_events(self, current_time: float) -> None:
        """Record details of newly quarantined agents.

        Parameters
        ----------
        current_time : float
            Current simulation time.
        """
        for agent in self.simulator.agents:
            if agent.is_quarantined:
                # Check if not already recorded
                already_recorded = any(
                    e["agent_id"] == agent.agent_id for e in self.quarantine_events
                )

                if not already_recorded:
                    self.quarantine_events.append({
                        "time": current_time,
                        "agent_id": agent.agent_id,
                        "is_faulty": agent.is_faulty,
                        "task_rate": agent.task_rate,
                    })

    def _extract_cascade_metrics(self, current_time: float) -> Dict:
        """Extract cascade metrics with fair damage/quarantine separation.

        Parameters
        ----------
        current_time : float
            Current simulation time.

        Returns
        -------
        metrics : dict
            Cascade metrics at current timestep.

        Notes
        -----
        This method separates three types of impact:
        1. Active damage: Healthy agents still active but impaired (task_rate < 0.8)
        2. Quarantine cost: Healthy agents removed from service
        3. Total affected: Sum of the above

        This separation enables fair comparison between methods that quarantine
        aggressively vs. those that rely on cascade containment.
        """
        originally_healthy = [a for a in self.simulator.agents if not a.is_faulty]

        if not originally_healthy:
            return self._empty_metrics(current_time)

        # Categorize healthy agents
        damaged_active = 0  # Active but impaired
        quarantined_healthy = 0  # Healthy but removed
        severely_damaged = 0  # Severely impaired (task_rate < 0.5)

        for agent in originally_healthy:
            if agent.is_quarantined:
                quarantined_healthy += 1
            elif agent.task_rate < 0.8:
                damaged_active += 1
                if agent.task_rate < 0.5:
                    severely_damaged += 1

        # Compute fractions
        total_healthy = len(originally_healthy)
        fraction_damaged = damaged_active / total_healthy
        fraction_quarantined = quarantined_healthy / total_healthy
        fraction_total_affected = fraction_damaged + fraction_quarantined
        fraction_severe = severely_damaged / total_healthy

        # Average performance of active healthy agents
        active_healthy = [a for a in originally_healthy if not a.is_quarantined]
        avg_healthy_performance = (
            np.mean([a.task_rate for a in active_healthy]) if active_healthy else 0.0
        )

        return {
            "time": current_time,
            "fraction_damaged": fraction_damaged,
            "fraction_quarantined": fraction_quarantined,
            "fraction_total_affected": fraction_total_affected,
            "fraction_severe": fraction_severe,
            "num_quarantined": sum(1 for a in self.simulator.agents if a.is_quarantined),
            "num_faulty": sum(1 for a in self.simulator.agents if a.is_faulty),
            "num_healthy": total_healthy,
            "avg_healthy_performance": avg_healthy_performance,
        }

    def _empty_metrics(self, current_time: float) -> Dict:
        """Return empty metrics structure for edge cases.

        Parameters
        ----------
        current_time : float
            Current simulation time.

        Returns
        -------
        metrics : dict
            Metrics with all values set to zero.
        """
        return {
            "time": current_time,
            "fraction_damaged": 0.0,
            "fraction_quarantined": 0.0,
            "fraction_total_affected": 0.0,
            "fraction_severe": 0.0,
            "num_quarantined": 0,
            "num_faulty": 0,
            "num_healthy": 0,
            "avg_healthy_performance": 0.0,
        }


async def collect_cascade_timelines(
        num_agents: int = DEFAULT_NUM_AGENTS,
        fault_rate: float = DEFAULT_FAULT_RATE,
        run_time: float = DEFAULT_RUN_TIME,
        seed: int = DEFAULT_SEED,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Optional[float]]]:
    """Collect cascade timeline data for all detection methods.

    Parameters
    ----------
    num_agents : int, optional
        Number of agents in swarm (default: 60).
    fault_rate : float, optional
        Fraction of faulty agents (default: 0.5).
    run_time : float, optional
        Simulation duration in seconds (default: 30.0).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    timeline_data : dict of str to list of dict
        Timeline metrics for each method.
    detection_times : dict of str to float or None
        First detection time for each method.

    Notes
    -----
    Runs a single representative trial for each method using the same
    random seed to ensure comparable fault scenarios.
    """
    print(f"Collecting cascade timeline data ({num_agents} agents, "
          f"{fault_rate:.0%} fault rate)")
    print("=" * 70)

    methods = [
        ("Baseline", BaselineSimulator),
        ("Threshold", ThresholdDetector),
        ("Voting", SimpleVotingSimulator),
        ("Hormone", LocalHormoneSimulator),
    ]

    timeline_data = {}
    detection_times = {}

    for method_name, method_class in methods:
        print(f"\n{method_name}:")

        # Configure simulation
        config = SimulationConfig(
            num_agents=num_agents,
            fault_rate=fault_rate,
            run_time=run_time,
            enable_quarantine=(method_name != "Baseline"),
            random_seed=seed,
        )

        # Create simulator with timeline collector
        simulator = method_class(config)
        collector = TimelineCollector(simulator)

        # Run with data collection
        timeline, first_detection, events = await collector.run_with_timeline_collection()

        # Store results
        timeline_data[method_name] = timeline
        detection_times[method_name] = first_detection

        # Print summary
        if timeline:
            max_damaged = max(p["fraction_damaged"] for p in timeline)
            max_quarantined = max(p["fraction_quarantined"] for p in timeline)
            max_total = max(p["fraction_total_affected"] for p in timeline)
            final_damaged = timeline[-1]["fraction_damaged"]
            final_quarantined = timeline[-1]["fraction_quarantined"]

            print(f"  Peak active damage:    {max_damaged:.1%}")
            print(f"  Peak quarantined:      {max_quarantined:.1%}")
            print(f"  Peak total affected:   {max_total:.1%}")
            print(f"  Final active damage:   {final_damaged:.1%}")
            print(f"  Final quarantined:     {final_quarantined:.1%}")

            if first_detection is not None:
                print(f"  First detection:       {first_detection:.2f}s")
            else:
                print(f"  First detection:       None")

    return timeline_data, detection_times


def save_timeline_csv(
        timeline_data: Dict[str, List[Dict]],
        detection_times: Dict[str, Optional[float]],
        filename: str = "cascade_timeline_fair.csv",
) -> Path:
    """Save timeline data to CSV format.

    Parameters
    ----------
    timeline_data : dict of str to list of dict
        Timeline metrics for each method.
    detection_times : dict of str to float or None
        First detection time for each method.
    filename : str, optional
        Output filename (default: 'cascade_timeline_fair.csv').

    Returns
    -------
    filepath : Path
        Path to saved CSV file.
    """
    # Ensure results directory exists
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename

    print(f"\nSaving timeline data to {filepath}...")

    # Prepare rows for CSV
    rows = []
    for method, timeline in timeline_data.items():
        first_detection = detection_times.get(method)

        for point in timeline:
            row = {
                "method": method,
                "time": point["time"],
                "fraction_damaged": point["fraction_damaged"],
                "fraction_quarantined": point["fraction_quarantined"],
                "fraction_total_affected": point["fraction_total_affected"],
                "fraction_severe": point["fraction_severe"],
                "num_quarantined": point["num_quarantined"],
                "avg_healthy_performance": point["avg_healthy_performance"],
                "first_detection_time": first_detection if first_detection is not None else -1,
            }
            rows.append(row)

    # Write to CSV
    if rows:
        with open(filepath, "w", newline="") as f:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Saved {len(rows)} timeline points")

    return filepath


def analyze_cascade_dynamics(
        timeline_data: Dict[str, List[Dict]], detection_times: Dict[str, Optional[float]]
) -> None:
    """Analyze and print cascade dynamics insights.

    Parameters
    ----------
    timeline_data : dict of str to list of dict
        Timeline metrics for each method.
    detection_times : dict of str to float or None
        First detection time for each method.
    """
    print("\n" + "=" * 70)
    print("CASCADE DYNAMICS ANALYSIS")
    print("=" * 70)

    for method in ["Baseline", "Threshold", "Voting", "Hormone"]:
        if method not in timeline_data:
            continue

        timeline = timeline_data[method]
        first_detection = detection_times.get(method)

        print(f"\n{method}:")

        # Cascade onset (damage exceeds 5%)
        cascade_onset = None
        for point in timeline:
            if point["fraction_damaged"] > 0.05:
                cascade_onset = point["time"]
                break

        # Peak metrics
        peak_damage = max(p["fraction_damaged"] for p in timeline)
        peak_time = next(p["time"] for p in timeline if p["fraction_damaged"] == peak_damage)
        peak_quarantined = max(p["fraction_quarantined"] for p in timeline)
        peak_total = max(p["fraction_total_affected"] for p in timeline)

        # Cascade velocity
        if cascade_onset and peak_time > cascade_onset:
            velocity = peak_damage / (peak_time - cascade_onset)
        else:
            velocity = 0.0

        # Print analysis
        if cascade_onset:
            print(f"  Cascade onset:         {cascade_onset:.2f}s")
        else:
            print(f"  Cascade onset:         Not detected")

        print(f"  Peak active damage:    {peak_damage:.1%} at {peak_time:.1f}s")
        print(f"  Peak quarantined:      {peak_quarantined:.1%}")
        print(f"  Peak total impact:     {peak_total:.1%}")
        print(f"  Damage velocity:       {velocity:.3f}/s")

        if first_detection:
            print(f"  First detection:       {first_detection:.2f}s")
            if cascade_onset:
                delay = first_detection - cascade_onset
                print(f"  Detection delay:       {delay:.2f}s after cascade onset")

        # Area under curve (cumulative impact)
        if timeline:
            damage_auc = sum(p["fraction_damaged"] * 0.1 for p in timeline)
            total_auc = sum(p["fraction_total_affected"] * 0.1 for p in timeline)
            print(f"  Cumulative damage:     {damage_auc:.2f}")
            print(f"  Cumulative total:      {total_auc:.2f}")


async def main():
    """Execute cascade timeline collection and analysis."""
    print("=" * 70)
    print("CASCADE TIMELINE DATA COLLECTION")
    print("Fair tracking with separate damage/quarantine metrics")
    print("=" * 70)

    # Collect data
    timeline_data, detection_times = await collect_cascade_timelines()

    # Save to CSV
    csv_path = save_timeline_csv(timeline_data, detection_times)

    # Analyze dynamics
    analyze_cascade_dynamics(timeline_data, detection_times)

    # Summary
    print("\n" + "=" * 70)
    print("Timeline data collection complete.")
    print(f"Data saved to: {csv_path}")
    print("\nNext steps:")
    print("1. Run generate_figures.py to create Figure 3")
    print("2. Figure will show active damage vs. quarantine cost")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())