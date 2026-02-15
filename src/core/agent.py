"""
Agent implementation for swarm fault tolerance simulations.

This module provides the core agent class for multi-agent swarm systems,
including fault modeling, performance degradation, and self-diagnostics.
"""

from typing import Tuple, Optional, List
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum


class FaultType(Enum):
    """Agent fault severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class AgentState:
    """Snapshot of agent state at a given time."""
    position: np.ndarray
    velocity: np.ndarray
    task_rate: float
    stress_level: float
    is_quarantined: bool
    energy_consumed: float
    area_covered: float


class BaseAgent:
    """
    Base agent class for swarm fault tolerance experiments.

    Agents operate in a bounded 2D arena, maintaining state including position,
    velocity, task performance, and fault status. Faulty agents exhibit
    progressive self-degradation and may impact nearby healthy agents.

    Parameters
    ----------
    agent_id : int
        Unique identifier for this agent.
    position : Tuple[float, float]
        Initial (x, y) coordinates in the arena.
    fault_type : FaultType, default=FaultType.NONE
        Fault severity level.
    arena_size : float, default=100.0
        Size of the square arena.
    packet_loss_rate : float, default=0.0
        Probability of communication failure (0-1).
    sensor_noise_level : float, default=0.0
        Proportional noise added to sensor readings (0-1).

    Attributes
    ----------
    task_rate : float
        Current task completion efficiency (0-1).
    is_quarantined : bool
        Whether agent is quarantined from the swarm.
    energy_consumed : float
        Cumulative energy consumption (Joules).
    """

    ENERGY_IDLE = 0.1
    ENERGY_ACTIVE = 1.2

    def __init__(
        self,
        agent_id: int,
        position: Tuple[float, float],
        fault_type: FaultType = FaultType.NONE,
        arena_size: float = 100.0,
        packet_loss_rate: float = 0.0,
        sensor_noise_level: float = 0.0
    ):
        self.agent_id = agent_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.random.uniform(-2, 2, size=2).astype(np.float64)
        self.fault_type = fault_type
        self.is_faulty = fault_type != FaultType.NONE
        self.arena_size = arena_size
        self.packet_loss_rate = packet_loss_rate
        self.sensor_noise_level = sensor_noise_level

        self.base_performance_variation = np.random.uniform(0.85, 1.0)

        self._initialize_fault_parameters()
        self.is_quarantined = False
        self.quarantine_time: Optional[float] = None
        self.damage_taken = 0.0
        self.energy_consumed = 0.0
        self.cumulative_work = 0.0
        self.current_coverage_area = 0.0
        self.faulty_neighbors = 0
        self.time_near_faulty = 0.0
        self.time_away_from_faulty = 0.0
        self.stress_history: List[float] = []
        self.cumulative_exposure = 0.0

        self.degradation_onset_time = None
        self.is_degrading = False
        self.performance_history = []
        self.max_history_length = 10
        self.recent_trend = 0.0
        self.trend_window = 5

        self.self_diagnostic_signal = 0.0
        self.self_diag_ready_time = None

    def update_self_diagnostic(self, t_now: float, dt: float = 0.1,
                               enabled: bool = True,
                               tpr: float = 0.78, fpr: float = 0.06,
                               delay: float = 0.6,
                               response_speed: float = 0.4,
                               hi_range: tuple = (0.62, 0.72),
                               lo_range: tuple = (0.28, 0.42),
                               ewma_tau: float = 2.5):
        """
        Update self-diagnostic signal based on observed performance.

        Generates a smoothed diagnostic signal reflecting the agent's
        self-assessment of its health status. Signal generation accounts
        for detection delay and imperfect self-observation.

        Parameters
        ----------
        t_now : float
            Current simulation time.
        dt : float, default=0.1
            Time step for EWMA update.
        enabled : bool, default=True
            Whether diagnostics are active.
        tpr : float, default=0.78
            True positive rate for fault detection.
        fpr : float, default=0.06
            False positive rate.
        delay : float, default=0.6
            Time delay before reliable detection.
        response_speed : float, default=0.4
            Responsiveness of signal updates.
        hi_range : tuple, default=(0.62, 0.72)
            Range for high diagnostic signals.
        lo_range : tuple, default=(0.28, 0.42)
            Range for low diagnostic signals.
        ewma_tau : float, default=2.5
            Time constant for exponential smoothing.
        """
        if not enabled:
            return

        onset = getattr(self, "degradation_onset_time", None)
        performance_degraded = self.task_rate < 0.85

        if performance_degraded and onset is not None and t_now >= (onset + delay):
            severity_factor = max(0, 0.92 - self.task_rate)
            detection_prob = min(tpr, 0.6 + severity_factor * 2.0)

            if np.random.random() < detection_prob:
                target = np.random.uniform(hi_range[0], hi_range[1])
            else:
                target = np.random.uniform(0.45, 0.65)
        else:
            if not performance_degraded:
                if np.random.random() < fpr:
                    target = np.random.uniform(0.50, 0.70)
                else:
                    target = np.random.uniform(lo_range[0], lo_range[1])
            else:
                target = np.random.uniform(lo_range[0], lo_range[1])

        alpha = dt / (ewma_tau + dt)
        self.self_diagnostic_signal = (1 - alpha) * self.self_diagnostic_signal + alpha * target
        self.self_diagnostic_signal = np.clip(self.self_diagnostic_signal, 0.0, 1.0)

    def _initialize_fault_parameters(self) -> None:
        """Initialize fault-specific parameters and degradation timing."""
        self.task_rate = self.base_performance_variation
        self.base_task_rate = self.base_performance_variation
        self.stress_level = 0.0
        self.stress_output = 0.0
        self.emission_severity = 0.0
        self.self_degradation_rate = 0.0

        if self.fault_type == FaultType.NONE:
            self.self_degradation_delay = float('inf')
            self.outward_damage_delay = float('inf')

        elif self.fault_type == FaultType.MINOR:
            self.stress_output = 50.0
            self.emission_severity = 0.3
            self.self_degradation_rate = 0.015
            self.self_degradation_delay = np.random.uniform(0.5, 1.0)
            gap = np.random.uniform(0.8, 1.5)
            self.outward_damage_delay = self.self_degradation_delay + gap

        elif self.fault_type == FaultType.MODERATE:
            self.stress_output = 120.0
            self.emission_severity = 0.6
            self.self_degradation_rate = 0.035
            self.self_degradation_delay = np.random.uniform(0.5, 1.0)
            gap = np.random.uniform(0.8, 1.5)
            self.outward_damage_delay = self.self_degradation_delay + gap

        elif self.fault_type == FaultType.SEVERE:
            self.stress_output = 200.0
            self.emission_severity = 0.9
            self.self_degradation_rate = 0.05
            self.self_degradation_delay = np.random.uniform(0.3, 0.6)
            gap = np.random.uniform(0.5, 1.0)
            self.outward_damage_delay = self.self_degradation_delay + gap

        self.last_probe_time = -999
        self.probe_interval = 2.0
        self.diagnostic_tpr = {
            FaultType.NONE: 0.0,
            FaultType.MINOR: 0.60,
            FaultType.MODERATE: 0.80,
            FaultType.SEVERE: 0.95
        }
        self.diagnostic_fpr = 0.05

    def self_probe(self, current_time: float) -> bool:
        """
        Perform self-diagnostic probe with realistic detection rates.

        Parameters
        ----------
        current_time : float
            Current simulation time.

        Returns
        -------
        bool
            True if fault detected, accounting for TPR/FPR.
        """
        if current_time - self.last_probe_time < self.probe_interval:
            return False

        self.last_probe_time = current_time

        if self.is_faulty:
            detection_prob = self.diagnostic_tpr[self.fault_type]
            return np.random.random() < detection_prob
        else:
            return np.random.random() < self.diagnostic_fpr

    def observe_self_performance(self) -> float:
        """
        Get noisy self-observation of task performance.

        Returns
        -------
        float
            Observed task rate with added measurement noise.
        """
        noise = np.random.normal(0, 0.05)
        observed = self.task_rate + noise
        return np.clip(observed, 0.0, 1.0)

    def update_self_degradation(self, elapsed_time: float, dt: float = 0.1) -> None:
        """
        Apply progressive performance degradation for faulty agents.

        Parameters
        ----------
        elapsed_time : float
            Total elapsed simulation time.
        dt : float, default=0.1
            Time step size.
        """
        if not self.is_faulty or self.is_quarantined:
            return

        fault_start = getattr(self, 'fault_conversion_time', 0.0)
        time_since_fault = elapsed_time - fault_start

        if time_since_fault < self.self_degradation_delay:
            return

        self.task_rate -= self.self_degradation_rate * dt

    def update_position(self, dt: float = 0.1) -> None:
        """
        Update agent position with boundary reflection.

        Parameters
        ----------
        dt : float, default=0.1
            Time step for position update.
        """
        self.position += self.velocity * dt

        margin = 5.0
        for i in range(2):
            if self.position[i] < margin:
                self.position[i] = margin
                self.velocity[i] = abs(self.velocity[i])
            elif self.position[i] > self.arena_size - margin:
                self.position[i] = self.arena_size - margin
                self.velocity[i] = -abs(self.velocity[i])

    def take_damage(self, damage: float) -> None:
        """
        Apply cumulative performance degradation from external factors.

        Parameters
        ----------
        damage : float
            Amount of damage to accumulate.
        """
        if not self.is_faulty:
            self.damage_taken += damage
            self.task_rate = max(0.0, 1.0 - self.damage_taken)

    def apply_recovery(self, recovery_rate: float = 0.003) -> None:
        """
        Gradually recover performance when isolated from faults.

        Parameters
        ----------
        recovery_rate : float, default=0.003
            Rate of damage recovery per time step.
        """
        if not self.is_faulty and self.faulty_neighbors == 0:
            self.damage_taken = max(0.0, self.damage_taken - recovery_rate)
            self.task_rate = min(1.0, 1.0 - self.damage_taken)

    def can_transmit(self) -> bool:
        """Check if transmission succeeds given packet loss."""
        return random.random() > self.packet_loss_rate

    def can_receive(self) -> bool:
        """Check if reception succeeds given packet loss."""
        return random.random() > self.packet_loss_rate

    def can_communicate(self) -> bool:
        """Check if bidirectional communication succeeds."""
        return self.can_transmit() and self.can_receive()

    def sense_neighbor_stress(self, actual_stress: float) -> float:
        """
        Sense neighbor stress level with measurement noise.

        Parameters
        ----------
        actual_stress : float
            True stress level of neighbor.

        Returns
        -------
        float
            Noisy stress measurement, or 0 if communication fails.
        """
        if not self.can_communicate():
            return 0.0

        noise = np.random.normal(0, self.sensor_noise_level * actual_stress)
        sensed_stress = actual_stress + noise

        if actual_stress > 0:
            return np.clip(sensed_stress, 0.0, actual_stress * 2.0)
        else:
            return max(0.0, sensed_stress)

    def sense_distance(self, actual_distance: float) -> float:
        """
        Sense distance to neighbor with measurement noise.

        Parameters
        ----------
        actual_distance : float
            True distance to neighbor.

        Returns
        -------
        float
            Noisy distance measurement.
        """
        noise = np.random.normal(0, self.sensor_noise_level * actual_distance * 0.1)
        sensed_distance = actual_distance + noise
        return max(0.1, sensed_distance)

    def update_energy_and_coverage(self, dt: float = 0.1) -> None:
        """
        Update energy consumption and coverage area.

        Parameters
        ----------
        dt : float, default=0.1
            Time step size.
        """
        if self.is_quarantined:
            self.energy_consumed += self.ENERGY_IDLE * dt
        else:
            self.energy_consumed += self.ENERGY_ACTIVE * dt
            self.cumulative_work += self.task_rate * dt

        if not self.is_quarantined:
            coverage_radius = 10.0 * self.task_rate
            self.current_coverage_area = np.pi * coverage_radius ** 2
        else:
            self.current_coverage_area = 0.0

    def update_stress_history(self, stress: float) -> None:
        """Maintain rolling history of stress measurements."""
        self.stress_history.append(stress)
        if len(self.stress_history) > 20:
            self.stress_history.pop(0)

    def get_stress_persistence(self) -> float:
        """
        Calculate fraction of recent samples with elevated stress.

        Returns
        -------
        float
            Fraction of recent samples above stress threshold.
        """
        if len(self.stress_history) < 5:
            return 0.0

        stress_threshold = 50.0
        recent_samples = self.stress_history[-10:]
        stressed_samples = sum(1 for s in recent_samples if s > stress_threshold)
        return stressed_samples / len(recent_samples)

    def get_state(self) -> AgentState:
        """
        Get current agent state snapshot.

        Returns
        -------
        AgentState
            Immutable snapshot of current state.
        """
        return AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            task_rate=self.task_rate,
            stress_level=self.stress_level,
            is_quarantined=self.is_quarantined,
            energy_consumed=self.energy_consumed,
            area_covered=self.current_coverage_area
        )

    def distance_to(self, other: 'BaseAgent') -> float:
        """
        Compute Euclidean distance to another agent.

        Parameters
        ----------
        other : BaseAgent
            Target agent.

        Returns
        -------
        float
            Distance in arena units.
        """
        return np.linalg.norm(self.position - other.position)

    def update_degradation_tracking(self, current_time: float) -> None:
        """
        Track performance degradation onset and trends.

        Parameters
        ----------
        current_time : float
            Current simulation time.
        """
        self.performance_history.append(self.task_rate)
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)

        if self.task_rate < 0.98 and self.degradation_onset_time is None:
            self.degradation_onset_time = current_time
            self.is_degrading = True

        if len(self.performance_history) >= self.trend_window:
            mid = len(self.performance_history) // 2
            older_avg = sum(self.performance_history[:mid]) / mid
            recent_avg = sum(self.performance_history[mid:]) / (len(self.performance_history) - mid)
            self.recent_trend = recent_avg - older_avg

    def get_degradation_age(self, current_time: float) -> float:
        """
        Get time since degradation onset.

        Parameters
        ----------
        current_time : float
            Current simulation time.

        Returns
        -------
        float
            Time since onset, or 0 if not yet degrading.
        """
        if self.degradation_onset_time is None:
            return 0.0
        return current_time - self.degradation_onset_time

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Agent(id={self.agent_id}, "
            f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}), "
            f"fault={self.fault_type.value}, "
            f"task_rate={self.task_rate:.2f}, "
            f"quarantined={self.is_quarantined})"
        )
