"""
Base simulator framework for swarm fault tolerance experiments.

Provides abstract infrastructure for multi-agent simulations with
configurable fault injection patterns, detection strategies, and
performance metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import asyncio
import random
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from src.core.agent import BaseAgent, FaultType


class ScalingMode(Enum):
    """Arena scaling strategies for varying swarm sizes."""
    FIXED_ARENA = "fixed_arena"
    FIXED_DENSITY = "fixed_density"


@dataclass
class SimulationConfig:
    """
    Configuration parameters for simulation experiments.

    Parameters
    ----------
    num_agents : int
        Number of agents in the swarm.
    fault_rate : float
        Fraction of agents initially faulty (0-1).
    run_time : float, default=60.0
        Simulation duration in seconds.
    scaling_mode : ScalingMode, default=FIXED_ARENA
        How arena size scales with agent count.
    enable_quarantine : bool, default=True
        Whether quarantine mechanism is active.
    packet_loss_rate : float, default=0.0
        Communication packet loss probability (0-1).
    sensor_noise_level : float, default=0.0
        Proportional sensor noise (0-1).
    random_seed : int or None, default=None
        Random seed for reproducibility.
    diag_scenario : str, default="rosy"
        Diagnostic realism level: "rosy", "realistic", or "adversarial".
    fault_wave_mode : str, default="static"
        Fault injection pattern: "static", "waves", or "cascading".
    wave_parameters : dict or None
        Wave timing and size specifications.
    use_ratio_conditioned_threshold : bool, default=True
        Whether to use adaptive thresholds.
    """
    num_agents: int
    fault_rate: float
    run_time: float = 60.0
    scaling_mode: ScalingMode = ScalingMode.FIXED_ARENA
    enable_quarantine: bool = True
    packet_loss_rate: float = 0.0
    sensor_noise_level: float = 0.0
    random_seed: Optional[int] = None
    diag_scenario: str = "rosy"
    fault_wave_mode: str = "static"
    wave_parameters: Optional[Dict] = None
    use_ratio_conditioned_threshold: bool = True


@dataclass
class SimulationMetrics:
    """
    Comprehensive metrics from a simulation run.

    Attributes
    ----------
    num_agents : int
        Swarm size.
    fault_rate : float
        Proportion of initially faulty agents.
    task_completion : float
        Average task rate across swarm.
    quarantine_efficiency : float
        Fraction of faulty agents quarantined.
    cascade_prevention : float
        Cascade mitigation effectiveness.
    work_per_energy : float
        Energy efficiency metric.
    """
    num_agents: int
    fault_rate: float
    arena_size: float
    density: float
    scaling_mode: str
    task_completion: float
    avg_healthy_performance: float
    mission_success: bool
    severely_impaired: int
    num_healthy: int
    num_faulty: int
    quarantined: int
    quarantine_efficiency: float
    cascade_prevention: float
    avg_response_time: float
    work_per_energy: float
    coverage_efficiency: float
    healthy_preservation_rate: float
    active_agent_efficiency: float
    packet_loss_rate: float
    sensor_noise_level: float
    quarantine_enabled: bool


class BaseSimulator(ABC):
    """
    Abstract base class for swarm simulations.

    Provides core simulation infrastructure including agent initialization,
    spatial dynamics, fault injection, and metrics collection. Subclasses
    implement specific fault detection strategies.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration parameters.

    Attributes
    ----------
    agents : list of BaseAgent
        All agents in the swarm.
    config : SimulationConfig
        Configuration parameters.
    """

    BASE_ARENA_SIZE = 100.0
    BASE_DENSITY = 0.005
    COMMUNICATION_RANGE = 25.0
    IMPEDANCE_RANGE = 15.0

    def __init__(self, config: SimulationConfig):
        self.config = config

        self.diag_cfg = {
            "enabled": True,
            "tpr": 0.80,
            "fpr": 0.08,
            "delay": 0.6,
            "response_speed": 0.5,
            "hi_range": (0.65, 0.80),
            "lo_range": (0.20, 0.35),
            "ewma_tau": 2.0,
        }

        if hasattr(config, 'diag_scenario'):
            if config.diag_scenario == "realistic":
                self.diag_cfg.update({
                    "tpr": 0.55,
                    "fpr": 0.18,
                    "delay": 1.8,
                    "response_speed": 0.25
                })
            elif config.diag_scenario == "adversarial":
                self.diag_cfg.update({
                    "tpr": 0.40,
                    "fpr": 0.28,
                    "delay": 3.0,
                    "response_speed": 0.18
                })

        self.USE_STRESS_SYSTEM = False
        self.USE_DAMAGE_SYSTEM = True
        self.USE_HORMONE_SYSTEM = True

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.agents: List[BaseAgent] = []

    def calculate_arena_size(self) -> float:
        """
        Determine arena dimensions based on scaling mode.

        Returns
        -------
        float
            Arena side length.
        """
        if hasattr(self.config, "arena_size_override") and self.config.arena_size_override:
            return float(self.config.arena_size_override)

        if self.config.scaling_mode == ScalingMode.FIXED_ARENA:
            return self.BASE_ARENA_SIZE
        else:
            area_needed = self.config.num_agents / self.BASE_DENSITY
            return float(np.sqrt(area_needed))

    def calculate_density(self, arena_size: float) -> float:
        """
        Compute swarm density.

        Parameters
        ----------
        arena_size : float
            Arena side length.

        Returns
        -------
        float
            Agents per square unit.
        """
        area = arena_size * arena_size
        return self.config.num_agents / area

    def create_agents(self) -> List[BaseAgent]:
        """
        Initialize agent population with spatial distribution.

        Creates agents in grid formation with position jitter and
        applies fault injection according to configuration.

        Returns
        -------
        list of BaseAgent
            Initialized agent population.
        """
        arena_size = self.calculate_arena_size()

        grid_size = int(np.ceil(np.sqrt(self.config.num_agents)))
        spacing = arena_size / (grid_size - 1) if grid_size > 1 else arena_size

        agents = []
        for i in range(self.config.num_agents):
            row = i // grid_size
            col = i % grid_size

            x_pos = np.clip(col * spacing + random.uniform(-5, 5), 10, arena_size - 10)
            y_pos = np.clip(row * spacing + random.uniform(-5, 5), 10, arena_size - 10)
            position = (x_pos, y_pos)

            agent = self._create_agent(i, position, FaultType.NONE, arena_size)
            agents.append(agent)

        if self.config.fault_wave_mode == "static":
            self._apply_initial_faults(agents)
        else:
            self._prepare_fault_assignments()

        self.agents = agents
        return agents

    def _apply_initial_faults(self, agents: List[BaseAgent]) -> None:
        """Apply faults to agents at simulation start."""
        num_faulty = int(self.config.num_agents * self.config.fault_rate)
        fault_assignments = self._assign_fault_types(num_faulty)
        fault_indices = set(random.sample(range(len(agents)), num_faulty))

        fault_counter = 0
        for i, agent in enumerate(agents):
            if i in fault_indices:
                fault_type = fault_assignments[fault_counter]
                fault_counter += 1

                agent.fault_type = fault_type
                agent.is_faulty = True

                if hasattr(agent, '_initialize_fault_parameters'):
                    agent._initialize_fault_parameters()

                if hasattr(self.config, "timing_regime"):
                    gmin, gmax = self.config.timing_regime
                    gap = np.random.uniform(gmin, gmax)
                    agent.outward_damage_delay = float(agent.self_degradation_delay + gap)

    def _prepare_fault_assignments(self) -> None:
        """Prepare fault assignments for wave-based injection."""
        num_faulty = int(self.config.num_agents * self.config.fault_rate)
        self.pending_fault_assignments = self._assign_fault_types(num_faulty)
        self.pending_fault_indices = []
        self.faults_introduced = 0
        self.next_wave_index = 0

    def introduce_fault_wave(self, elapsed_time: float) -> None:
        """
        Inject faults in waves according to schedule.

        Parameters
        ----------
        elapsed_time : float
            Current simulation time.
        """
        if not self.config.wave_parameters:
            return

        wave_times = self.config.wave_parameters.get('wave_times', [])
        wave_sizes = self.config.wave_parameters.get('wave_sizes', [])

        if len(wave_times) != len(wave_sizes):
            return

        if not hasattr(self, 'next_wave_index'):
            self.next_wave_index = 0

        if self.next_wave_index >= len(wave_times):
            return

        if elapsed_time >= wave_times[self.next_wave_index]:
            wave_size = wave_sizes[self.next_wave_index]
            num_to_convert = int(self.config.num_agents * wave_size)

            healthy = [a for a in self.agents if not a.is_faulty and not a.is_quarantined]

            if len(healthy) >= num_to_convert:
                if self.config.fault_wave_mode == "cascading":
                    selected = self._select_cascading_targets(healthy, num_to_convert)
                else:
                    selected = random.sample(healthy, num_to_convert)

                for agent in selected:
                    self._convert_agent_to_faulty(agent, elapsed_time)

            self.next_wave_index += 1

    def _select_cascading_targets(self, healthy_agents: List[BaseAgent], num_to_convert: int) -> List[BaseAgent]:
        """Select agents near existing faults for cascading spread."""
        faulty_agents = [a for a in self.agents if a.is_faulty and not a.is_quarantined]

        if not faulty_agents:
            return random.sample(healthy_agents, min(num_to_convert, len(healthy_agents)))

        proximity_scores = []
        for healthy in healthy_agents:
            min_dist = min(healthy.distance_to(faulty) for faulty in faulty_agents)
            proximity_scores.append((min_dist, healthy))

        proximity_scores.sort(key=lambda x: x[0])
        return [agent for _, agent in proximity_scores[:num_to_convert]]

    def _convert_agent_to_faulty(self, agent: BaseAgent, conversion_time: float) -> None:
        """Convert healthy agent to faulty at runtime."""
        if self.faults_introduced < len(self.pending_fault_assignments):
            fault_type = self.pending_fault_assignments[self.faults_introduced]
            self.faults_introduced += 1
        else:
            fault_type = random.choice([FaultType.MINOR, FaultType.MODERATE, FaultType.SEVERE])

        agent.fault_type = fault_type
        agent.is_faulty = True
        agent.fault_conversion_time = conversion_time

        if hasattr(agent, '_initialize_fault_parameters'):
            agent._initialize_fault_parameters()

        if hasattr(self, '_final_stats_computed'):
            self._final_stats_computed = False

        if not hasattr(self, 'fault_conversion_log'):
            self.fault_conversion_log = []
        self.fault_conversion_log.append({
            'time': conversion_time,
            'agent_id': agent.agent_id,
            'fault_type': fault_type.value
        })

    def _assign_fault_types(self, num_faulty: int) -> List[FaultType]:
        """
        Assign fault severity levels with realistic distribution.

        Parameters
        ----------
        num_faulty : int
            Number of faults to assign.

        Returns
        -------
        list of FaultType
            Shuffled list of fault assignments.
        """
        if num_faulty == 0:
            return []

        dist = getattr(self.config, "severity_distribution", None)
        if dist:
            p_minor = float(dist.get("minor", 0.60))
            p_moderate = float(dist.get("moderate", 0.30))
            n_minor = int(round(num_faulty * p_minor))
            n_mod = int(round(num_faulty * p_moderate))
            n_severe = max(0, num_faulty - n_minor - n_mod)
        else:
            n_severe = max(1, int(num_faulty * 0.15))
            n_mod = int(num_faulty * 0.35)
            n_minor = num_faulty - n_severe - n_mod

        lst = ([FaultType.SEVERE] * n_severe +
               [FaultType.MODERATE] * n_mod +
               [FaultType.MINOR] * n_minor)
        random.shuffle(lst)
        return lst

    def _create_agent(
            self,
            agent_id: int,
            position: Tuple[float, float],
            fault_type: FaultType,
            arena_size: float
    ) -> BaseAgent:
        """Create single agent instance with specified parameters."""
        return BaseAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )

    async def run_simulation(self, baseline_metrics=None):
        """
        Execute complete simulation run.

        Performs time-stepped simulation with agent updates, fault detection,
        and metrics collection. Tracks cascade progression for analysis.

        Parameters
        ----------
        baseline_metrics : SwarmMetrics or None
            Baseline metrics instance for counterfactual comparison.

        Returns
        -------
        SimulationMetrics
            Final metrics from the simulation.
        """
        arena_size = self.calculate_arena_size()
        density = self.calculate_density(arena_size)

        self.create_agents()

        if hasattr(self, 'ABLATE_PRECEDENCE'):
            for agent in self.agents:
                if hasattr(agent, 'precedence_enabled'):
                    agent.precedence_enabled = not self.ABLATE_PRECEDENCE

        self.initialize_fault_detection(density)

        steps = int(self.config.run_time * 10)

        self.peak_cascade_damage = 0.0
        self.cascade_at_first_quarantine = None
        self.first_quarantine_time = None
        self.cascade_measurements = []

        from src.core.metrics import SwarmMetrics
        metrics_tracker = SwarmMetrics()

        if not self.config.enable_quarantine:
            metrics_tracker.set_as_baseline(True)

        for step in range(steps):
            simulation_time = step / 10.0

            metrics_tracker.track_agent_states(self.agents, simulation_time)

            self._update_agent_states(simulation_time)
            self._update_agent_interactions(density, simulation_time)

            if self.config.fault_wave_mode in ["waves", "cascading"]:
                self.introduce_fault_wave(simulation_time)

            metrics_tracker.track_agent_states(self.agents, simulation_time)

            healthy_agents = [a for a in self.agents if not a.is_faulty]
            if healthy_agents:
                impaired = sum(1 for a in healthy_agents if a.task_rate < 0.8)
                current_cascade_damage = impaired / len(healthy_agents)
                self.peak_cascade_damage = max(self.peak_cascade_damage, current_cascade_damage)
                self.cascade_measurements.append({
                    'time': simulation_time,
                    'cascade_damage': current_cascade_damage,
                    'impaired_count': impaired
                })

            quarantined_before = sum(1 for a in self.agents if a.is_quarantined)

            if self.config.enable_quarantine:
                await self.update_fault_detection(simulation_time, density)

            quarantined_after = sum(1 for a in self.agents if a.is_quarantined)

            if quarantined_after > quarantined_before and self.first_quarantine_time is None:
                self.first_quarantine_time = simulation_time
                if healthy_agents:
                    impaired_now = sum(1 for a in healthy_agents if a.task_rate < 0.8)
                    self.cascade_at_first_quarantine = 1.0 - (impaired_now / len(healthy_agents))
                else:
                    self.cascade_at_first_quarantine = 0.0

            await asyncio.sleep(0.001)

        metrics_tracker.track_agent_states(self.agents, self.config.run_time)

        metrics = self._calculate_metrics(arena_size, density)
        final_metrics = metrics_tracker.calculate_all_metrics(
            self.agents,
            self.config.run_time,
            baseline_metrics=baseline_metrics
        )

        if not self.config.enable_quarantine:
            metrics.cascade_prevention = 0.0
        else:
            if baseline_metrics is not None:
                metrics.cascade_prevention = final_metrics['counterfactual_cascade_prevention']
            else:
                metrics.cascade_prevention = 1.0 - self.peak_cascade_damage

        self.metrics_tracker = metrics_tracker

        if self.config.enable_quarantine:
            self._finalize_detection_stats()

        return metrics

    def _finalize_detection_stats(self):
        """Finalize confusion matrix at simulation end."""
        if not hasattr(self, 'detection_stats'):
            self.detection_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        if not hasattr(self, '_final_stats_computed'):
            self._final_stats_computed = False

        if self._final_stats_computed:
            return

        quarantined_faulty = set()
        quarantined_healthy = set()
        unquarantined_faulty = set()
        unquarantined_healthy = set()

        for agent in self.agents:
            if agent.is_quarantined:
                if agent.is_faulty:
                    quarantined_faulty.add(agent.agent_id)
                else:
                    quarantined_healthy.add(agent.agent_id)
            else:
                if agent.is_faulty:
                    unquarantined_faulty.add(agent.agent_id)
                else:
                    unquarantined_healthy.add(agent.agent_id)

        self.detection_stats = {
            'tp': len(quarantined_faulty),
            'fp': len(quarantined_healthy),
            'fn': len(unquarantined_faulty),
            'tn': len(unquarantined_healthy)
        }

        self._final_stats_computed = True

    def _update_agent_states(self, elapsed_time: float = 0.0) -> None:
        """
        Update all agent states for current timestep.

        Parameters
        ----------
        elapsed_time : float
            Current simulation time.
        """
        for agent in self.agents:
            if not agent.is_quarantined:
                agent.update_position()

            agent.update_energy_and_coverage()

            if hasattr(agent, 'update_self_degradation'):
                agent.update_self_degradation(elapsed_time)

            if hasattr(agent, 'update_degradation_tracking'):
                agent.update_degradation_tracking(elapsed_time)

            if hasattr(agent, 'update_self_diagnostic'):
                agent.update_self_diagnostic(
                    t_now=elapsed_time,
                    dt=0.1,
                    enabled=self.diag_cfg.get("enabled", True),
                    tpr=self.diag_cfg.get("tpr", 0.78),
                    fpr=self.diag_cfg.get("fpr", 0.06),
                    delay=self.diag_cfg.get("delay", 0.6),
                    response_speed=self.diag_cfg.get("response_speed", 0.4),
                    hi_range=self.diag_cfg.get("hi_range", (0.62, 0.72)),
                    lo_range=self.diag_cfg.get("lo_range", (0.28, 0.42)),
                    ewma_tau=self.diag_cfg.get("ewma_tau", 2.5)
                )

    def _update_agent_interactions(self, density: float, elapsed_time: float = 0.0) -> None:
        """
        Process agent-to-agent interactions and damage propagation.

        Implements exposure-based damage accumulation with spatial indexing
        for efficiency. Handles both primary fault emissions and secondary
        damage from degraded agents.

        Parameters
        ----------
        density : float
            Current swarm density.
        elapsed_time : float
            Current simulation time.
        """
        params = self.get_adaptive_parameters(density)

        def adaptive_impedance_radius_3d_partial(density, rho0=0.005, r0=15.0, g=0.6, rmin=3.0, rmax=25.0):
            if density <= 0:
                return rmax
            r = r0 * (density / rho0) ** (-g / 3.0)
            return float(np.clip(r, rmin, rmax))

        effective_impedance = adaptive_impedance_radius_3d_partial(density)

        for agent in self.agents:
            agent.faulty_neighbors = 0

        cell_size = max(self.COMMUNICATION_RANGE, effective_impedance)
        grid = defaultdict(list)

        for agent in self.agents:
            if not agent.is_quarantined:
                cell_x = int(agent.position[0] / cell_size)
                cell_y = int(agent.position[1] / cell_size)
                grid[(cell_x, cell_y)].append(agent)

        def get_nearby_agents(agent, max_range):
            nearby = []
            cell_x = int(agent.position[0] / cell_size)
            cell_y = int(agent.position[1] / cell_size)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell_key = (cell_x + dx, cell_y + dy)
                    if cell_key in grid:
                        for other in grid[cell_key]:
                            if other.agent_id != agent.agent_id:
                                dist = agent.distance_to(other)
                                if dist <= max_range:
                                    nearby.append((other, dist))
            return nearby

        if self.USE_DAMAGE_SYSTEM:
            dt = 0.1
            EXPOSURE_THRESHOLD = 0.5

            for agent in self.agents:
                if agent.is_quarantined or agent.is_faulty:
                    continue

                nearby = get_nearby_agents(agent, effective_impedance)

                total_exposure = 0.0
                faulty_neighbor_count = 0

                for other, actual_dist in nearby:
                    if other.is_quarantined:
                        continue

                    if other.is_faulty:
                        fault_start = getattr(other, 'fault_conversion_time', 0.0)
                        time_since_fault = elapsed_time - fault_start

                        if time_since_fault < other.outward_damage_delay:
                            continue

                        if not hasattr(other, 'degradation_onset_time') or other.degradation_onset_time is None:
                            continue

                        agent.faulty_neighbors += 1
                        faulty_neighbor_count += 1

                        severity = getattr(other, 'emission_severity', 0.5)
                        proximity = np.exp(-actual_dist / 10.0)
                        exposure = severity * proximity * dt * 1.0

                        if self.config.enable_quarantine and not agent.can_receive():
                            exposure *= 1.2

                        total_exposure += exposure

                    elif other.task_rate < 0.6:
                        degradation_level = 1.0 - other.task_rate
                        secondary_severity = degradation_level * 0.1
                        proximity = np.exp(-actual_dist / 10.0)
                        secondary_exposure = secondary_severity * proximity * dt * 0.5
                        total_exposure += secondary_exposure

                if faulty_neighbor_count > 1:
                    density_multiplier = 1.0 + (faulty_neighbor_count - 1) * 0.15
                    total_exposure *= density_multiplier

                if not hasattr(agent, 'cumulative_exposure'):
                    agent.cumulative_exposure = 0.0

                agent.cumulative_exposure += total_exposure

                if agent.cumulative_exposure > EXPOSURE_THRESHOLD:
                    excess_exposure = agent.cumulative_exposure - EXPOSURE_THRESHOLD

                    if excess_exposure < 0.5:
                        damage_rate = 0.1 * excess_exposure
                    elif excess_exposure < 1.5:
                        damage_rate = 0.05 + 0.1 * excess_exposure
                    else:
                        damage_rate = 0.1

                    agent.damage_taken += damage_rate
                    agent.task_rate = max(0.0, 1.0 - agent.damage_taken)

                    agent.time_near_faulty += dt
                    agent.time_away_from_faulty = 0
                else:
                    agent.time_away_from_faulty += dt
                    agent.time_near_faulty = 0

        for agent in self.agents:
            if not agent.is_faulty and agent.faulty_neighbors == 0:
                agent.time_away_from_faulty += 0.1

                if agent.time_away_from_faulty > 2.0:
                    recovery_rate = 0.003

                    if self.config.enable_quarantine and self.config.packet_loss_rate > 0:
                        recovery_rate *= (1.0 - self.config.packet_loss_rate * 0.5)

                    agent.damage_taken = max(0, agent.damage_taken - recovery_rate)
                    agent.task_rate = min(1.0, 1.0 - agent.damage_taken)

                    if hasattr(agent, 'cumulative_exposure'):
                        agent.cumulative_exposure = max(0, agent.cumulative_exposure - recovery_rate * 10)

            else:
                agent.time_away_from_faulty = 0

    def _calculate_metrics(self, arena_size: float, density: float) -> SimulationMetrics:
        """
        Compute final simulation metrics.

        Parameters
        ----------
        arena_size : float
            Arena side length.
        density : float
            Final swarm density.

        Returns
        -------
        SimulationMetrics
            Comprehensive performance metrics.
        """
        healthy_agents = [a for a in self.agents if not a.is_faulty]
        faulty_agents = [a for a in self.agents if a.is_faulty]
        active_agents = [a for a in self.agents if not a.is_quarantined]

        if self.config.num_agents > 0:
            total_task_rate = sum(agent.task_rate for agent in self.agents if not agent.is_quarantined)
            task_completion = total_task_rate / self.config.num_agents
        else:
            task_completion = 0.0

        if healthy_agents:
            healthy_task_sum = sum(a.task_rate for a in healthy_agents if not a.is_quarantined)
            healthy_active_count = sum(1 for a in healthy_agents if not a.is_quarantined)
            avg_healthy_performance = healthy_task_sum / healthy_active_count if healthy_active_count > 0 else 0.0
        else:
            avg_healthy_performance = 0.0

        mission_success = self._evaluate_mission_success(healthy_agents, density, self.config.fault_rate)

        severely_impaired = sum(1 for a in healthy_agents if a.task_rate < 0.8)

        quarantined_count = sum(1 for a in self.agents if a.is_quarantined)

        if faulty_agents:
            quarantined_faulty = sum(1 for a in faulty_agents if a.is_quarantined)
            quarantine_efficiency = quarantined_faulty / len(faulty_agents)
        else:
            quarantine_efficiency = 0.0

        cascade_prevention = 0.0

        total_energy = sum(a.energy_consumed for a in self.agents)

        if total_energy > 0:
            total_work = sum(a.task_rate for a in active_agents)
            work_per_energy = total_work / total_energy
        else:
            work_per_energy = 0.0

        total_coverage = sum(a.current_coverage_area for a in active_agents)
        coverage_efficiency = total_coverage / len(active_agents) if active_agents else 0.0

        if healthy_agents:
            high_performing = sum(1 for a in healthy_agents if a.task_rate > 0.9)
            healthy_preservation_rate = high_performing / len(healthy_agents)
        else:
            healthy_preservation_rate = 0.0

        if active_agents:
            active_task_sum = sum(a.task_rate for a in active_agents)
            active_agent_efficiency = active_task_sum / len(active_agents)
        else:
            active_agent_efficiency = 0.0

        return SimulationMetrics(
            num_agents=self.config.num_agents,
            fault_rate=self.config.fault_rate,
            arena_size=arena_size,
            density=density,
            scaling_mode=self.config.scaling_mode.value,
            task_completion=task_completion,
            avg_healthy_performance=avg_healthy_performance,
            mission_success=mission_success,
            severely_impaired=severely_impaired,
            num_healthy=len(healthy_agents),
            num_faulty=len(faulty_agents),
            quarantined=quarantined_count,
            quarantine_efficiency=quarantine_efficiency,
            cascade_prevention=cascade_prevention,
            avg_response_time=self._calculate_response_time(),
            work_per_energy=work_per_energy,
            coverage_efficiency=coverage_efficiency,
            healthy_preservation_rate=healthy_preservation_rate,
            active_agent_efficiency=active_agent_efficiency,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level,
            quarantine_enabled=self.config.enable_quarantine
        )

    def _evaluate_mission_success(
            self,
            healthy_agents: List[BaseAgent],
            density: float,
            fault_rate: float
    ) -> bool:
        """
        Evaluate mission success based on healthy agent performance.

        Parameters
        ----------
        healthy_agents : list of BaseAgent
            Non-faulty agents.
        density : float
            Swarm density.
        fault_rate : float
            Fault proportion.

        Returns
        -------
        bool
            Whether mission success criteria are met.
        """
        if not healthy_agents:
            return False

        if density > 0.01:
            threshold = 0.6
            required_fraction = 0.7
        elif fault_rate <= 0.10:
            threshold = 0.75
            required_fraction = 0.8
        else:
            threshold = 0.65
            required_fraction = 0.75

        above_threshold = sum(
            1 for a in healthy_agents
            if a.task_rate > threshold and not a.is_quarantined
        )

        success_fraction = above_threshold / len(healthy_agents)
        return success_fraction >= required_fraction

    def _log_quarantine_event(self, agent, t_now):
        """
        Log quarantine event with temporal and spatial context.

        Parameters
        ----------
        agent : BaseAgent
            Agent being quarantined.
        t_now : float
            Quarantine time.
        """
        label_at_t = bool(agent.is_faulty)
        onset_self = getattr(agent, 'degradation_onset_time', None)

        neighbor_rates = []
        neighbor_onsets = []

        if hasattr(agent, 'k_neighbors') and agent.k_neighbors:
            for dist, nb in agent.k_neighbors:
                neighbor_rates.append(nb.task_rate)
                if getattr(nb, 'degradation_onset_time', None) is not None:
                    neighbor_onsets.append(nb.degradation_onset_time)
        else:
            for other in self.agents:
                if other.agent_id != agent.agent_id and not other.is_quarantined:
                    dist = agent.distance_to(other)
                    if dist <= 25.0:
                        neighbor_rates.append(other.task_rate)
                        if getattr(other, 'degradation_onset_time', None) is not None:
                            neighbor_onsets.append(other.degradation_onset_time)

        neighbor_avg = float(np.mean(neighbor_rates)) if neighbor_rates else 1.0
        relative_gap = max(0.0, neighbor_avg - agent.task_rate)

        min_nb_onset = min(neighbor_onsets) if neighbor_onsets else float("inf")
        precedence_flag = int(
            (agent.degradation_onset_time or float("inf")) < (min_nb_onset - 0.5)
        )

        delta_rel = 0.10
        agent_class = "source-like" if (precedence_flag == 1 and relative_gap > delta_rel) else "victim-like"
        baseline_proxy = self._rate_ewma.get(agent.agent_id, 1.0) if hasattr(self, '_rate_ewma') else 1.0

        event = {
            "agent_id": agent.agent_id,
            "t_quarantine": t_now,
            "label_at_t": label_at_t,
            "degradation_onset_time": onset_self,
            "neighbor_min_onset_time": min_nb_onset,
            "personal_drop": max(0.0, baseline_proxy - agent.task_rate),
            "neighbor_avg": neighbor_avg,
            "relative_gap": relative_gap,
            "precedence_flag": precedence_flag,
            "class": agent_class,
        }

        if not hasattr(self, 'quarantine_event_log'):
            self.quarantine_event_log = []
        self.quarantine_event_log.append(event)

    def get_quarantine_event_metrics(self):
        """
        Calculate precision and recall from quarantine events.

        Returns
        -------
        dict
            Quarantine decision metrics.
        """
        if not hasattr(self, 'quarantine_event_log'):
            return {'tp': 0, 'fp': 0, 'precision': 0.0, 'recall': 0.0, 'total_quarantined': 0}

        events = self.quarantine_event_log

        tp_ids = {e["agent_id"] for e in events if e.get("label_at_t", False)}
        fp_ids = {e["agent_id"] for e in events if not e.get("label_at_t", False)}
        faulty_ids = {a.agent_id for a in self.agents if a.is_faulty}

        tp = len(tp_ids)
        fp = len(fp_ids)
        total_quarantined = len({e["agent_id"] for e in events})

        precision = tp / total_quarantined if total_quarantined > 0 else 0.0
        recall = len(tp_ids & faulty_ids) / len(faulty_ids) if len(faulty_ids) > 0 else 0.0

        return {
            'tp': tp,
            'fp': fp,
            'precision': precision,
            'recall': recall,
            'total_quarantined': total_quarantined,
            'total_faulty': len(faulty_ids)
        }

    def verify_confusion_matrix(self):
        """
        Verify confusion matrix integrity.

        Returns
        -------
        bool
            True if matrix is valid.
        """
        if not hasattr(self, 'detection_stats'):
            return True

        ds = self.detection_stats
        tp = ds.get('tp', 0)
        fp = ds.get('fp', 0)
        fn = ds.get('fn', 0)
        tn = ds.get('tn', 0)

        n_faulty = sum(1 for a in self.agents if a.is_faulty)
        n_healthy = len(self.agents) - n_faulty

        try:
            assert tp + fn == n_faulty
            assert fp + tn == n_healthy
            assert tp + fp + fn + tn == len(self.agents)
            assert tp >= 0 and fp >= 0 and fn >= 0 and tn >= 0
            assert tp <= n_faulty
            assert fp <= n_healthy
            return True
        except AssertionError:
            return False

    def set_detection_params(self, params: Dict[str, Any]) -> None:
        """
        Override detection parameters for calibration.

        Parameters
        ----------
        params : dict
            Parameter overrides.
        """
        if hasattr(self, 'h_params'):
            self.h_params.update(params)
        elif hasattr(self, 't_params'):
            self.t_params.update(params)
        elif hasattr(self, 'v_params'):
            self.v_params.update(params)

        self.override_params = params

    @abstractmethod
    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """
        Get density-adjusted parameters.

        Parameters
        ----------
        density : float
            Current swarm density.

        Returns
        -------
        dict
            Adapted parameters.
        """
        pass

    @abstractmethod
    def initialize_fault_detection(self, density: float) -> None:
        """
        Initialize fault detection strategy.

        Parameters
        ----------
        density : float
            Initial swarm density.
        """
        pass

    @abstractmethod
    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """
        Update fault detection and quarantine decisions.

        Parameters
        ----------
        elapsed : float
            Elapsed simulation time.
        density : float
            Current swarm density.
        """
        pass

    @abstractmethod
    def _calculate_response_time(self) -> float:
        """
        Calculate average detection response time.

        Returns
        -------
        float
            Mean response time.
        """
        pass
