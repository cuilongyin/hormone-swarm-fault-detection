"""Bio-inspired hormone signaling for distributed fault detection.

This module implements a fault detection mechanism inspired by biological
stress hormone signaling. Agents produce internal "stress hormones" based
on performance degradation, which diffuse through the network to neighbors.
High total hormone levels (internal + external) trigger quarantine.

The key innovation is emergent source-victim discrimination: agents adapt
their emission behavior based on feedback from neighbors, naturally developing
different signaling patterns depending on whether they're fault sources or
victims of cascading effects.
"""

from typing import Dict, Any, Tuple, List
from collections import defaultdict
import numpy as np

from src.core.simulator import BaseSimulator, SimulationConfig
from src.core.agent import BaseAgent, FaultType


class HormoneAgent(BaseAgent):
    """Agent with adaptive bio-inspired hormone signaling.

    Attributes
    ----------
    internal_hormone : float
        Self-generated stress signal based on performance
    external_hormone : float
        Accumulated stress signals from neighbors
    emission_gain : float
        Adaptive multiplier for hormone emission (enables source-victim discrimination)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hormone system state
        self.internal_hormone = 0.0
        self.external_hormone = 0.0
        self.hormone_emission = 0.0
        self._prev_external_hormone = 0.0

        # Network state
        self.k_neighbors: List[Tuple[float, 'HormoneAgent']] = []

        # History tracking for decisions
        self._total_hormone_history = []
        self._neighbor_stress_history = []

        # Emergent source-victim discrimination via adaptive emission
        self.emission_gain = 1.0
        self._gain_alpha = 0.05  # Learning rate
        self._min_gain = 0.1     # Sources converge here
        self._max_gain = 2.5     # Victims can reach here
        self._feedback_window = 10

        # Quarantine tracking
        self.quarantine_reason = None

    def update_internal_hormone(self, dt: float = 0.1) -> None:
        """Update internal stress hormone based on performance degradation.

        All agents produce hormone proportional to their stress level,
        with nonlinear scaling to amplify severe degradation.

        Parameters
        ----------
        dt : float
            Time step for integration
        """
        # Compute stress from task performance
        stress = max(0, 1.0 - self.task_rate)
        baseline = 0.02

        # Nonlinear production function
        if stress < 0.05:
            production = stress * 10.0
        elif stress < 0.3:
            production = stress ** 1.5 * 7.0
        else:
            production = stress ** 2 * 15.0

        target = baseline + production

        # Smooth integration
        alpha = dt / (0.5 + dt)
        self.internal_hormone = (1 - alpha) * self.internal_hormone + alpha * target

    def update_emission_feedback(self) -> None:
        """Adapt emission gain based on neighbor stress dynamics.

        This creates emergent source-victim discrimination:
        - If neighbors improve → emission is helping → increase gain (victim)
        - If neighbors degrade → emission isn't helping → decrease gain (source)

        This feedback loop naturally segregates agents without hard thresholds.
        """
        if not self.k_neighbors:
            return

        # Measure neighbor stress changes
        neighbor_deltas = [
            neighbor.external_hormone - neighbor._prev_external_hormone
            for _, neighbor in self.k_neighbors
            if hasattr(neighbor, '_prev_external_hormone')
        ]

        if not neighbor_deltas:
            return

        # Update history
        avg_delta = np.mean(neighbor_deltas)
        self._neighbor_stress_history.append(avg_delta)

        if len(self._neighbor_stress_history) > self._feedback_window:
            self._neighbor_stress_history.pop(0)

        # Need minimum samples for reliable feedback
        if len(self._neighbor_stress_history) < 3:
            return

        # Compute recent trend
        recent_trend = np.mean(self._neighbor_stress_history[-5:]) \
                       if len(self._neighbor_stress_history) >= 5 else avg_delta

        # Adapt emission gain based on effectiveness
        if recent_trend < -0.01:  # Neighbors improving
            self.emission_gain *= (1.0 + self._gain_alpha)
        elif recent_trend > 0.01:  # Neighbors degrading
            self.emission_gain *= (1.0 - self._gain_alpha)

        # Clamp to valid range
        self.emission_gain = np.clip(self.emission_gain, self._min_gain, self._max_gain)

    def calculate_emission(self) -> float:
        """Compute hormone emission using adaptive gain.

        Returns
        -------
        float
            Hormone amount to emit this timestep
        """
        if self.is_quarantined:
            return 0.0

        # Primary emission: internal stress × adaptive gain
        primary = self.internal_hormone * 1.2 * self.emission_gain

        # Secondary emission: severe degradation signal
        secondary = 0.0
        if self.task_rate < 0.4:
            degradation = 1.0 - self.task_rate
            secondary = degradation * 0.3

        self.hormone_emission = primary + secondary
        return self.hormone_emission

    def update_external_hormone(self, incoming: float, dt: float = 0.1) -> None:
        """Update external hormone with absorption regulation.

        Parameters
        ----------
        incoming : float
            Hormone amount arriving from neighbors
        dt : float
            Time step
        """
        # Store for feedback calculation
        self._prev_external_hormone = self.external_hormone

        # Regulate absorption based on capacity
        max_capacity = 1.0
        current_total = self.internal_hormone + self.external_hormone
        remaining_capacity = max(0, max_capacity - current_total)

        # Absorption proportional to remaining capacity
        absorption_rate = remaining_capacity / max_capacity if max_capacity > 0 else 0.0
        self.external_hormone += incoming * absorption_rate

        # Adaptive decay
        if self.external_hormone > 0.5:
            decay_rate = 0.25
        elif self.external_hormone > 0.3:
            decay_rate = 0.15
        else:
            decay_rate = 0.08

        self.external_hormone *= (1.0 - decay_rate)

        # Enforce capacity constraint
        new_total = self.internal_hormone + self.external_hormone
        if new_total > max_capacity:
            self.external_hormone = max(0.0, max_capacity - self.internal_hormone)

    def get_hormone_levels(self) -> Tuple[float, float]:
        """Get current and recent average hormone levels.

        Returns
        -------
        current : float
            Current total hormone level
        recent_avg : float
            Average over recent history
        """
        # Enforce capacity constraint
        max_capacity = 1.0
        total = self.internal_hormone + self.external_hormone

        if total > max_capacity:
            self.external_hormone = max(0.0, max_capacity - self.internal_hormone)
            total = max_capacity

        # Update history
        self._total_hormone_history.append(total)
        if len(self._total_hormone_history) > 10:
            self._total_hormone_history.pop(0)

        # Compute recent average
        recent_avg = np.mean(self._total_hormone_history[-5:]) \
                     if len(self._total_hormone_history) >= 5 else total

        return total, recent_avg


class HormoneDetector(BaseSimulator):
    """Bio-inspired fault detector using hormone signaling.

    Implements distributed fault detection through diffusive hormone
    signaling between k-nearest neighbors. Features emergent source-victim
    discrimination through adaptive emission feedback.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration

    Attributes
    ----------
    k_neighbors : int
        Number of nearest neighbors for communication
    max_hop_distance : float
        Maximum distance for neighbor connections
    """

    K_NEIGHBORS = 6
    MAX_HOP_DISTANCE = 30.0

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # Detection thresholds with hysteresis
        self._thresholds = {
            'activation': 0.35,
            'sustained_count': 1,
            'recovery': 0.25,
            'recovery_count': 3,
        }

        # Adaptive threshold feature
        self._use_ratio_conditioning = getattr(
            config, 'use_ratio_conditioned_threshold', True
        )

        # State tracking
        self._network_graph = {}
        self._consecutive_high = defaultdict(int)
        self._consecutive_low = defaultdict(int)
        self._detection_stats = self._init_stats()
        self._counted_agents = set()

        self._warmup_time = 1.0

    def _init_stats(self) -> Dict[str, int]:
        """Initialize detection statistics."""
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    def _create_agent(self, agent_id: int, position: Tuple[float, float],
                      fault_type: FaultType, arena_size: float) -> BaseAgent:
        """Create hormone-capable agent."""
        return HormoneAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )

    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Return adaptive parameters.

        Parameters
        ----------
        density : float
            Current agent density

        Returns
        -------
        dict
            Simulation parameters
        """
        return {
            'damage_rate': 0.008,
            'recovery_rate': 0.008,
            'k_neighbors': self.K_NEIGHBORS,
        }

    def set_detection_params(self, params: Dict[str, Any]) -> None:
        """Update detection parameters.

        Parameters
        ----------
        params : dict
            Parameter updates for threshold dictionary
        """
        self._thresholds.update(params)

    def initialize_fault_detection(self, density: float) -> None:
        """Initialize detection state for new simulation run."""
        self._counted_agents.clear()
        self._detection_stats = self._init_stats()
        self._consecutive_high.clear()
        self._consecutive_low.clear()

    def _build_network_graph(self) -> None:
        """Construct k-nearest neighbor communication graph."""
        self._network_graph.clear()

        for agent in self.agents:
            if not isinstance(agent, HormoneAgent) or agent.is_quarantined:
                continue

            # Find all neighbors within range
            neighbors = [
                (agent.distance_to(other), other)
                for other in self.agents
                if (isinstance(other, HormoneAgent) and
                    other.agent_id != agent.agent_id and
                    not other.is_quarantined and
                    agent.distance_to(other) <= self.MAX_HOP_DISTANCE)
            ]

            # Keep k nearest
            neighbors.sort(key=lambda x: x[0])
            agent.k_neighbors = neighbors[:self.K_NEIGHBORS]
            self._network_graph[agent.agent_id] = agent.k_neighbors

    def _propagate_hormones(self, dt: float = 0.1) -> None:
        """Execute one hormone diffusion cycle.

        Parameters
        ----------
        dt : float
            Time step for diffusion
        """
        # Update internal stress levels
        for agent in self.agents:
            if isinstance(agent, HormoneAgent):
                agent.update_internal_hormone(dt)

        # Update emission feedback (emergent discrimination mechanism)
        for agent in self.agents:
            if isinstance(agent, HormoneAgent) and not agent.is_quarantined:
                agent.update_emission_feedback()

        # Propagate emissions to neighbors
        for agent in self.agents:
            if not isinstance(agent, HormoneAgent) or agent.is_quarantined:
                continue

            emission = agent.calculate_emission()

            if emission <= 0:
                continue

            # Diffuse to k-nearest neighbors
            for dist, neighbor in agent.k_neighbors:
                if not (agent.can_transmit() and neighbor.can_receive()):
                    continue

                # Hormone diffusion: flux ∝ concentration gradient / distance
                diffusion_rate = 600.0
                spatial_decay = np.exp(-dist / 25.0)
                transfer_rate = diffusion_rate * spatial_decay / (dist + 1.0)

                propagated = emission * transfer_rate * dt
                neighbor.update_external_hormone(propagated * dt, dt)

    def _get_adaptive_threshold(self, agent: HormoneAgent) -> float:
        """Compute adaptive quarantine threshold based on stress source.

        Agents with high internal/total ratio are likely fault sources
        and require lower thresholds. Agents with low ratios are likely
        victims of cascading effects and need higher thresholds.

        Parameters
        ----------
        agent : HormoneAgent
            Agent to evaluate

        Returns
        -------
        float
            Effective quarantine threshold
        """
        if not self._use_ratio_conditioning:
            return self._thresholds['activation']

        # Compute source likelihood
        total = agent.internal_hormone + agent.external_hormone + 0.01
        internal_ratio = agent.internal_hormone / total

        # Source-adaptive thresholds
        if internal_ratio > 0.6:
            return 0.30  # Likely source: lower threshold
        elif internal_ratio > 0.4:
            return 0.35  # Mixed
        else:
            return 0.45  # Likely victim: higher threshold

    def _evaluate_quarantine(self, agent: HormoneAgent, elapsed: float) -> bool:
        """Determine if agent should be quarantined.

        Parameters
        ----------
        agent : HormoneAgent
            Agent to evaluate
        elapsed : float
            Current simulation time

        Returns
        -------
        bool
            Whether to quarantine agent
        """
        current_level, _ = agent.get_hormone_levels()
        effective_threshold = self._get_adaptive_threshold(agent)

        # High hormone: potential fault
        if current_level > effective_threshold:
            self._consecutive_high[agent.agent_id] += 1
            self._consecutive_low[agent.agent_id] = 0

            if self._consecutive_high[agent.agent_id] >= self._thresholds['sustained_count']:
                return True

        # Low hormone: potential recovery
        elif current_level < self._thresholds['recovery']:
            self._consecutive_low[agent.agent_id] += 1
            self._consecutive_high[agent.agent_id] = 0

        return False

    def _execute_quarantine(self, agent: HormoneAgent, elapsed: float) -> None:
        """Quarantine agent and update statistics.

        Parameters
        ----------
        agent : HormoneAgent
            Agent to quarantine
        elapsed : float
            Current simulation time
        """
        agent.is_quarantined = True
        agent.quarantine_time = elapsed
        agent.quarantine_reason = 'fault_detected'

        if agent.agent_id not in self._counted_agents:
            self._counted_agents.add(agent.agent_id)

            if agent.is_faulty:
                self._detection_stats['tp'] += 1
            else:
                self._detection_stats['fp'] += 1

    def _check_recovery(self, agent: HormoneAgent) -> None:
        """Check if quarantined agent can be released.

        Parameters
        ----------
        agent : HormoneAgent
            Quarantined agent to check
        """
        if not agent.is_quarantined:
            return

        # Performance-based recovery for collision hazards
        if agent.quarantine_reason == 'collision_hazard' and agent.task_rate > 0.7:
            agent.is_quarantined = False
            agent.quarantine_reason = None
            self._consecutive_high[agent.agent_id] = 0
            self._consecutive_low[agent.agent_id] = 0
            return

        # Hormone-based recovery
        if self._consecutive_low[agent.agent_id] >= self._thresholds['recovery_count']:
            agent.is_quarantined = False
            agent.quarantine_reason = None
            self._consecutive_high[agent.agent_id] = 0
            self._consecutive_low[agent.agent_id] = 0

    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """Execute one detection cycle.

        Parameters
        ----------
        elapsed : float
            Simulation time elapsed
        density : float
            Current agent density (unused)
        """
        if not self.config.enable_quarantine or elapsed < self._warmup_time:
            return

        # Build communication network
        self._build_network_graph()

        # Propagate hormone signals
        self._propagate_hormones()

        # Evaluate quarantine decisions
        for agent in self.agents:
            if not isinstance(agent, HormoneAgent):
                continue

            if not agent.is_quarantined:
                if self._evaluate_quarantine(agent, elapsed):
                    self._execute_quarantine(agent, elapsed)
            else:
                self._check_recovery(agent)

    def _calculate_response_time(self) -> float:
        """Compute average time to quarantine faulty agents.

        Returns
        -------
        float
            Mean response time, or infinity if no faulty agents quarantined
        """
        response_times = [
            agent.quarantine_time
            for agent in self.agents
            if (isinstance(agent, HormoneAgent) and
                agent.is_faulty and agent.is_quarantined and
                hasattr(agent, 'quarantine_time'))
        ]

        return np.mean(response_times) if response_times else float('inf')

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Compute detection performance metrics.

        Returns
        -------
        dict
            Detection metrics for evaluation
        """
        tp = self._detection_stats['tp']
        fp = self._detection_stats['fp']
        total_faulty = sum(1 for a in self.agents if a.is_faulty)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_faulty if total_faulty > 0 else 0.0

        return {
            'tp': tp,
            'fp': fp,
            'precision': precision,
            'recall': recall,
            'total_quarantined': tp + fp,
            'total_faulty': total_faulty,
            'avg_response_time': self._calculate_response_time()
        }

    # Alias for backward compatibility
    get_quarantine_event_metrics = get_detection_metrics

    def _log_quarantine_event(self, agent: BaseAgent, elapsed: float) -> None:
        """Log quarantine event for debugging (optional override)."""
        pass


# Backward compatibility aliases
HormoneAgent = HormoneAgent  # Already good name
LocalHormoneSimulator = HormoneDetector