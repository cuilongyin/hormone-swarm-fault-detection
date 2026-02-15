"""Self-diagnosis threshold detector for fault detection.

This module provides the simplest fault detection approach: each agent
independently monitors its own performance and self-quarantines when
performance falls below a threshold for consecutive timesteps.

This serves as a baseline for comparison with more sophisticated distributed
detection methods.
"""

from typing import Dict, Any
from collections import defaultdict

from src.core.simulator import BaseSimulator, SimulationConfig
from src.core.agent import BaseAgent


class ThresholdDetector(BaseSimulator):
    """Simple threshold-based self-diagnosis fault detector.

    Each agent monitors its own task completion rate independently.
    Quarantine occurs when performance remains below threshold for
    consecutive timesteps, preventing spurious detections from
    temporary performance fluctuations.

    No inter-agent communication or voting is used - this represents
    the minimal viable fault detection strategy.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration

    Attributes
    ----------
    threshold : float
        Task rate below which agent appears faulty (default: 0.8)
    consecutive_count : int
        Number of consecutive low readings required (default: 2)
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # Detection parameters
        self.threshold = 0.8
        self.consecutive_count = 2

        # Per-agent state tracking
        self._low_performance_counters = defaultdict(int)

        # Detection statistics
        self._detection_stats = self._init_stats()
        self._counted_agents = set()

        self._warmup_time = 1.0

    def _init_stats(self) -> Dict[str, int]:
        """Initialize detection statistics."""
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Return fixed parameters.

        Threshold detection doesn't adapt to density changes.

        Parameters
        ----------
        density : float
            Agent density (unused)

        Returns
        -------
        dict
            Simulation parameters
        """
        return {
            'damage_rate': 0.008,
            'recovery_rate': 0.008,
            'communication_range': self.COMMUNICATION_RANGE,
            'impedance_range': self.IMPEDANCE_RANGE
        }

    def set_detection_params(self, params: Dict[str, Any]) -> None:
        """Update detection parameters.

        Parameters
        ----------
        params : dict
            Parameter updates. Valid keys:
            - 'threshold': performance threshold
            - 'consecutive_count': required consecutive low readings
        """
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'consecutive_count' in params:
            self.consecutive_count = params['consecutive_count']

    def initialize_fault_detection(self, density: float) -> None:
        """Initialize detection state for new simulation run."""
        self._low_performance_counters.clear()
        self._detection_stats = self._init_stats()
        self._counted_agents.clear()

    def _check_agent_performance(self, agent: BaseAgent, elapsed: float) -> None:
        """Monitor individual agent's performance and update quarantine state.

        Parameters
        ----------
        agent : BaseAgent
            Agent to monitor
        elapsed : float
            Current simulation time
        """
        agent_id = agent.agent_id

        # Check performance against threshold
        if agent.task_rate < self.threshold:
            # Increment consecutive low performance counter
            self._low_performance_counters[agent_id] += 1

            # Quarantine if threshold met
            if self._low_performance_counters[agent_id] >= self.consecutive_count:
                agent.is_quarantined = True
                agent.quarantine_time = elapsed

                # Update statistics (count each agent only once)
                if agent_id not in self._counted_agents:
                    self._counted_agents.add(agent_id)

                    if agent.is_faulty:
                        self._detection_stats['tp'] += 1
                    else:
                        self._detection_stats['fp'] += 1
        else:
            # Reset counter when performance recovers
            self._low_performance_counters[agent_id] = 0

    def _finalize_statistics(self) -> None:
        """Count undetected agents at simulation end."""
        for agent in self.agents:
            if agent.agent_id in self._counted_agents:
                continue

            if agent.is_faulty and not agent.is_quarantined:
                self._detection_stats['fn'] += 1
            elif not agent.is_faulty and not agent.is_quarantined:
                self._detection_stats['tn'] += 1

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

        # Check each agent independently
        for agent in self.agents:
            if not agent.is_quarantined:
                self._check_agent_performance(agent, elapsed)

        # Finalize statistics at simulation end
        if elapsed >= self.config.run_time - 0.5:
            self._finalize_statistics()

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
            if agent.is_faulty and agent.is_quarantined
            and hasattr(agent, 'quarantine_time')
        ]

        return sum(response_times) / len(response_times) if response_times else float('inf')

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Compute detection performance metrics.

        Returns
        -------
        dict
            Detection metrics including:
            - true_positives, false_positives, true_negatives, false_negatives
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - avg_response_time: mean time to quarantine faulty agents
        """
        tp = self._detection_stats['tp']
        fp = self._detection_stats['fp']
        fn = self._detection_stats['fn']
        tn = self._detection_stats['tn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'avg_response_time': self._calculate_response_time()
        }