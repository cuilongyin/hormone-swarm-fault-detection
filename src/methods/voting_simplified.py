"""Observation-based voting detector for multi-agent fault detection.

This module implements a distributed fault detection mechanism where agents
observe their neighbors' performance and vote to quarantine underperforming
agents. The approach balances sensitivity with false positive reduction through
adaptive voting thresholds.

The detector uses three key parameters:
    - observation_threshold: Performance level below which agents appear faulty
    - voting_threshold: Fraction of neighbors that must vote for quarantine
    - min_neighbors: Minimum observers required for reliable decisions
"""

from typing import Dict, Any, Tuple
import numpy as np

from src.core.simulator import BaseSimulator, SimulationConfig
from src.core.agent import BaseAgent, FaultType


class VotingAgent(BaseAgent):
    """Lightweight agent for observation-based voting.

    This agent doesn't maintain complex state - voting decisions are made
    by the simulator based on cross-agent observations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ObservationVotingDetector(BaseSimulator):
    """Distributed fault detector using neighbor observation and voting.

    Each agent observes its neighbors' task completion rates and votes to
    quarantine agents performing below threshold. Quarantine occurs when
    vote ratio exceeds the voting threshold, with adaptive adjustments based
    on the voter's own relative performance.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration including arena size, agent count, and
        communication parameters.

    Attributes
    ----------
    voting_threshold : float
        Base fraction of votes needed for quarantine (default: 0.5)
    observation_threshold : float
        Task rate below which an agent appears faulty (default: 0.65)
    min_neighbors : int
        Minimum observers required for reliable voting (default: 5)
    communication_range : float
        Maximum distance for agent communication (default: 25.0)
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # Detection parameters - tuned for balance between precision and recall
        self.voting_threshold = 0.5
        self.observation_threshold = 0.65
        self.communication_range = 25.0
        self.min_neighbors = 5

        # Adaptive threshold adjustments based on voter performance
        self._threshold_adjustments = {
            'severe_deficit': (-0.15, 0.65),  # (gap, effective_threshold)
            'moderate_deficit': (-0.05, 0.55),
            'normal': (0.0, 0.5)
        }

        self._warmup_time = 2.0
        self._detection_stats = self._init_stats()
        self._counted_agents = set()

    def _init_stats(self) -> Dict[str, int]:
        """Initialize detection statistics."""
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    def _create_agent(self, agent_id: int, position: Tuple[float, float],
                      fault_type: FaultType, arena_size: float) -> BaseAgent:
        """Create a voting-capable agent."""
        return VotingAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )

    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Return fixed parameters for this detector.

        Parameters
        ----------
        density : float
            Agent density (unused - included for interface compatibility)

        Returns
        -------
        dict
            Simulation parameters
        """
        return {
            'damage_rate': 0.008,
            'recovery_rate': 0.008,
            'communication_range': self.communication_range,
            'impedance_range': self.IMPEDANCE_RANGE
        }

    def set_detection_params(self, params: Dict[str, Any]) -> None:
        """Update detection parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameter updates. Valid keys:
            - 'threshold': voting_threshold
            - 'observation_threshold': observation_threshold
            - 'communication_range': communication_range
        """
        if 'threshold' in params:
            self.voting_threshold = params['threshold']
        if 'observation_threshold' in params:
            self.observation_threshold = params['observation_threshold']
        if 'communication_range' in params:
            self.communication_range = params['communication_range']

    def initialize_fault_detection(self, density: float) -> None:
        """Initialize detection state for new simulation run."""
        self._detection_stats = self._init_stats()
        self._counted_agents.clear()

    def _get_adaptive_threshold(self, performance_gap: float) -> float:
        """Compute adaptive voting threshold based on voter performance.

        Agents performing worse than their neighbors require stronger consensus
        before quarantining others, reducing false positives from struggling
        agents misidentifying healthy neighbors.

        Parameters
        ----------
        performance_gap : float
            Average difference between voter's performance and neighbors'
            (positive means voter is better)

        Returns
        -------
        float
            Effective voting threshold
        """
        for gap, threshold in self._threshold_adjustments.values():
            if performance_gap < gap:
                return threshold
        return self.voting_threshold

    def _collect_votes(self, agent: BaseAgent) -> Tuple[int, int, np.ndarray]:
        """Collect votes from neighbors about agent's performance.

        Parameters
        ----------
        agent : BaseAgent
            Agent being evaluated

        Returns
        -------
        votes_quarantine : int
            Number of votes for quarantine
        total_votes : int
            Total number of votes cast
        performance_gaps : ndarray
            Difference between agent's performance and each neighbor's
        """
        votes_quarantine = 0
        total_votes = 0
        performance_gaps = []

        for other in self.agents:
            if other.agent_id == agent.agent_id:
                continue

            dist = agent.distance_to(other)
            if dist > self.communication_range:
                continue

            # Simulate packet loss
            if np.random.random() < self.config.packet_loss_rate:
                continue

            total_votes += 1

            # Vote based on observation
            if other.task_rate < self.observation_threshold:
                votes_quarantine += 1

            # Track relative performance for adaptive thresholding
            performance_gaps.append(agent.task_rate - other.task_rate)

        return votes_quarantine, total_votes, np.array(performance_gaps)

    def _update_statistics(self, agent: BaseAgent) -> None:
        """Update detection statistics when agent is quarantined.

        Parameters
        ----------
        agent : BaseAgent
            Newly quarantined agent
        """
        if agent.agent_id in self._counted_agents:
            return

        self._counted_agents.add(agent.agent_id)

        if agent.is_faulty:
            self._detection_stats['tp'] += 1
        else:
            self._detection_stats['fp'] += 1

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
        """Execute one fault detection cycle.

        Parameters
        ----------
        elapsed : float
            Simulation time elapsed
        density : float
            Current agent density (unused)
        """
        if not self.config.enable_quarantine or elapsed < self._warmup_time:
            return

        # Voting phase
        for agent in self.agents:
            if agent.is_quarantined:
                continue

            votes_quarantine, total_votes, performance_gaps = \
                self._collect_votes(agent)

            # Require minimum neighbors for reliable decision
            if total_votes < self.min_neighbors:
                continue

            # Adaptive threshold based on relative performance
            avg_gap = np.mean(performance_gaps)
            effective_threshold = self._get_adaptive_threshold(avg_gap)

            # Quarantine decision
            vote_ratio = votes_quarantine / total_votes
            if vote_ratio > effective_threshold:
                agent.is_quarantined = True
                agent.quarantine_time = elapsed
                self._update_statistics(agent)

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

        return np.mean(response_times) if response_times else float('inf')

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


# Backward compatibility aliases
SimpleVotingAgent = VotingAgent
SimpleVotingSimulator = ObservationVotingDetector