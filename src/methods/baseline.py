"""Baseline simulator with no fault detection.

This module provides a baseline simulation with no fault detection mechanism,
serving as a control for measuring the impact of detection strategies on
system performance.
"""

from typing import Dict, Any

from src.core.simulator import BaseSimulator, SimulationConfig


class BaselineSimulator(BaseSimulator):
    """No-operation baseline simulator without fault detection.

    This simulator runs the multi-agent system without any fault detection
    or quarantine mechanisms, providing baseline performance metrics for
    comparison with detection strategies.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Return fixed simulation parameters.

        Parameters
        ----------
        density : float
            Agent density (unused)

        Returns
        -------
        dict
            Fixed simulation parameters
        """
        return {
            'damage_rate': 0.05,
            'recovery_rate': 0.008,
            'communication_range': self.COMMUNICATION_RANGE,
            'impedance_range': self.IMPEDANCE_RANGE
        }

    def initialize_fault_detection(self, density: float) -> None:
        """No-op: baseline has no detection system."""
        pass

    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """No-op: baseline has no detection updates."""
        pass

    def _calculate_response_time(self) -> float:
        """Return infinite response time for baseline.

        Returns
        -------
        float
            Infinity (no detection occurs)
        """
        return float('inf')