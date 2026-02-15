"""
Evaluation metrics for swarm fault tolerance experiments.

This module provides metrics to assess fault detection strategies
in multi-agent swarms, with emphasis on counterfactual cascade
analysis and energy efficiency.
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass, field

from src.core.agent import BaseAgent, FaultType


@dataclass
class MetricHistory:
    """Time-series storage for performance metrics."""
    timestamps: List[float] = field(default_factory=list)
    tcr_values: List[float] = field(default_factory=list)
    qe_values: List[float] = field(default_factory=list)
    wpe_values: List[float] = field(default_factory=list)
    cpr_values: List[float] = field(default_factory=list)
    impairment_tracking: Dict[int, List[float]] = field(default_factory=dict)
    exposure_tracking: Dict[int, List[float]] = field(default_factory=dict)

    def add_sample(self, timestamp: float, tcr: float, qe: float, wpe: float, cpr: float) -> None:
        """Record metric sample at specified timestamp."""
        self.timestamps.append(timestamp)
        self.tcr_values.append(tcr)
        self.qe_values.append(qe)
        self.wpe_values.append(wpe)
        self.cpr_values.append(cpr)


@dataclass
class CascadeMetrics:
    """Cascade prevention metrics using counterfactual analysis."""
    counterfactual_cascade_prevention: float
    integrated_hazard_reduction: float
    mean_time_to_impairment: float
    agents_saved: int
    baseline_impaired_ids: Set[int]
    method_impaired_ids: Set[int]


class SwarmMetrics:
    """
    Performance metrics calculator for swarm experiments.

    Computes standard metrics (TCR, QE, WPE) and counterfactual
    cascade prevention metrics by comparing against baseline runs.

    Parameters
    ----------
    None

    Attributes
    ----------
    history : MetricHistory
        Time-series of metric values.
    baseline_impairment_data : dict or None
        Baseline impairment data for counterfactual analysis.
    """

    ENERGY_IDLE = 0.1
    ENERGY_ACTIVE = 1.2
    IMPAIRMENT_THRESHOLD = 0.8

    def __init__(self):
        self.history = MetricHistory()
        self.baseline_impairment_data = None
        self.baseline_exposure_data = None
        self.is_baseline_run = False

    def calculate_task_completion_rate(self, agents: List[BaseAgent]) -> float:
        """
        Compute average task completion rate across swarm.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.

        Returns
        -------
        float
            Mean task rate (0-1).
        """
        if not agents:
            return 0.0
        return sum(agent.task_rate for agent in agents) / len(agents)

    def calculate_quarantine_efficiency(self, agents: List[BaseAgent]) -> float:
        """
        Compute fraction of faulty agents successfully quarantined.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.

        Returns
        -------
        float
            Quarantine efficiency (0-1).
        """
        faulty_agents = [a for a in agents if a.is_faulty]
        if not faulty_agents:
            return 1.0

        quarantined_faulty = sum(1 for a in faulty_agents if a.is_quarantined)
        return quarantined_faulty / len(faulty_agents)

    def calculate_work_per_energy(self, agents: List[BaseAgent], dt: float = 0.1) -> float:
        """
        Compute instantaneous work per unit energy.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.
        dt : float, default=0.1
            Time step size.

        Returns
        -------
        float
            Work output per Joule.
        """
        if not agents:
            return 0.0

        total_work = sum(a.task_rate * dt for a in agents if not a.is_quarantined)
        total_energy = sum(
            (self.ENERGY_IDLE if a.is_quarantined else self.ENERGY_ACTIVE) * dt
            for a in agents
        )

        return total_work / total_energy if total_energy > 0 else 0.0

    def calculate_cumulative_work_per_energy(self, agents: List[BaseAgent]) -> float:
        """
        Compute cumulative work per energy from stored values.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.

        Returns
        -------
        float
            Cumulative work per Joule.
        """
        if not agents:
            return 0.0

        total_energy = sum(agent.energy_consumed for agent in agents)
        total_work = sum(agent.cumulative_work for agent in agents)

        return total_work / total_energy if total_energy > 0 else 0.0

    def set_as_baseline(self, is_baseline: bool = True):
        """
        Mark this instance as baseline for counterfactual analysis.

        Parameters
        ----------
        is_baseline : bool, default=True
            Whether this is a baseline run.
        """
        self.is_baseline_run = is_baseline
        if is_baseline:
            self.baseline_impairment_data = {}
            self.baseline_exposure_data = {}

    def track_agent_states(self, agents: List[BaseAgent], timestamp: float):
        """
        Track agent states for counterfactual analysis.

        Records task rates, exposure levels, and first impairment times
        for each agent to enable comparison across conditions.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.
        timestamp : float
            Current simulation time.
        """
        if not hasattr(self.history, 'impairment_tracking'):
            self.history.impairment_tracking = {}
        if not hasattr(self.history, 'exposure_tracking'):
            self.history.exposure_tracking = {}
        if not hasattr(self.history, 'first_impairment_time'):
            self.history.first_impairment_time = {}
        if not hasattr(self.history, 'ever_impaired'):
            self.history.ever_impaired = set()

        for agent in agents:
            if agent.agent_id not in self.history.impairment_tracking:
                self.history.impairment_tracking[agent.agent_id] = []
            self.history.impairment_tracking[agent.agent_id].append(agent.task_rate)

            if agent.agent_id not in self.history.exposure_tracking:
                self.history.exposure_tracking[agent.agent_id] = []
            exposure = getattr(agent, 'cumulative_exposure', 0.0)
            self.history.exposure_tracking[agent.agent_id].append(exposure)

            if not agent.is_faulty and agent.task_rate < self.IMPAIRMENT_THRESHOLD:
                if agent.agent_id not in self.history.first_impairment_time:
                    self.history.first_impairment_time[agent.agent_id] = timestamp
                    self.history.ever_impaired.add(agent.agent_id)

        if self.is_baseline_run:
            self.baseline_impairment_data = dict(self.history.impairment_tracking)
            self.baseline_exposure_data = dict(self.history.exposure_tracking)
            self.baseline_first_impairment = dict(self.history.first_impairment_time)
            self.baseline_ever_impaired = set(self.history.ever_impaired)

    def calculate_counterfactual_cascade_metrics(
            self,
            agents: List[BaseAgent],
            baseline_metrics: Optional['SwarmMetrics'] = None,
            run_time: float = 15.0
    ) -> CascadeMetrics:
        """
        Compute cascade prevention using counterfactual analysis.

        Compares cascade outcomes against baseline to measure actual
        prevention rather than just current state. Includes penalties
        for excessive false positives.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.
        baseline_metrics : SwarmMetrics or None
            Baseline metrics instance for comparison.
        run_time : float, default=15.0
            Total simulation time.

        Returns
        -------
        CascadeMetrics
            Comprehensive cascade prevention metrics.
        """
        if baseline_metrics:
            baseline_ever_impaired = getattr(baseline_metrics, 'baseline_ever_impaired', set())
            baseline_first_impairment = getattr(baseline_metrics, 'baseline_first_impairment', {})
        else:
            return self._calculate_fallback_cascade_metrics(agents)

        method_ever_impaired = getattr(self.history, 'ever_impaired', set())

        if not baseline_ever_impaired:
            raw_ccp = 0.0
            agents_protected = set()
        else:
            agents_protected = baseline_ever_impaired - method_ever_impaired
            raw_ccp = len(agents_protected) / len(baseline_ever_impaired)

        healthy_quarantined = sum(1 for a in agents if not a.is_faulty and a.is_quarantined)
        total_healthy = sum(1 for a in agents if not a.is_faulty)
        quarantine_penalty = healthy_quarantined / total_healthy if total_healthy > 0 else 0
        adjusted_ccp = raw_ccp * (1 - quarantine_penalty)

        baseline_total_exposure = 0.0
        method_total_exposure = 0.0

        for agent in agents:
            if agent.is_faulty:
                continue

            if baseline_metrics and agent.agent_id in baseline_metrics.baseline_exposure_data:
                baseline_agent_exposure = baseline_metrics.baseline_exposure_data[agent.agent_id]
                baseline_total_exposure += max(baseline_agent_exposure) if baseline_agent_exposure else 0

            if agent.agent_id in self.history.exposure_tracking:
                method_agent_exposure = self.history.exposure_tracking[agent.agent_id]
                method_total_exposure += max(method_agent_exposure) if method_agent_exposure else 0

        eps = 1e-6

        if baseline_total_exposure <= eps or not baseline_ever_impaired:
            ihr = 0.0
        else:
            ihr = max(0.0, 1.0 - (method_total_exposure / baseline_total_exposure))

        tti_gains = []

        for agent_id in baseline_first_impairment:
            baseline_time = baseline_first_impairment[agent_id]

            if agent_id in self.history.first_impairment_time:
                method_time = self.history.first_impairment_time[agent_id]
                gain = method_time - baseline_time
            else:
                gain = run_time - baseline_time

            tti_gains.append(gain)

        mean_tti_gain = np.mean(tti_gains) if tti_gains else 0.0
        agents_saved = len(agents_protected)

        return CascadeMetrics(
            counterfactual_cascade_prevention=adjusted_ccp,
            integrated_hazard_reduction=ihr,
            mean_time_to_impairment=mean_tti_gain,
            agents_saved=agents_saved,
            baseline_impaired_ids=baseline_ever_impaired,
            method_impaired_ids=method_ever_impaired
        )

    def _calculate_fallback_cascade_metrics(self, agents: List[BaseAgent]) -> CascadeMetrics:
        """Fallback cascade metrics when baseline unavailable."""
        healthy_agents = [a for a in agents if not a.is_faulty]
        impaired_ids = set(a.agent_id for a in healthy_agents if a.task_rate < self.IMPAIRMENT_THRESHOLD)

        if len(healthy_agents) > 0:
            ccp = 1.0 - (len(impaired_ids) / len(healthy_agents))
        else:
            ccp = 0.0

        return CascadeMetrics(
            counterfactual_cascade_prevention=ccp,
            integrated_hazard_reduction=0.0,
            mean_time_to_impairment=0.0,
            agents_saved=0,
            baseline_impaired_ids=impaired_ids,
            method_impaired_ids=impaired_ids
        )

    def calculate_all_metrics(
            self,
            agents: List[BaseAgent],
            timestamp: Optional[float] = None,
            store_history: bool = True,
            baseline_metrics: Optional['SwarmMetrics'] = None
    ) -> Dict[str, float]:
        """
        Compute all primary metrics.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.
        timestamp : float or None
            Current simulation time.
        store_history : bool, default=True
            Whether to store values in history.
        baseline_metrics : SwarmMetrics or None
            Baseline for counterfactual analysis.

        Returns
        -------
        dict
            Dictionary of metric names to values.
        """
        if timestamp is not None:
            self.track_agent_states(agents, timestamp)

        tcr = self.calculate_task_completion_rate(agents)
        qe = self.calculate_quarantine_efficiency(agents)
        wpe = self.calculate_work_per_energy(agents)

        if self.is_baseline_run:
            cascade_metrics = CascadeMetrics(
                counterfactual_cascade_prevention=0.0,
                integrated_hazard_reduction=0.0,
                mean_time_to_impairment=0.0,
                agents_saved=0,
                baseline_impaired_ids=set(),
                method_impaired_ids=set()
            )
            cpr = 0.0
        else:
            cascade_metrics = self.calculate_counterfactual_cascade_metrics(
                agents,
                baseline_metrics,
                run_time=timestamp if timestamp else 15.0
            )
            cpr = cascade_metrics.counterfactual_cascade_prevention

        if store_history and timestamp is not None:
            self.history.add_sample(timestamp, tcr, qe, wpe, cpr)

        return {
            'task_completion_rate': tcr,
            'quarantine_efficiency': qe,
            'work_per_energy': wpe,
            'cascade_prevention_rate': cpr,
            'counterfactual_cascade_prevention': cascade_metrics.counterfactual_cascade_prevention,
            'integrated_hazard_reduction': cascade_metrics.integrated_hazard_reduction,
            'time_to_impairment_gain': cascade_metrics.mean_time_to_impairment,
            'agents_saved': cascade_metrics.agents_saved
        }

    def get_additional_metrics(self, agents: List[BaseAgent]) -> Dict[str, Any]:
        """
        Compute supplementary analysis metrics.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.

        Returns
        -------
        dict
            Additional performance and diagnostic metrics.
        """
        healthy_agents = [a for a in agents if not a.is_faulty]
        faulty_agents = [a for a in agents if a.is_faulty]
        active_agents = [a for a in agents if not a.is_quarantined]

        task_rates = [a.task_rate for a in agents]
        stress_levels = [a.stress_level for a in agents]

        total_coverage = sum(a.current_coverage_area for a in active_agents)
        num_quarantined = sum(1 for a in agents if a.is_quarantined)

        false_positives = sum(1 for a in healthy_agents if a.is_quarantined)
        false_positive_rate = false_positives / len(healthy_agents) if healthy_agents else 0.0

        false_negatives = sum(1 for a in faulty_agents if not a.is_quarantined)
        false_negative_rate = false_negatives / len(faulty_agents) if faulty_agents else 0.0

        highly_stressed = sum(1 for s in stress_levels if s > 50)

        avg_task_rate = np.mean(task_rates) if task_rates else 0.0
        task_rate_std = np.std(task_rates) if task_rates else 0.0

        healthy_task_rates = [a.task_rate for a in healthy_agents]
        avg_healthy_task_rate = np.mean(healthy_task_rates) if healthy_task_rates else 0.0

        avg_stress_level = np.mean(stress_levels) if stress_levels else 0.0
        coverage_per_agent = total_coverage / len(active_agents) if active_agents else 0.0

        return {
            'num_healthy': len(healthy_agents),
            'num_faulty': len(faulty_agents),
            'num_quarantined': num_quarantined,
            'num_active': len(active_agents),
            'avg_task_rate': avg_task_rate,
            'avg_healthy_task_rate': avg_healthy_task_rate,
            'task_rate_std': task_rate_std,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'avg_stress_level': avg_stress_level,
            'highly_stressed_count': highly_stressed,
            'total_coverage': total_coverage,
            'coverage_per_agent': coverage_per_agent
        }

    def get_fault_type_metrics(self, agents: List[BaseAgent]) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics segmented by fault severity.

        Parameters
        ----------
        agents : list of BaseAgent
            All agents in the swarm.

        Returns
        -------
        dict
            Metrics for each fault type present.
        """
        fault_type_metrics = {}

        for fault_type in FaultType:
            type_agents = [a for a in agents if a.fault_type == fault_type]

            if type_agents:
                quarantined_count = sum(1 for a in type_agents if a.is_quarantined)
                avg_stress = sum(a.stress_level for a in type_agents) / len(type_agents)
                avg_task_rate = sum(a.task_rate for a in type_agents) / len(type_agents)

                fault_type_metrics[fault_type.value] = {
                    'count': len(type_agents),
                    'quarantined': quarantined_count,
                    'quarantine_rate': quarantined_count / len(type_agents),
                    'avg_stress': avg_stress,
                    'avg_task_rate': avg_task_rate
                }

        return fault_type_metrics

    def calculate_response_metrics(
            self,
            detection_times: Dict[int, float],
            quarantine_times: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Compute detection and quarantine response time statistics.

        Parameters
        ----------
        detection_times : dict
            Mapping of agent IDs to detection times.
        quarantine_times : dict
            Mapping of agent IDs to quarantine times.

        Returns
        -------
        dict
            Response time statistics.
        """
        if not detection_times:
            return {
                'avg_detection_time': 0.0,
                'avg_quarantine_time': 0.0,
                'avg_response_time': 0.0,
                'min_response_time': 0.0,
                'max_response_time': 0.0
            }

        response_times = [
            quarantine_times[agent_id] - detection_time
            for agent_id, detection_time in detection_times.items()
            if agent_id in quarantine_times
        ]

        avg_detection = np.mean(list(detection_times.values()))
        avg_quarantine = np.mean(list(quarantine_times.values())) if quarantine_times else 0.0

        if response_times:
            avg_response = np.mean(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
        else:
            avg_response = 0.0
            min_response = 0.0
            max_response = 0.0

        return {
            'avg_detection_time': avg_detection,
            'avg_quarantine_time': avg_quarantine,
            'avg_response_time': avg_response,
            'min_response_time': min_response,
            'max_response_time': max_response
        }

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics from metric history.

        Returns
        -------
        dict
            Summary statistics (mean, std, min, max, final) for each metric.
        """
        if not self.history.timestamps:
            return {}

        metrics = {
            'task_completion_rate': self.history.tcr_values,
            'quarantine_efficiency': self.history.qe_values,
            'work_per_energy': self.history.wpe_values,
            'cascade_prevention_rate': self.history.cpr_values
        }

        summary = {}
        for name, values in metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'final': values[-1]
                }

        if hasattr(self.history, 'impairment_tracking') and self.history.impairment_tracking:
            summary['cascade_analysis'] = {
                'agents_tracked': len(self.history.impairment_tracking),
                'has_baseline': self.baseline_impairment_data is not None
            }

        return summary
