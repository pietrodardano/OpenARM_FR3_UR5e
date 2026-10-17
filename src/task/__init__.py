"""Task module: the weigh-and-sort demo state machine and its component contracts."""

from .contact import ContactConfig, ExternalContactDetector, wait_for_contact
from .interfaces import (
    ComparisonResult,
    ManipulationSkills,
    MassEstimate,
    ObjectDetection,
    ObjectDetector,
    PayloadEstimator,
    TcpTarget,
)
from .prompt import DemoMode, TaskIntent, parse_prompt
from .weigh_and_sort import DemoConfig, Phase, TaskReport, WeighAndSortTask
from .weighing import combine_estimates, compare_estimates

__all__ = [
    "ComparisonResult",
    "ContactConfig",
    "DemoConfig",
    "DemoMode",
    "ExternalContactDetector",
    "ManipulationSkills",
    "MassEstimate",
    "ObjectDetection",
    "ObjectDetector",
    "PayloadEstimator",
    "Phase",
    "TaskIntent",
    "TaskReport",
    "TcpTarget",
    "WeighAndSortTask",
    "combine_estimates",
    "compare_estimates",
    "parse_prompt",
    "wait_for_contact",
]
