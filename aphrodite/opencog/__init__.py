"""
OpenCog cognitive architecture integration for Aphrodite engine.

This module provides large-scale inference orchestration and cognitive 
acceleration through OpenCog's atomspace and reasoning frameworks.
"""

from .cognitive_engine import CognitiveEngine, CognitiveConfig
from .atomspace import AtomSpaceManager, Atom, Link, Node
from .reasoning import ProbabilisticReasoner, LogicEngine
from .orchestrator import InferenceOrchestrator, AttentionManager
from .memory import CognitiveMemory, MemoryPattern
from .accelerator import CognitiveAccelerator, OptimizationPattern

__all__ = [
    "CognitiveEngine",
    "CognitiveConfig", 
    "AtomSpaceManager",
    "Atom",
    "Link", 
    "Node",
    "ProbabilisticReasoner",
    "LogicEngine",
    "InferenceOrchestrator",
    "AttentionManager",
    "CognitiveMemory",
    "MemoryPattern",
    "CognitiveAccelerator",
    "OptimizationPattern",
]