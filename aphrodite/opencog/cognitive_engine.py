"""
Cognitive Engine for OpenCog integration with Aphrodite.

Provides the main cognitive processing engine that orchestrates large-scale 
inference through cognitive architecture patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)

from .atomspace import AtomSpaceManager, Atom, TruthValue
from .reasoning import ProbabilisticReasoner, LogicEngine
from .orchestrator import InferenceOrchestrator, AttentionManager
from .memory import CognitiveMemory
from .accelerator import CognitiveAccelerator


@dataclass
class CognitiveConfig:
    """Configuration for the cognitive engine."""
    
    # AtomSpace configuration
    atomspace_max_size: int = 1000000
    attention_allocation_rate: float = 0.1
    
    # Reasoning configuration
    max_inference_steps: int = 1000
    reasoning_threads: int = 4
    probabilistic_threshold: float = 0.7
    
    # Memory configuration
    memory_capacity: int = 100000
    forgetting_rate: float = 0.01
    consolidation_threshold: float = 0.8
    
    # Orchestration configuration
    batch_optimization: bool = True
    parallel_inference: bool = True
    adaptive_scheduling: bool = True
    
    # Performance configuration
    cognitive_cycles_per_second: int = 100
    max_concurrent_inferences: int = 16
    cache_size: int = 10000  # LRU cache size for accelerator
    
    # Learning configuration
    learning_rate: float = 0.1
    pattern_recognition_threshold: float = 0.6
    
    # Integration settings
    enable_attention_mechanism: bool = True
    enable_memory_consolidation: bool = True
    enable_cognitive_acceleration: bool = True
    
    @classmethod
    def create_performance_optimized(cls) -> 'CognitiveConfig':
        """
        Create a performance-optimized configuration preset.
        
        Optimizations:
        - Larger caches for better hit rates
        - Reduced consolidation frequency
        - Increased parallelism
        - Faster convergence thresholds
        """
        return cls(
            atomspace_max_size=2000000,  # 2x capacity
            cache_size=50000,  # 5x cache size
            reasoning_threads=8,  # More parallelism
            max_concurrent_inferences=32,  # More concurrent work
            cognitive_cycles_per_second=50,  # Less frequent cycles
            enable_memory_consolidation=True,  # Keep enabled but adaptive
            forgetting_rate=0.02,  # Faster forgetting to free memory
            max_inference_steps=500,  # Faster convergence
            batch_optimization=True,
            parallel_inference=True,
            adaptive_scheduling=True,
        )
    
    @classmethod
    def create_memory_optimized(cls) -> 'CognitiveConfig':
        """
        Create a memory-optimized configuration preset.
        
        Optimizations:
        - Smaller atomspace and caches
        - More aggressive forgetting
        - Reduced parallelism
        """
        return cls(
            atomspace_max_size=100000,  # 10x smaller
            cache_size=5000,  # Smaller cache
            memory_capacity=10000,
            reasoning_threads=2,
            max_concurrent_inferences=4,
            cognitive_cycles_per_second=20,
            enable_memory_consolidation=False,  # Disable to save memory
            forgetting_rate=0.05,  # Aggressive forgetting
        )
    
    @classmethod
    def create_balanced(cls) -> 'CognitiveConfig':
        """
        Create a balanced configuration (default).
        """
        return cls()  # Use default values


class CognitiveEngine:
    """
    Main cognitive engine that integrates OpenCog capabilities with Aphrodite.
    
    Provides large-scale inference orchestration through cognitive architecture
    patterns including attention allocation, memory management, and probabilistic
    reasoning.
    """
    
    def __init__(self, config: CognitiveConfig):
        self.config = config
        self._running = False
        self._lock = threading.RLock()
        
        # Initialize core components
        self.atomspace = AtomSpaceManager(max_size=config.atomspace_max_size)
        self.reasoner = ProbabilisticReasoner(
            atomspace=self.atomspace,
            threshold=config.probabilistic_threshold,
            max_steps=config.max_inference_steps
        )
        self.logic_engine = LogicEngine(self.atomspace)
        
        # Initialize orchestration components
        self.orchestrator = InferenceOrchestrator(
            config=config,
            atomspace=self.atomspace
        )
        self.attention_manager = AttentionManager(
            atomspace=self.atomspace,
            allocation_rate=config.attention_allocation_rate
        )
        
        # Initialize memory and acceleration
        self.memory = CognitiveMemory(
            atomspace=self.atomspace,
            capacity=config.memory_capacity,
            forgetting_rate=config.forgetting_rate
        )
        self.accelerator = CognitiveAccelerator(
            config=config,
            atomspace=self.atomspace
        )
        
        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=config.reasoning_threads)
        self._cognitive_cycle_task = None
        
        logger.info(f"Cognitive engine initialized with config: {config}")
    
    async def start(self):
        """Start the cognitive engine."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            logger.info("Starting cognitive engine...")
            
            # Start cognitive cycle
            self._cognitive_cycle_task = asyncio.create_task(
                self._cognitive_cycle_loop()
            )
            
            # Initialize components
            await self.orchestrator.start()
            await self.attention_manager.start()
            
            if self.config.enable_memory_consolidation:
                await self.memory.start_consolidation()
            
            logger.info("Cognitive engine started successfully")
    
    async def stop(self):
        """Stop the cognitive engine."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            logger.info("Stopping cognitive engine...")
            
            # Stop cognitive cycle
            if self._cognitive_cycle_task:
                self._cognitive_cycle_task.cancel()
                try:
                    await self._cognitive_cycle_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components
            await self.orchestrator.stop()
            await self.attention_manager.stop()
            await self.memory.stop_consolidation()
            
            # Shutdown resources
            self.executor.shutdown(wait=True)
            self.atomspace.shutdown()
            
            logger.info("Cognitive engine stopped")
    
    async def process_inference_request(self, 
                                      prompt: str, 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an inference request through the cognitive architecture.
        
        Args:
            prompt: The input prompt for inference
            context: Optional context information
            
        Returns:
            Processed inference results with cognitive annotations
        """
        if not self._running:
            raise RuntimeError("Cognitive engine not running")
        
        # Create cognitive representation of the request
        request_atom = await self._create_request_atom(prompt, context)
        
        # Apply attention mechanism
        if self.config.enable_attention_mechanism:
            await self.attention_manager.allocate_attention(request_atom)
        
        # Perform cognitive reasoning
        reasoning_results = await self._perform_cognitive_reasoning(request_atom)
        
        # Apply memory consolidation
        if self.config.enable_memory_consolidation:
            await self.memory.consolidate_experience(request_atom, reasoning_results)
        
        # Apply cognitive acceleration
        if self.config.enable_cognitive_acceleration:
            accelerated_results = await self.accelerator.accelerate_inference(
                reasoning_results
            )
        else:
            accelerated_results = reasoning_results
        
        # Orchestrate final response
        final_response = await self.orchestrator.orchestrate_response(
            accelerated_results, context
        )
        
        return final_response
    
    async def _create_request_atom(self, prompt: str, 
                                  context: Optional[Dict[str, Any]]) -> Atom:
        """Create an atom representation of the inference request."""
        # Create concept node for the prompt
        prompt_node = self.atomspace.create_node(
            f"Prompt: {prompt[:100]}...",  # Truncate for readability
            truth_value=TruthValue(0.9, 0.9)
        )
        
        # Add context information if provided
        if context:
            for key, value in context.items():
                context_node = self.atomspace.create_node(f"Context: {key}={value}")
                # Link context to prompt
                self.atomspace.create_link(
                    f"HasContext({prompt_node.name}, {context_node.name})",
                    [prompt_node, context_node]
                )
        
        return prompt_node
    
    async def _perform_cognitive_reasoning(self, request_atom: Atom) -> Dict[str, Any]:
        """Perform cognitive reasoning on the request."""
        reasoning_tasks = []
        
        # Probabilistic reasoning
        reasoning_tasks.append(
            self.reasoner.infer_async(request_atom)
        )
        
        # Logic-based reasoning
        reasoning_tasks.append(
            self.logic_engine.reason_async(request_atom)
        )
        
        # Pattern matching
        reasoning_tasks.append(
            self.memory.find_similar_patterns_async(request_atom)
        )
        
        # Execute all reasoning tasks concurrently
        results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {
            'probabilistic': results[0] if not isinstance(results[0], Exception) else None,
            'logical': results[1] if not isinstance(results[1], Exception) else None,
            'patterns': results[2] if not isinstance(results[2], Exception) else None,
            'confidence': self._calculate_combined_confidence(results),
            'request_atom': request_atom
        }
        
        return combined_results
    
    def _calculate_combined_confidence(self, results: List[Any]) -> float:
        """Calculate combined confidence from multiple reasoning results."""
        valid_results = [r for r in results if not isinstance(r, Exception)]
        if not valid_results:
            return 0.0
        
        # Simple average for now - could use more sophisticated combination
        confidences = []
        for result in valid_results:
            if hasattr(result, 'confidence'):
                confidences.append(result.confidence)
            elif isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    async def _cognitive_cycle_loop(self):
        """Main cognitive cycle loop for continuous processing."""
        cycle_interval = 1.0 / self.config.cognitive_cycles_per_second
        
        while self._running:
            try:
                await self._execute_cognitive_cycle()
                await asyncio.sleep(cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cognitive cycle: {e}")
                await asyncio.sleep(cycle_interval)
    
    async def _execute_cognitive_cycle(self):
        """Execute one cognitive cycle."""
        # Update attention values
        if self.config.enable_attention_mechanism:
            await self.attention_manager.update_attention_values()
        
        # Perform memory maintenance
        if self.config.enable_memory_consolidation:
            await self.memory.perform_maintenance()
        
        # Update cognitive patterns
        await self.accelerator.update_optimization_patterns()
        
        # Orchestrate any pending inferences
        await self.orchestrator.process_pending_inferences()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cognitive engine statistics."""
        return {
            'atomspace_size': self.atomspace.size(),
            'running': self._running,
            'memory_patterns': len(self.memory.patterns) if hasattr(self.memory, 'patterns') else 0,
            'attention_allocated_atoms': len([a for a in self.atomspace._atoms.values() 
                                            if a.attention_value > 0]),
            'reasoning_cache_size': len(getattr(self.reasoner, '_cache', {})),
            'orchestrator_queue_size': getattr(self.orchestrator, '_queue_size', 0),
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()