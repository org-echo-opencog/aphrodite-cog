"""
Inference orchestrator and attention management for OpenCog integration.

Provides intelligent orchestration of large-scale inference operations and
attention allocation mechanisms for cognitive resource management.
"""

import asyncio
import heapq
import time
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
logger = logging.getLogger(__name__)

from .atomspace import AtomSpaceManager, Atom, AtomType, TruthValue
from .cognitive_engine import CognitiveConfig


@dataclass
class InferenceRequest:
    """Represents an inference request in the orchestration queue."""
    request_id: str
    atom: Atom
    priority: float
    timestamp: float
    context: Optional[Dict[str, Any]] = None
    dependencies: Set[str] = field(default_factory=set)
    
    def __lt__(self, other):
        """Priority queue comparison."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class AttentionAllocation:
    """Attention allocation for an atom."""
    atom: Atom
    attention_value: float
    importance: float
    urgency: float
    last_accessed: float


class InferenceOrchestrator:
    """
    Orchestrates large-scale inference operations with intelligent scheduling
    and resource management.
    """
    
    def __init__(self, config: CognitiveConfig, atomspace: AtomSpaceManager):
        self.config = config
        self.atomspace = atomspace
        self._running = False
        
        # Request management
        self._request_queue: List[InferenceRequest] = []
        self._active_requests: Dict[str, InferenceRequest] = {}
        self._completed_requests: Dict[str, Any] = {}
        self._failed_requests: Dict[str, Exception] = {}
        
        # Scheduling and optimization
        self._batch_requests: List[InferenceRequest] = []
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_inferences)
        self._lock = threading.RLock()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._performance_metrics: Dict[str, Any] = {
            'requests_processed': 0,
            'average_processing_time': 0.0,
            'batch_efficiency': 0.0,
            'resource_utilization': 0.0
        }
        
        logger.info("Inference orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processing_task = asyncio.create_task(self._processing_loop())
            logger.info("Inference orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            # Wait for active requests to complete
            if self._active_requests:
                logger.info(f"Waiting for {len(self._active_requests)} active requests to complete")
                await self._wait_for_completion()
            
            self.executor.shutdown(wait=True)
            logger.info("Inference orchestrator stopped")
    
    async def submit_inference_request(self, atom: Atom, priority: float = 0.5,
                                     context: Optional[Dict[str, Any]] = None,
                                     dependencies: Optional[Set[str]] = None) -> str:
        """
        Submit an inference request for orchestration.
        
        Args:
            atom: The atom to perform inference on
            priority: Priority level (0.0 to 1.0, higher is more important)
            context: Optional context information
            dependencies: Set of request IDs this request depends on
            
        Returns:
            Request ID for tracking
        """
        request_id = f"req_{int(time.time() * 1000000)}_{atom.uuid[:8]}"
        
        request = InferenceRequest(
            request_id=request_id,
            atom=atom,
            priority=priority,
            timestamp=time.time(),
            context=context or {},
            dependencies=dependencies or set()
        )
        
        with self._lock:
            heapq.heappush(self._request_queue, request)
            
            # Update dependency graph
            for dep_id in request.dependencies:
                self._dependency_graph[dep_id].add(request_id)
        
        logger.debug(f"Submitted inference request {request_id} with priority {priority}")
        return request_id
    
    async def get_request_result(self, request_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of an inference request.
        
        Args:
            request_id: The request ID to get results for
            timeout: Optional timeout in seconds
            
        Returns:
            The inference result
            
        Raises:
            TimeoutError: If timeout is reached
            Exception: If the request failed
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check if completed
                if request_id in self._completed_requests:
                    return self._completed_requests[request_id]
                
                # Check if failed
                if request_id in self._failed_requests:
                    raise self._failed_requests[request_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(0.1)
    
    async def orchestrate_response(self, reasoning_results: Dict[str, Any],
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate the final response from reasoning results.
        
        Args:
            reasoning_results: Results from cognitive reasoning
            context: Optional context information
            
        Returns:
            Orchestrated response
        """
        # Combine different reasoning modalities
        combined_confidence = reasoning_results.get('confidence', 0.5)
        
        # Apply context-aware adjustments
        if context:
            combined_confidence = self._apply_context_adjustments(
                combined_confidence, context
            )
        
        # Generate orchestrated response
        response = {
            'reasoning_results': reasoning_results,
            'final_confidence': combined_confidence,
            'orchestration_metadata': {
                'timestamp': time.time(),
                'orchestrator_version': '1.0',
                'context_applied': context is not None
            }
        }
        
        # Add performance metrics
        response['performance'] = self._get_current_performance_metrics()
        
        return response
    
    async def process_pending_inferences(self):
        """Process any pending inference requests."""
        if not self._running:
            return
        
        with self._lock:
            if not self._request_queue:
                return
            
            # Get ready requests (no unresolved dependencies)
            ready_requests = self._get_ready_requests()
            
            if not ready_requests:
                return
        
        # Process batch if batch optimization is enabled
        if self.config.batch_optimization and len(ready_requests) > 1:
            await self._process_batch(ready_requests[:self.config.max_concurrent_inferences])
        else:
            # Process individual requests
            for request in ready_requests[:self.config.max_concurrent_inferences]:
                asyncio.create_task(self._process_individual_request(request))
    
    def _get_ready_requests(self) -> List[InferenceRequest]:
        """Get requests that are ready to process (no pending dependencies)."""
        ready_requests = []
        
        while self._request_queue:
            request = heapq.heappop(self._request_queue)
            
            # Check if dependencies are satisfied
            if self._are_dependencies_satisfied(request):
                ready_requests.append(request)
                self._active_requests[request.request_id] = request
            else:
                # Put back in queue
                heapq.heappush(self._request_queue, request)
                break
        
        return ready_requests
    
    def _are_dependencies_satisfied(self, request: InferenceRequest) -> bool:
        """Check if all dependencies for a request are satisfied."""
        for dep_id in request.dependencies:
            if (dep_id not in self._completed_requests and 
                dep_id not in self._failed_requests):
                return False
        return True
    
    async def _processing_loop(self):
        """Main processing loop for the orchestrator."""
        while self._running:
            try:
                await self.process_pending_inferences()
                await self._cleanup_completed_requests()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestrator processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of inference requests together."""
        logger.debug(f"Processing batch of {len(requests)} requests")
        
        batch_start_time = time.time()
        
        # Group requests by similarity for optimization
        grouped_requests = self._group_similar_requests(requests)
        
        batch_tasks = []
        for group in grouped_requests:
            batch_tasks.append(
                asyncio.create_task(self._process_request_group(group))
            )
        
        # Wait for all batch tasks to complete
        await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Update performance metrics
        batch_time = time.time() - batch_start_time
        self._update_batch_performance_metrics(len(requests), batch_time)
    
    def _group_similar_requests(self, requests: List[InferenceRequest]) -> List[List[InferenceRequest]]:
        """Group similar requests for batch processing optimization."""
        # Simple grouping by atom type for now
        groups = defaultdict(list)
        
        for request in requests:
            group_key = request.atom.atom_type
            groups[group_key].append(request)
        
        return list(groups.values())
    
    async def _process_request_group(self, requests: List[InferenceRequest]):
        """Process a group of similar requests."""
        for request in requests:
            asyncio.create_task(self._process_individual_request(request))
    
    async def _process_individual_request(self, request: InferenceRequest):
        """Process an individual inference request."""
        try:
            logger.debug(f"Processing request {request.request_id}")
            start_time = time.time()
            
            # Simulate cognitive processing (in real implementation, 
            # this would call the actual reasoning engines)
            result = await self._simulate_cognitive_processing(request)
            
            processing_time = time.time() - start_time
            
            with self._lock:
                self._completed_requests[request.request_id] = {
                    'result': result,
                    'processing_time': processing_time,
                    'timestamp': time.time()
                }
                
                if request.request_id in self._active_requests:
                    del self._active_requests[request.request_id]
                
                # Trigger dependent requests
                self._trigger_dependent_requests(request.request_id)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            logger.debug(f"Completed request {request.request_id} in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to process request {request.request_id}: {e}")
            
            with self._lock:
                self._failed_requests[request.request_id] = e
                
                if request.request_id in self._active_requests:
                    del self._active_requests[request.request_id]
                
                # Mark dependent requests as failed too
                self._mark_dependent_requests_failed(request.request_id, e)
    
    async def _simulate_cognitive_processing(self, request: InferenceRequest) -> Dict[str, Any]:
        """Simulate cognitive processing for a request."""
        # In real implementation, this would call reasoning engines
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'atom': request.atom,
            'inferred_truth_value': TruthValue(0.8, 0.7),
            'reasoning_confidence': 0.75,
            'cognitive_patterns_applied': ['pattern_matching', 'probabilistic_inference'],
            'context_influence': request.context
        }
    
    def _trigger_dependent_requests(self, completed_request_id: str):
        """Trigger processing of requests that depended on the completed request."""
        # This is automatically handled by the main processing loop
        # which checks dependencies before processing
        pass
    
    def _mark_dependent_requests_failed(self, failed_request_id: str, error: Exception):
        """Mark dependent requests as failed when a dependency fails."""
        for dependent_id in self._dependency_graph.get(failed_request_id, set()):
            self._failed_requests[dependent_id] = error
            
            # Recursively mark dependents of dependents
            self._mark_dependent_requests_failed(dependent_id, error)
    
    async def _cleanup_completed_requests(self):
        """Clean up old completed and failed requests to save memory."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        with self._lock:
            # Clean completed requests
            to_remove = []
            for req_id, result in self._completed_requests.items():
                if current_time - result.get('timestamp', 0) > cleanup_age:
                    to_remove.append(req_id)
            
            for req_id in to_remove:
                del self._completed_requests[req_id]
                self._dependency_graph.pop(req_id, None)
            
            # Clean failed requests
            to_remove = []
            for req_id in self._failed_requests:
                # Assuming failed requests have a timestamp (would need to track this)
                to_remove.append(req_id)  # Simplified cleanup
            
            # Keep only recent failures for debugging
            if len(to_remove) > 1000:
                for req_id in to_remove[:-100]:  # Keep last 100 failures
                    del self._failed_requests[req_id]
                    self._dependency_graph.pop(req_id, None)
    
    async def _wait_for_completion(self):
        """Wait for all active requests to complete."""
        while self._active_requests:
            await asyncio.sleep(0.1)
    
    def _apply_context_adjustments(self, confidence: float, 
                                 context: Dict[str, Any]) -> float:
        """Apply context-based adjustments to confidence."""
        # Simple context adjustments
        adjustments = 0.0
        
        if context.get('high_importance'):
            adjustments += 0.1
        
        if context.get('time_critical'):
            adjustments += 0.05
        
        if context.get('uncertain_input'):
            adjustments -= 0.1
        
        return max(0.0, min(1.0, confidence + adjustments))
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics."""
        with self._lock:
            self._performance_metrics['requests_processed'] += 1
            
            # Update running average of processing time
            current_avg = self._performance_metrics['average_processing_time']
            count = self._performance_metrics['requests_processed']
            
            self._performance_metrics['average_processing_time'] = (
                (current_avg * (count - 1) + processing_time) / count
            )
    
    def _update_batch_performance_metrics(self, batch_size: int, batch_time: float):
        """Update batch processing performance metrics."""
        expected_individual_time = (
            self._performance_metrics['average_processing_time'] * batch_size
        )
        
        if expected_individual_time > 0:
            efficiency = expected_individual_time / batch_time
            self._performance_metrics['batch_efficiency'] = efficiency
    
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            return self._performance_metrics.copy()
    
    @property
    def _queue_size(self) -> int:
        """Get current queue size."""
        return len(self._request_queue) + len(self._active_requests)


class AttentionManager:
    """
    Manages attention allocation across atoms in the atomspace for
    efficient cognitive resource utilization.
    """
    
    def __init__(self, atomspace: AtomSpaceManager, allocation_rate: float = 0.1):
        self.atomspace = atomspace
        self.allocation_rate = allocation_rate
        self._running = False
        
        # Attention management
        self._attention_allocations: Dict[str, AttentionAllocation] = {}
        self._attention_focus_queue: deque = deque(maxlen=1000)
        
        # Threading
        self._update_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        logger.info("Attention manager initialized")
    
    async def start(self):
        """Start the attention manager."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._update_task = asyncio.create_task(self._attention_update_loop())
            logger.info("Attention manager started")
    
    async def stop(self):
        """Stop the attention manager."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Attention manager stopped")
    
    async def allocate_attention(self, atom: Atom, importance: float = 0.5,
                                urgency: float = 0.5):
        """
        Allocate attention to a specific atom.
        
        Args:
            atom: The atom to allocate attention to
            importance: How important this atom is (0.0 to 1.0)
            urgency: How urgent attention allocation is (0.0 to 1.0)
        """
        allocation = AttentionAllocation(
            atom=atom,
            attention_value=importance * urgency,
            importance=importance,
            urgency=urgency,
            last_accessed=time.time()
        )
        
        with self._lock:
            self._attention_allocations[atom.uuid] = allocation
            self._attention_focus_queue.append(atom.uuid)
        
        # Update atom's attention value
        atom.attention_value = allocation.attention_value
        
        logger.debug(f"Allocated attention {allocation.attention_value:.3f} to atom {atom.uuid}")
    
    async def update_attention_values(self):
        """Update attention values for all atoms with batch processing."""
        current_time = time.time()
        
        # Collect updates without holding lock for entire operation
        updates_to_apply = []
        atoms_to_remove = []
        
        with self._lock:
            # Quick snapshot of allocations
            allocations_snapshot = list(self._attention_allocations.items())
        
        # Process updates outside the lock
        for atom_uuid, allocation in allocations_snapshot:
            # Decay attention over time
            time_decay = max(0.0, 1.0 - (current_time - allocation.last_accessed) * 0.001)
            new_attention = allocation.attention_value * time_decay
            
            if new_attention < 0.01:
                atoms_to_remove.append(atom_uuid)
            else:
                updates_to_apply.append((atom_uuid, new_attention, allocation))
        
        # Apply all updates in one lock acquisition
        with self._lock:
            for atom_uuid, new_attention, allocation in updates_to_apply:
                allocation.attention_value = new_attention
                allocation.atom.attention_value = new_attention
            
            # Batch remove low-attention atoms
            for atom_uuid in atoms_to_remove:
                self._attention_allocations.pop(atom_uuid, None)
    
    async def _attention_update_loop(self):
        """Main attention update loop."""
        while self._running:
            try:
                await self.update_attention_values()
                await self._redistribute_attention()
                await asyncio.sleep(1.0 / 10.0)  # 10 Hz update rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in attention update loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _redistribute_attention(self):
        """Redistribute attention based on global constraints with optimized calculation."""
        # Quick check without lock if possible
        if not self._attention_allocations:
            return
        
        total_attention_budget = len(self.atomspace._atoms) * self.allocation_rate
        
        # Calculate current total and prepare scale factor outside main lock
        with self._lock:
            if not self._attention_allocations:
                return
            
            # Fast sum of attention values
            current_total = sum(a.attention_value for a in self._attention_allocations.values())
        
        # Only redistribute if over budget (most common case is under budget)
        if current_total <= total_attention_budget:
            return
        
        # Calculate scale factor
        scale_factor = total_attention_budget / current_total
        
        # Batch apply scaling
        with self._lock:
            for allocation in self._attention_allocations.values():
                allocation.attention_value *= scale_factor
                allocation.atom.attention_value = allocation.attention_value