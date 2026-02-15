"""
Cognitive memory management for OpenCog integration.

Provides episodic and semantic memory systems with pattern recognition
and consolidation capabilities for large-scale inference optimization.
"""

import asyncio
import time
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import logging
logger = logging.getLogger(__name__)

from .atomspace import AtomSpaceManager, Atom, AtomType, TruthValue, Node, Link


@dataclass
class MemoryPattern:
    """Represents a learned pattern in cognitive memory."""
    pattern_id: str
    atoms: List[Atom]
    frequency: int
    last_accessed: float
    strength: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def similarity_to(self, other: 'MemoryPattern') -> float:
        """Calculate similarity to another pattern."""
        if not self.atoms or not other.atoms:
            return 0.0
        
        # Simple Jaccard similarity for now
        set1 = {atom.uuid for atom in self.atoms}
        set2 = {atom.uuid for atom in other.atoms}
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


@dataclass
class EpisodicMemory:
    """Represents an episodic memory trace."""
    memory_id: str
    timestamp: float
    context_atoms: List[Atom]
    result_atoms: List[Atom]
    success_rate: float
    confidence: float
    access_count: int = 0
    
    def get_age(self) -> float:
        """Get age of memory in seconds."""
        return time.time() - self.timestamp


class CognitiveMemory:
    """
    Cognitive memory system providing episodic and semantic memory
    with pattern recognition and consolidation capabilities.
    """
    
    def __init__(self, atomspace: AtomSpaceManager, capacity: int = 100000,
                 forgetting_rate: float = 0.01):
        self.atomspace = atomspace
        self.capacity = capacity
        self.forgetting_rate = forgetting_rate
        
        # Memory stores
        self.patterns: Dict[str, MemoryPattern] = {}
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        
        # Indexing structures
        self._pattern_by_atom: Dict[str, Set[str]] = defaultdict(set)
        self._memory_by_context: Dict[str, Set[str]] = defaultdict(set)
        
        # Consolidation management
        self._consolidation_running = False
        self._consolidation_task: Optional[asyncio.Task] = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        
        # Performance tracking
        self._access_stats = {
            'pattern_hits': 0,
            'pattern_misses': 0,
            'consolidations_performed': 0,
            'memories_forgotten': 0
        }
        
        logger.info(f"Cognitive memory initialized with capacity {capacity}")
    
    async def store_experience(self, context_atoms: List[Atom], 
                              result_atoms: List[Atom],
                              success_rate: float = 1.0,
                              confidence: float = 0.8) -> str:
        """
        Store an episodic memory of an inference experience.
        
        Args:
            context_atoms: Atoms representing the context/input
            result_atoms: Atoms representing the results/output
            success_rate: How successful this inference was
            confidence: Confidence in storing this memory
            
        Returns:
            Memory ID
        """
        memory_id = f"episodic_{int(time.time() * 1000000)}"
        
        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=time.time(),
            context_atoms=context_atoms.copy(),
            result_atoms=result_atoms.copy(),
            success_rate=success_rate,
            confidence=confidence
        )
        
        with self._lock:
            # Check capacity and remove old memories if necessary
            if len(self.episodic_memories) >= self.capacity:
                await self._forget_old_memories()
            
            self.episodic_memories[memory_id] = memory
            
            # Update indexing
            for atom in context_atoms:
                context_key = f"{atom.atom_type.value}:{atom.name[:50]}"
                self._memory_by_context[context_key].add(memory_id)
        
        # Extract patterns from this experience
        await self._extract_patterns_from_experience(memory)
        
        logger.debug(f"Stored episodic memory {memory_id}")
        return memory_id
    
    async def find_similar_patterns_async(self, query_atom: Atom) -> Dict[str, Any]:
        """Find patterns similar to the query atom asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.find_similar_patterns, query_atom
        )
    
    def find_similar_patterns(self, query_atom: Atom, 
                            similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Find memory patterns similar to the given atom.
        
        Args:
            query_atom: The atom to find patterns for
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with similar patterns and confidence
        """
        similar_patterns = []
        
        with self._lock:
            # Check direct pattern matches first
            if query_atom.uuid in self._pattern_by_atom:
                for pattern_id in self._pattern_by_atom[query_atom.uuid]:
                    if pattern_id in self.patterns:
                        pattern = self.patterns[pattern_id]
                        similar_patterns.append((pattern, 1.0))  # Exact match
            
            # Find semantic similarities
            for pattern in self.patterns.values():
                similarity = self._calculate_pattern_similarity(query_atom, pattern)
                if similarity >= similarity_threshold:
                    similar_patterns.append((pattern, similarity))
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Update access statistics
        if similar_patterns:
            self._access_stats['pattern_hits'] += 1
        else:
            self._access_stats['pattern_misses'] += 1
        
        # Update pattern access times
        for pattern, _ in similar_patterns[:5]:  # Top 5 matches
            pattern.last_accessed = time.time()
            pattern.frequency += 1
        
        return {
            'patterns': [p for p, _ in similar_patterns[:10]],  # Top 10
            'similarities': [s for _, s in similar_patterns[:10]],
            'confidence': self._calculate_retrieval_confidence(similar_patterns),
            'total_matches': len(similar_patterns)
        }
    
    async def consolidate_experience(self, request_atom: Atom, 
                                   reasoning_results: Dict[str, Any]) -> bool:
        """
        Consolidate a reasoning experience into long-term memory.
        
        Args:
            request_atom: The original request atom
            reasoning_results: Results from reasoning process
            
        Returns:
            True if consolidation was successful
        """
        try:
            # Extract key atoms from reasoning results
            result_atoms = []
            if 'probabilistic' in reasoning_results:
                prob_result = reasoning_results['probabilistic']
                if hasattr(prob_result, 'atom'):
                    result_atoms.append(prob_result.atom)
            
            # Store episodic memory
            confidence = reasoning_results.get('confidence', 0.5)
            await self.store_experience([request_atom], result_atoms, 1.0, confidence)
            
            return True
        except Exception as e:
            logger.error(f"Failed to consolidate experience: {e}")
            return False
    
    async def start_consolidation(self):
        """Start the memory consolidation process."""
        with self._lock:
            if self._consolidation_running:
                return
            
            self._consolidation_running = True
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
            logger.info("Memory consolidation started")
    
    async def stop_consolidation(self):
        """Stop the memory consolidation process."""
        with self._lock:
            if not self._consolidation_running:
                return
            
            self._consolidation_running = False
            
            if self._consolidation_task:
                self._consolidation_task.cancel()
                try:
                    await self._consolidation_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Memory consolidation stopped")
    
    async def perform_maintenance(self):
        """Perform memory maintenance operations."""
        await self._apply_forgetting()
        await self._consolidate_similar_patterns()
        await self._cleanup_weak_patterns()
    
    async def _consolidation_loop(self):
        """Main consolidation loop."""
        while self._consolidation_running:
            try:
                await self.perform_maintenance()
                await asyncio.sleep(60.0)  # Consolidate every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _extract_patterns_from_experience(self, memory: EpisodicMemory):
        """Extract reusable patterns from an episodic memory."""
        # Simple pattern extraction - could be more sophisticated
        if len(memory.context_atoms) >= 2:
            pattern_id = f"pattern_{len(self.patterns)}"
            
            pattern = MemoryPattern(
                pattern_id=pattern_id,
                atoms=memory.context_atoms,
                frequency=1,
                last_accessed=time.time(),
                strength=memory.success_rate,
                confidence=memory.confidence,
                context={'source_memory': memory.memory_id}
            )
            
            with self._lock:
                self.patterns[pattern_id] = pattern
                
                # Update indexing
                for atom in memory.context_atoms:
                    self._pattern_by_atom[atom.uuid].add(pattern_id)
    
    def _calculate_pattern_similarity(self, query_atom: Atom, 
                                    pattern: MemoryPattern) -> float:
        """Calculate similarity between query atom and a pattern."""
        if not pattern.atoms:
            return 0.0
        
        # Check for direct atom match
        for atom in pattern.atoms:
            if atom.uuid == query_atom.uuid:
                return 1.0
        
        # Check for name/type similarity
        similarities = []
        for atom in pattern.atoms:
            name_sim = self._calculate_name_similarity(query_atom.name, atom.name)
            type_sim = 1.0 if query_atom.atom_type == atom.atom_type else 0.0
            atom_sim = (name_sim * 0.7 + type_sim * 0.3)
            similarities.append(atom_sim)
        
        return max(similarities) if similarities else 0.0
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two atom names."""
        if name1 == name2:
            return 1.0
        
        # Simple Levenshtein-based similarity
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0
        
        distance = levenshtein_distance(name1, name2)
        return 1.0 - (distance / max_len)
    
    def _calculate_retrieval_confidence(self, similar_patterns: List[Tuple[MemoryPattern, float]]) -> float:
        """Calculate confidence in pattern retrieval results."""
        if not similar_patterns:
            return 0.0
        
        # Base confidence on best similarity and pattern strength
        best_pattern, best_similarity = similar_patterns[0]
        
        confidence = (best_similarity * 0.6 + 
                     best_pattern.confidence * 0.3 + 
                     min(1.0, math.log(best_pattern.frequency + 1) * 0.1))
        
        return min(1.0, confidence)
    
    async def _apply_forgetting(self):
        """Apply forgetting mechanism to reduce memory size."""
        current_time = time.time()
        
        with self._lock:
            # Forget old episodic memories
            to_forget_episodic = []
            for memory_id, memory in self.episodic_memories.items():
                age_hours = memory.get_age() / 3600.0
                forget_probability = self.forgetting_rate * age_hours / (1.0 + memory.access_count)
                
                if forget_probability > 0.5:  # Simplified forgetting threshold
                    to_forget_episodic.append(memory_id)
            
            for memory_id in to_forget_episodic:
                self._forget_episodic_memory(memory_id)
            
            # Decay pattern strengths
            for pattern in self.patterns.values():
                age_hours = (current_time - pattern.last_accessed) / 3600.0
                decay_factor = math.exp(-self.forgetting_rate * age_hours)
                pattern.strength *= decay_factor
        
        if to_forget_episodic:
            logger.debug(f"Forgot {len(to_forget_episodic)} episodic memories")
            self._access_stats['memories_forgotten'] += len(to_forget_episodic)
    
    async def _consolidate_similar_patterns(self):
        """Consolidate similar patterns to reduce redundancy."""
        consolidation_threshold = 0.8
        
        with self._lock:
            patterns = list(self.patterns.values())
            consolidated_count = 0
            
            for i, pattern1 in enumerate(patterns):
                for pattern2 in patterns[i+1:]:
                    similarity = pattern1.similarity_to(pattern2)
                    
                    if similarity >= consolidation_threshold:
                        # Merge patterns
                        self._merge_patterns(pattern1, pattern2)
                        consolidated_count += 1
                        break  # pattern2 is now merged, skip other comparisons
        
        if consolidated_count > 0:
            logger.debug(f"Consolidated {consolidated_count} similar patterns")
            self._access_stats['consolidations_performed'] += consolidated_count
    
    def _merge_patterns(self, pattern1: MemoryPattern, pattern2: MemoryPattern):
        """Merge two similar patterns into one."""
        # Merge into pattern1, remove pattern2
        total_frequency = pattern1.frequency + pattern2.frequency
        
        # Weighted average of strengths
        pattern1.strength = ((pattern1.strength * pattern1.frequency + 
                            pattern2.strength * pattern2.frequency) / total_frequency)
        
        # Update frequency and access time
        pattern1.frequency = total_frequency
        pattern1.last_accessed = max(pattern1.last_accessed, pattern2.last_accessed)
        
        # Merge contexts
        pattern1.context.update(pattern2.context)
        
        # Remove pattern2 from indices
        for atom in pattern2.atoms:
            self._pattern_by_atom[atom.uuid].discard(pattern2.pattern_id)
        
        # Remove pattern2
        del self.patterns[pattern2.pattern_id]
    
    async def _cleanup_weak_patterns(self):
        """Remove patterns with very low strength."""
        strength_threshold = 0.1
        
        with self._lock:
            to_remove = []
            for pattern_id, pattern in self.patterns.items():
                if pattern.strength < strength_threshold:
                    to_remove.append(pattern_id)
            
            for pattern_id in to_remove:
                pattern = self.patterns[pattern_id]
                
                # Remove from indices
                for atom in pattern.atoms:
                    self._pattern_by_atom[atom.uuid].discard(pattern_id)
                
                del self.patterns[pattern_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} weak patterns")
    
    async def _forget_old_memories(self):
        """Remove old memories when capacity is exceeded."""
        # Remove oldest 10% of memories
        memories_to_remove = len(self.episodic_memories) // 10
        
        if memories_to_remove == 0:
            return
        
        # Sort by age (oldest first)
        sorted_memories = sorted(
            self.episodic_memories.items(),
            key=lambda x: x[1].timestamp
        )
        
        for i in range(memories_to_remove):
            memory_id, _ = sorted_memories[i]
            self._forget_episodic_memory(memory_id)
    
    def _forget_episodic_memory(self, memory_id: str):
        """Remove an episodic memory and update indices."""
        if memory_id not in self.episodic_memories:
            return
        
        memory = self.episodic_memories[memory_id]
        
        # Remove from context index
        for atom in memory.context_atoms:
            context_key = f"{atom.atom_type.value}:{atom.name[:50]}"
            self._memory_by_context[context_key].discard(memory_id)
            
            # Clean up empty entries
            if not self._memory_by_context[context_key]:
                del self._memory_by_context[context_key]
        
        del self.episodic_memories[memory_id]
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self._lock:
            return {
                'episodic_memories': len(self.episodic_memories),
                'patterns': len(self.patterns),
                'capacity_used': (len(self.episodic_memories) / self.capacity) * 100,
                'pattern_hit_rate': (
                    self._access_stats['pattern_hits'] / 
                    max(1, self._access_stats['pattern_hits'] + self._access_stats['pattern_misses'])
                ) * 100,
                'consolidations_performed': self._access_stats['consolidations_performed'],
                'memories_forgotten': self._access_stats['memories_forgotten'],
                'average_pattern_strength': (
                    sum(p.strength for p in self.patterns.values()) / 
                    max(1, len(self.patterns))
                ),
                'consolidation_running': self._consolidation_running
            }
    
    def clear_all_memories(self):
        """Clear all memories (for testing or reset purposes)."""
        with self._lock:
            self.patterns.clear()
            self.episodic_memories.clear()
            self._pattern_by_atom.clear()
            self._memory_by_context.clear()
            
            # Reset statistics
            self._access_stats = {
                'pattern_hits': 0,
                'pattern_misses': 0,
                'consolidations_performed': 0,
                'memories_forgotten': 0
            }
        
        logger.info("Cleared all cognitive memories")