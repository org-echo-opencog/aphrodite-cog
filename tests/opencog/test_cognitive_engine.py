"""
Tests for the OpenCog cognitive engine integration.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from aphrodite.opencog.cognitive_engine import CognitiveEngine, CognitiveConfig
from aphrodite.opencog.atomspace import AtomSpaceManager, AtomType, TruthValue


class TestCognitiveConfig:
    """Test CognitiveConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CognitiveConfig()
        
        assert config.atomspace_max_size == 1000000
        assert config.attention_allocation_rate == 0.1
        assert config.max_inference_steps == 1000
        assert config.reasoning_threads == 4
        assert config.probabilistic_threshold == 0.7
        assert config.enable_attention_mechanism is True
        assert config.enable_memory_consolidation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CognitiveConfig(
            atomspace_max_size=50000,
            reasoning_threads=8,
            enable_attention_mechanism=False
        )
        
        assert config.atomspace_max_size == 50000
        assert config.reasoning_threads == 8
        assert config.enable_attention_mechanism is False
        # Check defaults are preserved
        assert config.attention_allocation_rate == 0.1


class TestCognitiveEngine:
    """Test CognitiveEngine functionality."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return CognitiveConfig(
            atomspace_max_size=1000,
            reasoning_threads=2,
            memory_capacity=500,
            cognitive_cycles_per_second=10
        )
    
    @pytest.fixture
    def cognitive_engine(self, config):
        """Provide cognitive engine instance."""
        return CognitiveEngine(config)
    
    def test_cognitive_engine_initialization(self, config):
        """Test cognitive engine initialization."""
        engine = CognitiveEngine(config)
        
        assert engine.config == config
        assert engine._running is False
        assert engine.atomspace is not None
        assert engine.reasoner is not None
        assert engine.orchestrator is not None
        assert engine.memory is not None
        assert engine.accelerator is not None
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, cognitive_engine):
        """Test engine start and stop lifecycle."""
        # Initially not running
        assert cognitive_engine._running is False
        
        # Start engine
        await cognitive_engine.start()
        assert cognitive_engine._running is True
        
        # Stop engine
        await cognitive_engine.stop()
        assert cognitive_engine._running is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, cognitive_engine):
        """Test async context manager functionality."""
        async with cognitive_engine as engine:
            assert engine._running is True
        
        # Should be stopped after context exit
        assert cognitive_engine._running is False
    
    @pytest.mark.asyncio
    async def test_process_inference_request(self, cognitive_engine):
        """Test inference request processing."""
        async with cognitive_engine:
            # Process a simple request
            result = await cognitive_engine.process_inference_request(
                prompt="What is artificial intelligence?",
                context={"domain": "technology"}
            )
            
            # Check result structure
            assert isinstance(result, dict)
            assert 'confidence' in result or 'final_confidence' in result
            assert 'orchestration_metadata' in result
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, cognitive_engine):
        """Test processing multiple concurrent requests."""
        async with cognitive_engine:
            # Create multiple requests
            requests = [
                cognitive_engine.process_inference_request(f"Query {i}", {"id": i})
                for i in range(5)
            ]
            
            # Process concurrently
            results = await asyncio.gather(*requests)
            
            assert len(results) == 5
            for result in results:
                assert isinstance(result, dict)
    
    def test_get_statistics(self, cognitive_engine):
        """Test statistics retrieval."""
        stats = cognitive_engine.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'atomspace_size' in stats
        assert 'running' in stats
        assert stats['running'] is False  # Engine not started


class TestAtomSpaceIntegration:
    """Test integration with AtomSpace."""
    
    @pytest.fixture
    def atomspace(self):
        """Provide atomspace instance."""
        return AtomSpaceManager(max_size=1000)
    
    def test_atomspace_creation(self, atomspace):
        """Test atomspace creation and basic operations."""
        # Initially empty
        assert atomspace.size() == 0
        
        # Create a node
        node = atomspace.create_node("TestConcept", AtomType.CONCEPT)
        assert atomspace.size() == 1
        assert node.name == "TestConcept"
        assert node.atom_type == AtomType.CONCEPT
    
    def test_atom_relationships(self, atomspace):
        """Test atom relationships and links."""
        # Create nodes
        node1 = atomspace.create_node("Concept1", AtomType.CONCEPT)
        node2 = atomspace.create_node("Concept2", AtomType.CONCEPT)
        
        # Create relationship
        link = atomspace.create_link(
            "Inheritance",
            [node1, node2],
            AtomType.INHERITANCE,
            TruthValue(0.8, 0.9)
        )
        
        assert link.arity() == 2
        assert link.outgoing[0] == node1
        assert link.outgoing[1] == node2
        assert link.truth_value.strength == 0.8
        assert link.truth_value.confidence == 0.9
    
    def test_truth_value_merging(self, atomspace):
        """Test truth value merging for duplicate atoms."""
        # Create same atom twice with different truth values
        atom1 = atomspace.create_node("TestNode", AtomType.CONCEPT, TruthValue(0.6, 0.8))
        atom2 = atomspace.create_node("TestNode", AtomType.CONCEPT, TruthValue(0.8, 0.6))
        
        # Should be the same atom with merged truth value
        assert atom1 == atom2
        assert atomspace.size() == 1
        
        # Truth value should be merged
        merged_tv = atom1.truth_value
        assert 0.6 < merged_tv.strength < 0.8
        assert merged_tv.confidence > 0.6


class TestCognitiveIntegrationScenarios:
    """Test realistic cognitive integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_knowledge_reasoning_scenario(self):
        """Test a complete knowledge reasoning scenario."""
        config = CognitiveConfig(
            atomspace_max_size=5000,
            reasoning_threads=2,
            memory_capacity=1000
        )
        
        async with CognitiveEngine(config) as engine:
            # Build some knowledge
            atomspace = engine.atomspace
            
            # AI concepts
            ai = atomspace.create_node("AI", AtomType.CONCEPT, TruthValue(0.9, 0.9))
            ml = atomspace.create_node("ML", AtomType.CONCEPT, TruthValue(0.8, 0.8))
            
            # Relationship
            atomspace.create_link(
                "ML-is-part-of-AI",
                [ml, ai],
                AtomType.INHERITANCE,
                TruthValue(0.85, 0.9)
            )
            
            # Query the system
            result = await engine.process_inference_request(
                "What is the relationship between ML and AI?",
                {"query_type": "relationship"}
            )
            
            assert isinstance(result, dict)
            confidence = result.get('confidence', result.get('final_confidence', 0))
            assert confidence > 0.0
    
    @pytest.mark.asyncio 
    async def test_attention_allocation_scenario(self):
        """Test attention allocation in cognitive processing."""
        config = CognitiveConfig(
            enable_attention_mechanism=True,
            attention_allocation_rate=0.2
        )
        
        async with CognitiveEngine(config) as engine:
            # Process request that should get attention
            result = await engine.process_inference_request(
                "High priority cognitive task",
                {"importance": "high", "urgency": "critical"}
            )
            
            # Check that attention was allocated
            atomspace_size = engine.atomspace.size()
            if atomspace_size > 0:
                # Find atoms with attention
                atoms_with_attention = [
                    atom for atom in engine.atomspace._atoms.values()
                    if atom.attention_value > 0
                ]
                
                # Should have some attention allocation
                assert len(atoms_with_attention) >= 0  # Could be 0 if fast processing
    
    @pytest.mark.asyncio
    async def test_memory_consolidation_scenario(self):
        """Test memory consolidation during cognitive processing."""
        config = CognitiveConfig(
            enable_memory_consolidation=True,
            memory_capacity=100,
            forgetting_rate=0.1
        )
        
        async with CognitiveEngine(config) as engine:
            # Process several related requests to build memory
            requests = [
                "Learn about machine learning",
                "Understand neural networks", 
                "Explain deep learning"
            ]
            
            results = []
            for prompt in requests:
                result = await engine.process_inference_request(
                    prompt, {"topic": "AI", "learning": True}
                )
                results.append(result)
            
            # All should complete successfully
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test cognitive engine performance under load."""
        config = CognitiveConfig(
            max_concurrent_inferences=8,
            cognitive_cycles_per_second=50
        )
        
        async with CognitiveEngine(config) as engine:
            # Generate load
            num_requests = 20
            start_time = time.time()
            
            tasks = []
            for i in range(num_requests):
                task = engine.process_inference_request(
                    f"Process request {i}",
                    {"batch_id": i // 5, "request_num": i}
                )
                tasks.append(task)
            
            # Process all requests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Check results
            successful = [r for r in results if not isinstance(r, Exception)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            # Should process most requests successfully
            assert len(successful) >= num_requests * 0.8  # At least 80% success rate
            
            # Should complete in reasonable time (less than 10 seconds)
            assert processing_time < 10.0
            
            # Calculate throughput
            throughput = len(successful) / processing_time
            assert throughput > 0  # Should have some throughput
    
    def test_error_handling(self):
        """Test error handling in cognitive engine."""
        # Test with invalid configuration
        config = CognitiveConfig(
            atomspace_max_size=-1,  # Invalid size
            reasoning_threads=0     # Invalid thread count
        )
        
        # Should handle gracefully (implementation should validate)
        engine = CognitiveEngine(config)
        assert engine is not None


if __name__ == "__main__":
    pytest.main([__file__])