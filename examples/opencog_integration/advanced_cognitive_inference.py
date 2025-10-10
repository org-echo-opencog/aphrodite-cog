"""
Advanced cognitive inference example demonstrating large-scale 
orchestration and acceleration capabilities.
"""

import asyncio
import time
import random
from typing import List, Dict, Any

from aphrodite.opencog.cognitive_engine import CognitiveEngine, CognitiveConfig
from aphrodite.opencog.atomspace import AtomSpaceManager, AtomType, TruthValue
from aphrodite.opencog.reasoning import ProbabilisticReasoner, LogicEngine
from aphrodite.opencog.orchestrator import InferenceOrchestrator, AttentionManager
from aphrodite.opencog.memory import CognitiveMemory
from aphrodite.opencog.accelerator import CognitiveAccelerator


async def demonstrate_large_scale_inference():
    """Demonstrate large-scale inference orchestration."""
    print("=== Large-Scale Cognitive Inference Demo ===\n")
    
    # Create cognitive architecture with high-performance settings
    config = CognitiveConfig(
        atomspace_max_size=100000,
        reasoning_threads=8,
        memory_capacity=50000,
        cognitive_cycles_per_second=100,
        max_concurrent_inferences=32,
        enable_attention_mechanism=True,
        enable_memory_consolidation=True,
        enable_cognitive_acceleration=True,
        batch_optimization=True,
        parallel_inference=True,
        adaptive_scheduling=True
    )
    
    print("1. Initializing large-scale cognitive architecture...")
    async with CognitiveEngine(config) as cognitive_engine:
        
        # Generate a large number of inference requests
        print("2. Generating large-scale inference workload...")
        
        domains = ["science", "technology", "philosophy", "mathematics", "history"]
        complexities = ["simple", "medium", "complex", "expert"]
        
        inference_requests = []
        for i in range(100):  # 100 concurrent inference requests
            domain = random.choice(domains)
            complexity = random.choice(complexities)
            
            prompt = f"Explain {complexity} concepts in {domain} domain (request {i+1})"
            context = {
                "domain": domain,
                "complexity": complexity,
                "request_id": f"req_{i+1}",
                "batch_id": i // 10,  # Group into batches of 10
                "priority": random.uniform(0.1, 1.0)
            }
            
            inference_requests.append((prompt, context))
        
        # Process requests in parallel with cognitive orchestration
        print("3. Processing requests with cognitive orchestration...")
        start_time = time.time()
        
        # Submit all requests concurrently
        tasks = []
        for prompt, context in inference_requests:
            task = asyncio.create_task(
                cognitive_engine.process_inference_request(prompt, context)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        print(f"\n4. Processing Results:")
        print(f"  Total requests: {len(inference_requests)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {len(failed_results)}")
        print(f"  Total processing time: {processing_time:.2f}s")
        print(f"  Average per request: {processing_time/len(inference_requests):.3f}s")
        print(f"  Throughput: {len(inference_requests)/processing_time:.1f} requests/second")
        
        # Analyze confidence distribution
        confidences = [r.get('confidence', 0) for r in successful_results]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Get detailed cognitive statistics
        print("\n5. Cognitive Architecture Performance:")
        stats = cognitive_engine.get_statistics()
        
        for key, value in stats.items():
            print(f"  {key}: {value}")


async def demonstrate_cognitive_reasoning():
    """Demonstrate advanced reasoning capabilities."""
    print("\n=== Advanced Cognitive Reasoning Demo ===\n")
    
    # Initialize reasoning components
    atomspace = AtomSpaceManager(max_size=10000)
    reasoner = ProbabilisticReasoner(atomspace, threshold=0.6, max_steps=100)
    logic_engine = LogicEngine(atomspace)
    
    print("1. Building knowledge base with atoms and relationships...")
    
    # Create some sample knowledge
    knowledge_facts = [
        ("Artificial Intelligence", "Computer Science", AtomType.INHERITANCE),
        ("Machine Learning", "Artificial Intelligence", AtomType.INHERITANCE),
        ("Neural Networks", "Machine Learning", AtomType.INHERITANCE),
        ("Deep Learning", "Neural Networks", AtomType.INHERITANCE),
        ("ChatGPT", "Deep Learning", AtomType.INHERITANCE),
        ("Reasoning", "Intelligence", AtomType.SIMILARITY),
        ("Learning", "Intelligence", AtomType.SIMILARITY),
    ]
    
    # Add facts to atomspace
    atoms = {}
    for subject, predicate, relation_type in knowledge_facts:
        # Create nodes
        if subject not in atoms:
            atoms[subject] = atomspace.create_node(
                subject, AtomType.CONCEPT, TruthValue(0.9, 0.9)
            )
        if predicate not in atoms:
            atoms[predicate] = atomspace.create_node(
                predicate, AtomType.CONCEPT, TruthValue(0.9, 0.9)
            )
        
        # Create relationship
        relationship = atomspace.create_link(
            f"{subject}-{relation_type.value}-{predicate}",
            [atoms[subject], atoms[predicate]],
            relation_type,
            TruthValue(0.8, 0.8)
        )
    
    print(f"   Created {atomspace.size()} atoms in knowledge base")
    
    # Perform reasoning queries
    print("\n2. Performing probabilistic reasoning queries...")
    
    reasoning_queries = [
        "ChatGPT",
        "Machine Learning", 
        "Intelligence",
        "Computer Science"
    ]
    
    for query_name in reasoning_queries:
        if query_name in atoms:
            query_atom = atoms[query_name]
            
            print(f"\n   Reasoning about: {query_name}")
            
            # Probabilistic reasoning
            prob_result = await reasoner.infer_async(query_atom)
            print(f"   Probabilistic result: confidence={prob_result.confidence:.3f}, "
                  f"reasoning_path_length={len(prob_result.reasoning_path)}")
            
            # Logical reasoning
            logic_result = await logic_engine.reason_async(query_atom)
            print(f"   Logical result: conclusions={len(logic_result['conclusions'])}, "
                  f"consistent={logic_result['is_consistent']}")


async def demonstrate_memory_and_attention():
    """Demonstrate memory consolidation and attention mechanisms."""
    print("\n=== Memory and Attention Demo ===\n")
    
    # Initialize components
    atomspace = AtomSpaceManager(max_size=5000)
    memory = CognitiveMemory(atomspace, capacity=1000, forgetting_rate=0.01)
    attention_manager = AttentionManager(atomspace, allocation_rate=0.2)
    
    print("1. Starting memory and attention systems...")
    await memory.start_consolidation()
    await attention_manager.start()
    
    try:
        # Create various experiences and patterns
        print("2. Creating cognitive experiences...")
        
        experiences = [
            (["AI", "learns", "patterns"], ["improved", "performance"], 0.9),
            (["machine", "processes", "data"], ["generates", "insights"], 0.8),
            (["neural", "network", "training"], ["optimized", "weights"], 0.85),
            (["cognitive", "architecture", "reasoning"], ["enhanced", "intelligence"], 0.95),
        ]
        
        # Store experiences and allocate attention
        for i, (context_words, result_words, success_rate) in enumerate(experiences):
            # Create atoms for context and results
            context_atoms = []
            for word in context_words:
                atom = atomspace.create_node(f"context_{word}_{i}", AtomType.CONCEPT)
                context_atoms.append(atom)
            
            result_atoms = []
            for word in result_words:
                atom = atomspace.create_node(f"result_{word}_{i}", AtomType.CONCEPT)
                result_atoms.append(atom)
            
            # Store experience in memory
            memory_id = await memory.store_experience(
                context_atoms, result_atoms, success_rate, 0.8
            )
            
            print(f"   Stored experience {i+1}: {memory_id}")
            
            # Allocate attention to important atoms
            for atom in context_atoms + result_atoms:
                importance = success_rate * 0.8
                urgency = 0.6 if i < 2 else 0.4
                await attention_manager.allocate_attention(atom, importance, urgency)
        
        # Wait for some consolidation
        await asyncio.sleep(2.0)
        
        # Test pattern matching
        print("\n3. Testing pattern matching and retrieval...")
        
        # Create a query similar to stored experiences
        query_atom = atomspace.create_node("query_AI_learning", AtomType.CONCEPT)
        
        # Find similar patterns
        similar_patterns = await memory.find_similar_patterns_async(query_atom)
        
        print(f"   Found {len(similar_patterns['patterns'])} similar patterns")
        print(f"   Retrieval confidence: {similar_patterns['confidence']:.3f}")
        
        # Get memory statistics
        memory_stats = memory.get_memory_statistics()
        print(f"\n4. Memory System Statistics:")
        for key, value in memory_stats.items():
            print(f"   {key}: {value}")
        
    finally:
        # Cleanup
        await memory.stop_consolidation()
        await attention_manager.stop()


async def demonstrate_cognitive_acceleration():
    """Demonstrate cognitive acceleration capabilities."""
    print("\n=== Cognitive Acceleration Demo ===\n")
    
    # Initialize acceleration components
    config = CognitiveConfig(enable_cognitive_acceleration=True)
    atomspace = AtomSpaceManager(max_size=5000)
    accelerator = CognitiveAccelerator(config, atomspace)
    
    print("1. Testing cognitive acceleration strategies...")
    
    # Create mock reasoning results to accelerate
    test_cases = [
        {
            'confidence': 0.9,
            'probabilistic': {'strength': 0.85, 'confidence': 0.8},
            'logical': {'conclusions': ['A', 'B'], 'consistent': True},
            'request_atom': atomspace.create_node("high_confidence_query", AtomType.CONCEPT)
        },
        {
            'confidence': 0.5,
            'probabilistic': {'strength': 0.6, 'confidence': 0.5},
            'logical': {'conclusions': ['C'], 'consistent': True},
            'request_atom': atomspace.create_node("medium_confidence_query", AtomType.CONCEPT)
        },
        {
            'confidence': 0.2,
            'probabilistic': {'strength': 0.3, 'confidence': 0.4},
            'logical': {'conclusions': [], 'consistent': False},
            'request_atom': atomspace.create_node("low_confidence_query", AtomType.CONCEPT)
        }
    ]
    
    for i, reasoning_results in enumerate(test_cases, 1):
        print(f"\n   Test case {i} (original confidence: {reasoning_results['confidence']})...")
        
        # Apply acceleration
        start_time = time.time()
        accelerated_results = await accelerator.accelerate_inference(reasoning_results)
        acceleration_time = time.time() - start_time
        
        # Extract acceleration metadata
        metadata = accelerated_results.get('acceleration_metadata', {})
        strategy = metadata.get('strategy', 'none')
        speedup = metadata.get('speedup_factor', 1.0)
        new_confidence = accelerated_results.get('confidence', reasoning_results['confidence'])
        
        print(f"     Strategy applied: {strategy}")
        print(f"     Speedup factor: {speedup:.2f}x")
        print(f"     Confidence boost: {new_confidence - reasoning_results['confidence']:+.3f}")
        print(f"     Processing time: {acceleration_time:.4f}s")
    
    # Update optimization patterns
    print("\n2. Learning optimization patterns...")
    await accelerator.update_optimization_patterns()
    
    # Get acceleration statistics
    acceleration_stats = accelerator.get_acceleration_statistics()
    print(f"\n3. Acceleration Statistics:")
    for key, value in acceleration_stats.items():
        print(f"   {key}: {value}")


async def main():
    """Run all advanced cognitive demos."""
    print("Starting Advanced Cognitive Inference Demonstrations...\n")
    
    try:
        await demonstrate_large_scale_inference()
        await demonstrate_cognitive_reasoning()
        await demonstrate_memory_and_attention()
        await demonstrate_cognitive_acceleration()
        
        print("\n=== All demonstrations completed successfully! ===")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())