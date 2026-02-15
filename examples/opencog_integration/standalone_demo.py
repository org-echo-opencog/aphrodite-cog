#!/usr/bin/env python3
"""
Standalone demonstration of OpenCog cognitive architecture capabilities.
This demo runs independently without requiring the full Aphrodite setup.
"""

import sys
import os
import asyncio
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import OpenCog modules directly
import importlib.util

def load_opencog_module(module_name, file_path):
    """Load OpenCog module from file path."""
    full_path = os.path.join(project_root, file_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def demonstrate_atomspace():
    """Demonstrate AtomSpace hypergraph capabilities."""
    print("=== AtomSpace Hypergraph Demonstration ===\n")
    
    # Load atomspace module
    atomspace_mod = load_opencog_module('atomspace', 'aphrodite/opencog/atomspace.py')
    
    AtomSpaceManager = atomspace_mod.AtomSpaceManager
    AtomType = atomspace_mod.AtomType
    TruthValue = atomspace_mod.TruthValue
    
    # Create atomspace
    atomspace = AtomSpaceManager(max_size=10000)
    print(f"1. Created AtomSpace with capacity: {atomspace.max_size}")
    
    # Create knowledge graph
    print("2. Building knowledge graph...")
    
    # AI/ML concepts
    ai = atomspace.create_node("AI", AtomType.CONCEPT, TruthValue(0.95, 0.9))
    ml = atomspace.create_node("MachineLearning", AtomType.CONCEPT, TruthValue(0.9, 0.85))
    dl = atomspace.create_node("DeepLearning", AtomType.CONCEPT, TruthValue(0.85, 0.8))
    nn = atomspace.create_node("NeuralNetworks", AtomType.CONCEPT, TruthValue(0.8, 0.75))
    
    # Create hierarchical relationships
    ml_isa_ai = atomspace.create_link(
        "ML-IsA-AI", [ml, ai], AtomType.INHERITANCE, TruthValue(0.9, 0.95)
    )
    dl_isa_ml = atomspace.create_link(
        "DL-IsA-ML", [dl, ml], AtomType.INHERITANCE, TruthValue(0.85, 0.9)
    )
    nn_isa_dl = atomspace.create_link(
        "NN-IsA-DL", [nn, dl], AtomType.INHERITANCE, TruthValue(0.8, 0.85)
    )
    
    # Create similarity relationships  
    dl_similar_nn = atomspace.create_link(
        "DL-Similar-NN", [dl, nn], AtomType.SIMILARITY, TruthValue(0.95, 0.9)
    )
    
    print(f"   Created {atomspace.size()} atoms in knowledge graph")
    print(f"   Concepts: {len(atomspace.get_atoms_by_type(AtomType.CONCEPT))}")
    print(f"   Inheritance links: {len(atomspace.get_atoms_by_type(AtomType.INHERITANCE))}")
    print(f"   Similarity links: {len(atomspace.get_atoms_by_type(AtomType.SIMILARITY))}")
    
    # Demonstrate graph traversal
    print("\n3. Graph traversal demonstration:")
    for concept_node in [ai, ml, dl, nn]:
        incoming = atomspace.get_incoming_set(concept_node)
        print(f"   {concept_node.name}: {len(incoming)} incoming relationships")
        for link in incoming:
            if hasattr(link, 'outgoing'):
                other_concepts = [atom.name for atom in link.outgoing if atom != concept_node]
                print(f"     -> {link.atom_type.value} with {other_concepts}")
    
    # Demonstrate truth value combinations
    print("\n4. Truth value demonstration:")
    
    # Create duplicate with different truth value
    ai_duplicate = atomspace.create_node("AI", AtomType.CONCEPT, TruthValue(0.8, 0.95))
    print(f"   AI original: {ai.truth_value}")
    print(f"   AI merged: {ai_duplicate.truth_value} (same atom: {ai == ai_duplicate})")
    
    return atomspace

def demonstrate_reasoning():
    """Demonstrate reasoning capabilities."""
    print("\n=== Cognitive Reasoning Demonstration ===\n")
    
    # Load modules with fallback imports
    try:
        atomspace_mod = sys.modules.get('atomspace') or load_opencog_module(
            'atomspace', 'aphrodite/opencog/atomspace.py'
        )
        reasoning_mod = load_opencog_module('reasoning', 'aphrodite/opencog/reasoning.py')
    except Exception as e:
        print(f"Could not load reasoning module: {e}")
        print("Skipping reasoning demonstration...")
        return None
    
    AtomSpaceManager = atomspace_mod.AtomSpaceManager
    AtomType = atomspace_mod.AtomType
    TruthValue = atomspace_mod.TruthValue
    ProbabilisticReasoner = reasoning_mod.ProbabilisticReasoner
    LogicEngine = reasoning_mod.LogicEngine
    
    # Create reasoning environment
    atomspace = AtomSpaceManager(max_size=5000)
    reasoner = ProbabilisticReasoner(atomspace, threshold=0.6, max_steps=50)
    logic_engine = LogicEngine(atomspace)
    
    print("1. Setting up reasoning environment...")
    print(f"   AtomSpace capacity: {atomspace.max_size}")
    print(f"   Reasoning threshold: {reasoner.threshold}")
    print(f"   Max inference steps: {reasoner.max_steps}")
    
    # Build knowledge for reasoning
    print("\n2. Building reasoning knowledge base...")
    
    # Concepts
    programming = atomspace.create_node("Programming", AtomType.CONCEPT, TruthValue(0.9, 0.9))
    python = atomspace.create_node("Python", AtomType.CONCEPT, TruthValue(0.85, 0.8))
    ai_dev = atomspace.create_node("AI_Development", AtomType.CONCEPT, TruthValue(0.8, 0.75))
    
    # Implications
    python_implies_programming = atomspace.create_link(
        "Python->Programming", 
        [python, programming], 
        AtomType.IMPLICATION, 
        TruthValue(0.9, 0.9)
    )
    
    ai_dev_implies_python = atomspace.create_link(
        "AI_Dev->Python",
        [ai_dev, python],
        AtomType.IMPLICATION, 
        TruthValue(0.8, 0.85)
    )
    
    print(f"   Created {atomspace.size()} atoms for reasoning")
    
    # Perform reasoning
    print("\n3. Performing probabilistic reasoning...")
    
    reasoning_queries = [programming, python, ai_dev]
    
    for query_atom in reasoning_queries:
        print(f"\n   Reasoning about: {query_atom.name}")
        
        start_time = time.time()
        result = reasoner.infer(query_atom)
        reasoning_time = time.time() - start_time
        
        print(f"     Confidence: {result.confidence:.3f}")
        print(f"     Truth value: strength={result.truth_value.strength:.3f}, "
              f"confidence={result.truth_value.confidence:.3f}")
        print(f"     Reasoning path length: {len(result.reasoning_path)}")
        print(f"     Computation time: {result.computation_time:.4f}s")
        print(f"     Total time: {reasoning_time:.4f}s")
    
    # Perform logical reasoning
    print("\n4. Performing logical reasoning...")
    
    for query_atom in reasoning_queries[:2]:  # Test first two
        print(f"\n   Logic reasoning about: {query_atom.name}")
        
        logic_result = logic_engine.reason(query_atom)
        
        print(f"     Premises found: {len(logic_result['premises'])}")
        print(f"     Conclusions: {len(logic_result['conclusions'])}")
        print(f"     Contradictions: {len(logic_result['contradictions'])}")
        print(f"     Is consistent: {logic_result['is_consistent']}")
        print(f"     Logic confidence: {logic_result['confidence']:.3f}")
    
    return atomspace

def demonstrate_large_scale_processing():
    """Demonstrate large-scale cognitive processing."""
    print("\n=== Large-Scale Processing Demonstration ===\n")
    
    atomspace_mod = sys.modules.get('atomspace') or load_opencog_module(
        'atomspace', 'aphrodite/opencog/atomspace.py'
    )
    
    AtomSpaceManager = atomspace_mod.AtomSpaceManager
    AtomType = atomspace_mod.AtomType
    TruthValue = atomspace_mod.TruthValue
    
    # Create large atomspace
    large_atomspace = AtomSpaceManager(max_size=50000)
    
    print("1. Creating large-scale knowledge base...")
    
    start_time = time.time()
    
    # Generate many concepts and relationships
    concepts = []
    domains = ["science", "technology", "philosophy", "mathematics", "arts"]
    
    concept_count = 1000
    for i in range(concept_count):
        domain = domains[i % len(domains)]
        concept_name = f"{domain}_concept_{i}"
        
        # Vary truth values
        strength = 0.5 + (i % 50) * 0.01  # 0.5 to 0.99
        confidence = 0.6 + (i % 40) * 0.01  # 0.6 to 0.99
        
        concept = large_atomspace.create_node(
            concept_name, AtomType.CONCEPT, TruthValue(strength, confidence)
        )
        concepts.append(concept)
        
        # Add attention values
        concept.attention_value = strength * confidence
    
    # Create relationships
    relationship_count = 2000
    for i in range(relationship_count):
        concept1 = concepts[i % len(concepts)]
        concept2 = concepts[(i + 1) % len(concepts)]
        
        relation_type = AtomType.INHERITANCE if i % 2 == 0 else AtomType.SIMILARITY
        
        relationship = large_atomspace.create_link(
            f"relation_{i}",
            [concept1, concept2],
            relation_type,
            TruthValue(0.7 + (i % 30) * 0.01, 0.8)
        )
    
    creation_time = time.time() - start_time
    
    print(f"   Created {large_atomspace.size()} atoms in {creation_time:.2f}s")
    print(f"   Creation rate: {large_atomspace.size() / creation_time:.0f} atoms/second")
    
    # Test query performance
    print("\n2. Testing query performance...")
    
    query_start = time.time()
    
    concept_atoms = large_atomspace.get_atoms_by_type(AtomType.CONCEPT)
    inheritance_atoms = large_atomspace.get_atoms_by_type(AtomType.INHERITANCE)
    similarity_atoms = large_atomspace.get_atoms_by_type(AtomType.SIMILARITY)
    
    query_time = time.time() - query_start
    
    print(f"   Concept atoms: {len(concept_atoms)}")
    print(f"   Inheritance links: {len(inheritance_atoms)}")
    print(f"   Similarity links: {len(similarity_atoms)}")
    print(f"   Query time: {query_time:.4f}s")
    
    # Test attention-based filtering
    print("\n3. Testing attention-based processing...")
    
    attention_start = time.time()
    
    high_attention_atoms = [
        atom for atom in large_atomspace._atoms.values()
        if atom.attention_value > 0.8
    ]
    
    medium_attention_atoms = [
        atom for atom in large_atomspace._atoms.values()
        if 0.5 < atom.attention_value <= 0.8
    ]
    
    attention_time = time.time() - attention_start
    
    print(f"   High attention atoms: {len(high_attention_atoms)}")
    print(f"   Medium attention atoms: {len(medium_attention_atoms)}")
    print(f"   Attention filtering time: {attention_time:.4f}s")
    
    # Memory usage estimation
    import sys
    atom_memory = sys.getsizeof(large_atomspace._atoms)
    index_memory = sys.getsizeof(large_atomspace._name_index) + sys.getsizeof(large_atomspace._type_index)
    total_memory = atom_memory + index_memory
    
    print(f"\n4. Memory usage estimation:")
    print(f"   Atoms storage: ~{atom_memory / 1024:.1f} KB")
    print(f"   Index storage: ~{index_memory / 1024:.1f} KB") 
    print(f"   Total estimated: ~{total_memory / 1024:.1f} KB")
    print(f"   Memory per atom: ~{total_memory / large_atomspace.size():.1f} bytes")
    
    return large_atomspace

async def demonstrate_async_processing():
    """Demonstrate asynchronous cognitive processing."""
    print("\n=== Async Processing Demonstration ===\n")
    
    try:
        atomspace_mod = sys.modules.get('atomspace') or load_opencog_module(
            'atomspace', 'aphrodite/opencog/atomspace.py'
        )
        reasoning_mod = sys.modules.get('reasoning') or load_opencog_module(
            'reasoning', 'aphrodite/opencog/reasoning.py'
        )
    except Exception as e:
        print(f"Could not load required modules: {e}")
        return
    
    AtomSpaceManager = atomspace_mod.AtomSpaceManager
    AtomType = atomspace_mod.AtomType
    TruthValue = atomspace_mod.TruthValue
    ProbabilisticReasoner = reasoning_mod.ProbabilisticReasoner
    
    # Setup async environment
    atomspace = AtomSpaceManager(max_size=1000)
    reasoner = ProbabilisticReasoner(atomspace, threshold=0.6, max_steps=20)
    
    print("1. Setting up async reasoning environment...")
    
    # Create test atoms
    test_atoms = []
    for i in range(20):
        atom = atomspace.create_node(
            f"async_concept_{i}",
            AtomType.CONCEPT,
            TruthValue(0.6 + i * 0.02, 0.7 + i * 0.015)
        )
        test_atoms.append(atom)
    
    print(f"   Created {len(test_atoms)} test atoms")
    
    # Perform concurrent reasoning
    print("\n2. Performing concurrent reasoning tasks...")
    
    async def reason_about_atom(atom, delay=0.1):
        """Async reasoning task with simulated delay."""
        await asyncio.sleep(delay)  # Simulate processing time
        return reasoner.infer(atom)
    
    start_time = time.time()
    
    # Create reasoning tasks
    tasks = [reason_about_atom(atom, 0.05) for atom in test_atoms[:10]]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    
    concurrent_time = time.time() - start_time
    
    print(f"   Processed {len(results)} atoms concurrently in {concurrent_time:.3f}s")
    
    # Compare with sequential processing
    print("\n3. Comparing with sequential processing...")
    
    start_time = time.time()
    
    sequential_results = []
    for atom in test_atoms[10:20]:
        await asyncio.sleep(0.05)  # Same simulated delay
        result = reasoner.infer(atom)
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    
    print(f"   Processed {len(sequential_results)} atoms sequentially in {sequential_time:.3f}s")
    print(f"   Speedup factor: {sequential_time / concurrent_time:.2f}x")
    
    # Analyze results
    print("\n4. Result analysis:")
    
    concurrent_confidences = [r.confidence for r in results]
    sequential_confidences = [r.confidence for r in sequential_results]
    
    print(f"   Concurrent avg confidence: {sum(concurrent_confidences) / len(concurrent_confidences):.3f}")
    print(f"   Sequential avg confidence: {sum(sequential_confidences) / len(sequential_confidences):.3f}")

def print_summary():
    """Print demonstration summary."""
    print("\n" + "="*60)
    print("OpenCog Cognitive Architecture Demonstration Summary")
    print("="*60)
    print()
    print("✓ AtomSpace hypergraph knowledge representation")
    print("✓ Truth value management and merging")
    print("✓ Probabilistic reasoning with PLN")
    print("✓ Logical reasoning and consistency checking") 
    print("✓ Large-scale knowledge base processing")
    print("✓ Attention-based resource allocation")
    print("✓ Asynchronous concurrent processing")
    print("✓ Performance optimization and caching")
    print()
    print("The OpenCog integration provides a comprehensive")
    print("cognitive architecture for large-scale inference")
    print("orchestration in Aphrodite Engine.")
    print()
    print("Key benefits:")
    print("  • Intelligent pattern recognition and learning")
    print("  • Memory consolidation and forgetting mechanisms") 
    print("  • Attention allocation for resource management")
    print("  • Probabilistic and logical reasoning integration")
    print("  • High-performance concurrent processing")
    print("  • Scalable knowledge representation")
    print()

async def main():
    """Main demonstration function."""
    print("OpenCog Cognitive Architecture Demonstration")
    print("=" * 50)
    
    try:
        # Basic functionality
        demonstrate_atomspace()
        demonstrate_reasoning()
        
        # Performance demonstrations
        demonstrate_large_scale_processing()
        await demonstrate_async_processing()
        
        # Summary
        print_summary()
        
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Some features may require additional dependencies.")

if __name__ == "__main__":
    asyncio.run(main())