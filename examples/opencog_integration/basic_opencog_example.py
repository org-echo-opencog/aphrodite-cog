"""
Basic example of OpenCog integration with Aphrodite Engine.

Demonstrates how to use the cognitive architecture for enhanced
large-scale inference with pattern learning and memory consolidation.
"""

import asyncio
from typing import List, Optional

from aphrodite import EngineArgs, SamplingParams
from aphrodite.engine.aphrodite_engine import AphroditeEngine
from aphrodite.opencog.cognitive_engine import CognitiveEngine, CognitiveConfig
from aphrodite.opencog.integration import (
    OpenCogAphroditeEngine, 
    OpenCogAphroditeEngineBuilder,
    OpenCogAphroditeConfig
)


async def run_basic_opencog_example():
    """Run basic OpenCog integration example."""
    print("=== Basic OpenCog Integration Example ===\n")
    
    # Initialize base Aphrodite engine
    print("1. Initializing base Aphrodite engine...")
    engine_args = EngineArgs(
        model="microsoft/DialoGPT-medium",  # Using a smaller model for demo
        max_model_len=512,
        max_num_seqs=16,
    )
    
    try:
        base_engine = AphroditeEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"Note: Could not initialize model (expected in test environment): {e}")
        # Create a mock engine for demonstration
        base_engine = MockAphroditeEngine()
    
    # Configure cognitive architecture
    print("2. Configuring OpenCog cognitive architecture...")
    cognitive_config = CognitiveConfig(
        atomspace_max_size=10000,
        reasoning_threads=2,
        memory_capacity=5000,
        cognitive_cycles_per_second=50,
        enable_attention_mechanism=True,
        enable_memory_consolidation=True,
        enable_cognitive_acceleration=True
    )
    
    # Create enhanced engine
    print("3. Creating OpenCog-enhanced Aphrodite engine...")
    enhanced_engine = OpenCogAphroditeEngineBuilder.create_enhanced_engine(
        base_engine=base_engine,
        cognitive_config=cognitive_config,
        enable_all_features=True
    )
    
    # Start the enhanced engine
    print("4. Starting cognitive architecture...")
    async with enhanced_engine:
        
        # Example prompts for demonstration
        test_prompts = [
            "Explain the concept of artificial intelligence.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the process of photosynthesis.",
            "What is the theory of relativity?"
        ]
        
        print("5. Processing prompts with cognitive enhancement...\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Processing prompt {i}: {prompt[:50]}...")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=128,
            )
            
            # Submit request with cognitive enhancement
            request_id = f"req_{i}"
            cognitive_request_id = await enhanced_engine.add_request_with_cognitive_enhancement(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
                priority=1 if i <= 2 else 0,  # Higher priority for first two
                enable_cognitive=True
            )
            
            print(f"  -> Cognitive request ID: {cognitive_request_id}")
        
        # Process requests with cognitive orchestration
        print("\n6. Processing requests with cognitive orchestration...")
        for step in range(10):  # Process several steps
            outputs = await enhanced_engine.step_with_cognitive_orchestration()
            
            if outputs:
                print(f"  Step {step + 1}: Generated {len(outputs)} outputs")
                for output in outputs:
                    # Check for cognitive metadata
                    cognitive_metadata = getattr(output, 'cognitive_metadata', None)
                    if cognitive_metadata:
                        confidence = cognitive_metadata.get('cognitive_confidence', 0.0)
                        patterns = cognitive_metadata.get('cognitive_patterns_matched', 0)
                        print(f"    -> Cognitive confidence: {confidence:.3f}, Patterns matched: {patterns}")
            
            await asyncio.sleep(0.1)  # Small delay between steps
        
        # Get cognitive statistics
        print("\n7. Cognitive Performance Statistics:")
        stats = enhanced_engine.get_cognitive_statistics()
        
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Cognitively enhanced: {stats.get('cognitive_enhanced_requests', 0)}")
        print(f"  Average cognitive overhead: {stats.get('average_cognitive_overhead', 0):.3f}s")
        print(f"  AtomSpace size: {stats.get('atomspace_size', 0)}")
        print(f"  Memory patterns: {stats.get('memory_patterns', 0)}")
        print(f"  Attention allocated atoms: {stats.get('attention_allocated_atoms', 0)}")
    
    print("\n=== Example completed successfully! ===")


async def run_cognitive_architecture_demo():
    """Demonstrate standalone cognitive architecture capabilities."""
    print("\n=== Cognitive Architecture Demo ===\n")
    
    # Create and configure cognitive engine
    config = CognitiveConfig(
        atomspace_max_size=5000,
        reasoning_threads=2,
        memory_capacity=1000,
        cognitive_cycles_per_second=20
    )
    
    async with CognitiveEngine(config) as cognitive_engine:
        print("1. Cognitive engine started")
        
        # Test prompts for cognitive processing
        test_cases = [
            ("What is consciousness?", {"domain": "philosophy"}),
            ("How do neural networks learn?", {"domain": "machine_learning"}),
            ("Explain quantum entanglement", {"domain": "physics"}),
        ]
        
        print("2. Processing queries through cognitive architecture...")
        
        for i, (prompt, context) in enumerate(test_cases, 1):
            print(f"\nQuery {i}: {prompt}")
            
            result = await cognitive_engine.process_inference_request(
                prompt=prompt,
                context=context
            )
            
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            print(f"  Patterns found: {len(result.get('patterns', []))}")
            print(f"  Memory matches: {len(result.get('memory_matches', []))}")
        
        # Get final statistics
        final_stats = cognitive_engine.get_statistics()
        print(f"\n3. Final Statistics:")
        print(f"  AtomSpace size: {final_stats.get('atomspace_size', 0)}")
        print(f"  Memory patterns: {final_stats.get('memory_patterns', 0)}")
        print(f"  Reasoning cache size: {final_stats.get('reasoning_cache_size', 0)}")


class MockAphroditeEngine:
    """Mock Aphrodite engine for testing when real model is not available."""
    
    def __init__(self):
        self.request_counter = 0
        
    def add_request(self, request_id: str, prompt, params, **kwargs):
        """Mock add request method."""
        self.request_counter += 1
        
    def step(self):
        """Mock step method."""
        # Return mock outputs
        return [MockRequestOutput(f"mock_output_{i}") for i in range(min(2, self.request_counter))]
    
    def from_engine_args(cls, engine_args):
        """Mock class method."""
        return cls()


class MockRequestOutput:
    """Mock request output for testing."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.outputs = [MockOutput("This is a mock response for testing purposes.")]


class MockOutput:
    """Mock output for testing."""
    
    def __init__(self, text: str):
        self.text = text


async def main():
    """Run all examples."""
    try:
        await run_basic_opencog_example()
        await run_cognitive_architecture_demo()
    except Exception as e:
        print(f"Example error (expected in test environment): {e}")
        print("This example demonstrates the OpenCog integration structure.")


if __name__ == "__main__":
    asyncio.run(main())