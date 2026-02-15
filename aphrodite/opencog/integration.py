"""
OpenCog integration layer for Aphrodite Engine.

Provides seamless integration of cognitive architecture capabilities
with the existing Aphrodite inference engine for enhanced performance
and intelligent orchestration.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import time
import logging
logger = logging.getLogger(__name__)

from ..engine.aphrodite_engine import AphroditeEngine
from ..common.sampling_params import SamplingParams
from ..common.outputs import RequestOutput
from ..inputs import PromptType
from ..lora.request import LoRARequest
from .cognitive_engine import CognitiveEngine, CognitiveConfig
from .atomspace import AtomSpaceManager, AtomType, TruthValue


@dataclass
class OpenCogAphroditeConfig:
    """Configuration for OpenCog-enhanced Aphrodite engine."""
    
    # Base Aphrodite configuration would be passed separately
    enable_opencog: bool = True
    
    # Cognitive architecture settings
    cognitive_config: Optional[CognitiveConfig] = None
    
    # Integration settings
    cognitive_preprocessing: bool = True
    cognitive_postprocessing: bool = True
    cognitive_orchestration: bool = True
    
    # Performance settings
    cognitive_batch_size: int = 16
    cognitive_timeout: float = 30.0
    
    # Learning settings
    enable_pattern_learning: bool = True
    enable_memory_consolidation: bool = True


class OpenCogAphroditeEngine:
    """
    Enhanced Aphrodite Engine with OpenCog cognitive architecture integration.
    
    This class wraps the standard AphroditeEngine and adds cognitive capabilities
    including intelligent orchestration, pattern learning, and memory consolidation
    for improved inference performance at scale.
    """
    
    def __init__(self, 
                 base_engine: AphroditeEngine,
                 opencog_config: OpenCogAphroditeConfig):
        self.base_engine = base_engine
        self.opencog_config = opencog_config
        
        # Initialize cognitive engine if enabled
        self.cognitive_engine: Optional[CognitiveEngine] = None
        if opencog_config.enable_opencog:
            cognitive_config = opencog_config.cognitive_config or CognitiveConfig()
            self.cognitive_engine = CognitiveEngine(cognitive_config)
        
        # Cognitive request tracking
        self._cognitive_request_map: Dict[str, str] = {}
        self._request_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self._performance_stats = {
            'total_requests': 0,
            'cognitive_enhanced_requests': 0,
            'average_cognitive_overhead': 0.0,
            'cognitive_accuracy_improvement': 0.0
        }
        
        logger.info("OpenCog-enhanced Aphrodite engine initialized")
    
    async def start(self):
        """Start the enhanced engine."""
        if self.cognitive_engine:
            await self.cognitive_engine.start()
        logger.info("OpenCog-enhanced Aphrodite engine started")
    
    async def stop(self):
        """Stop the enhanced engine."""
        if self.cognitive_engine:
            await self.cognitive_engine.stop()
        logger.info("OpenCog-enhanced Aphrodite engine stopped")
    
    async def add_request_with_cognitive_enhancement(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Dict[str, str]] = None,
        priority: int = 0,
        enable_cognitive: bool = True
    ) -> str:
        """
        Add a request with cognitive enhancement.
        
        Args:
            request_id: Unique identifier for the request
            prompt: Input prompt
            params: Sampling parameters
            arrival_time: Optional arrival time
            lora_request: Optional LoRA request
            trace_headers: Optional trace headers
            priority: Request priority
            enable_cognitive: Whether to apply cognitive enhancement
            
        Returns:
            Enhanced request ID with cognitive annotations
        """
        cognitive_request_id = request_id
        
        if (self.opencog_config.enable_opencog and 
            self.cognitive_engine and enable_cognitive):
            
            # Apply cognitive preprocessing
            enhanced_context = await self._cognitive_preprocessing(
                prompt, params, priority
            )
            
            # Store cognitive context
            self._request_contexts[request_id] = enhanced_context
            
            # Map cognitive request
            cognitive_request_id = f"cognitive_{request_id}"
            self._cognitive_request_map[request_id] = cognitive_request_id
            
            # Adjust sampling parameters based on cognitive insights
            enhanced_params = await self._enhance_sampling_params(params, enhanced_context)
        else:
            enhanced_params = params
            enhanced_context = {}
        
        # Submit to base engine
        self.base_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=enhanced_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority
        )
        
        # Update statistics
        self._performance_stats['total_requests'] += 1
        if enable_cognitive:
            self._performance_stats['cognitive_enhanced_requests'] += 1
        
        return cognitive_request_id
    
    async def step_with_cognitive_orchestration(self) -> List[RequestOutput]:
        """
        Execute one step with cognitive orchestration.
        
        Returns:
            List of request outputs enhanced with cognitive insights
        """
        step_start_time = time.time()
        
        # Execute base engine step
        outputs = self.base_engine.step()
        
        # Apply cognitive postprocessing if enabled
        if (self.opencog_config.enable_opencog and 
            self.cognitive_engine and outputs):
            
            enhanced_outputs = []
            for output in outputs:
                enhanced_output = await self._cognitive_postprocessing(output)
                enhanced_outputs.append(enhanced_output)
            
            # Update performance metrics
            cognitive_overhead = time.time() - step_start_time
            self._update_performance_stats(cognitive_overhead)
            
            return enhanced_outputs
        
        return outputs
    
    async def _cognitive_preprocessing(
        self, 
        prompt: PromptType, 
        params: SamplingParams,
        priority: int
    ) -> Dict[str, Any]:
        """
        Apply cognitive preprocessing to enhance request understanding.
        
        Args:
            prompt: Input prompt
            params: Sampling parameters  
            priority: Request priority
            
        Returns:
            Enhanced context with cognitive insights
        """
        if not self.cognitive_engine:
            return {}
        
        try:
            # Convert prompt to cognitive representation
            prompt_str = self._extract_prompt_text(prompt)
            
            # Process through cognitive architecture
            cognitive_result = await self.cognitive_engine.process_inference_request(
                prompt_str, 
                {
                    'sampling_params': params,
                    'priority': priority,
                    'preprocessing_stage': True
                }
            )
            
            # Extract relevant insights
            enhanced_context = {
                'cognitive_confidence': cognitive_result.get('confidence', 0.5),
                'cognitive_patterns': cognitive_result.get('patterns', []),
                'attention_allocation': cognitive_result.get('attention', 0.5),
                'memory_associations': cognitive_result.get('memory_matches', []),
                'optimization_hints': cognitive_result.get('optimizations', [])
            }
            
            return enhanced_context
            
        except Exception as e:
            logger.warning(f"Cognitive preprocessing failed: {e}")
            return {}
    
    async def _cognitive_postprocessing(self, output: RequestOutput) -> RequestOutput:
        """
        Apply cognitive postprocessing to enhance output quality.
        
        Args:
            output: Base engine output
            
        Returns:
            Enhanced output with cognitive annotations
        """
        if not self.cognitive_engine or not hasattr(output, 'request_id'):
            return output
        
        try:
            request_id = output.request_id
            context = self._request_contexts.get(request_id, {})
            
            if not context:
                return output
            
            # Analyze output through cognitive lens
            output_text = self._extract_output_text(output)
            
            cognitive_analysis = await self.cognitive_engine.process_inference_request(
                output_text,
                {
                    **context,
                    'postprocessing_stage': True,
                    'original_output': output_text
                }
            )
            
            # Enhance output with cognitive insights
            enhanced_output = self._enhance_output_with_cognitive_data(
                output, cognitive_analysis, context
            )
            
            # Clean up context
            if request_id in self._request_contexts:
                del self._request_contexts[request_id]
            if request_id in self._cognitive_request_map:
                del self._cognitive_request_map[request_id]
            
            return enhanced_output
            
        except Exception as e:
            logger.warning(f"Cognitive postprocessing failed: {e}")
            return output
    
    async def _enhance_sampling_params(
        self, 
        params: SamplingParams, 
        context: Dict[str, Any]
    ) -> SamplingParams:
        """
        Enhance sampling parameters based on cognitive insights.
        
        Args:
            params: Original sampling parameters
            context: Cognitive context
            
        Returns:
            Enhanced sampling parameters
        """
        if not context:
            return params
        
        # Create enhanced parameters
        enhanced_params = SamplingParams(
            n=params.n,
            best_of=params.best_of,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            min_p=params.min_p,
            seed=params.seed,
            use_beam_search=params.use_beam_search,
            length_penalty=params.length_penalty,
            early_stopping=params.early_stopping,
            stop=params.stop,
            stop_token_ids=params.stop_token_ids,
            include_stop_str_in_output=params.include_stop_str_in_output,
            ignore_eos=params.ignore_eos,
            max_tokens=params.max_tokens,
            min_tokens=params.min_tokens,
            logprobs=params.logprobs,
            prompt_logprobs=params.prompt_logprobs,
            custom_token_bans=params.custom_token_bans,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
        )
        
        # Apply cognitive enhancements
        cognitive_confidence = context.get('cognitive_confidence', 0.5)
        
        # Adjust temperature based on confidence
        if cognitive_confidence > 0.8:
            enhanced_params.temperature = min(params.temperature * 0.9, params.temperature)
        elif cognitive_confidence < 0.3:
            enhanced_params.temperature = min(params.temperature * 1.1, 2.0)
        
        # Adjust top_p for better quality
        if context.get('optimization_hints'):
            enhanced_params.top_p = min(params.top_p * 1.05, 1.0)
        
        return enhanced_params
    
    def _extract_prompt_text(self, prompt: PromptType) -> str:
        """Extract text from prompt input."""
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            # Handle token lists or multi-part prompts
            if prompt and isinstance(prompt[0], str):
                return " ".join(prompt)
            elif prompt and isinstance(prompt[0], dict):
                # Handle structured prompts
                text_parts = []
                for part in prompt:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                return " ".join(text_parts)
        
        return str(prompt)  # Fallback
    
    def _extract_output_text(self, output: RequestOutput) -> str:
        """Extract text from output."""
        if hasattr(output, 'outputs') and output.outputs:
            # Get text from first output
            first_output = output.outputs[0]
            if hasattr(first_output, 'text'):
                return first_output.text
        
        return str(output)  # Fallback
    
    def _enhance_output_with_cognitive_data(
        self, 
        output: RequestOutput, 
        cognitive_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RequestOutput:
        """
        Enhance output with cognitive analysis data.
        
        This adds cognitive metadata to the output without modifying
        the core text generation results.
        """
        # For now, we'll add cognitive data as metadata
        # In a full implementation, this would be integrated into
        # the RequestOutput structure
        
        # Create enhanced output (simplified - would need proper RequestOutput handling)
        enhanced_output = output
        
        # Add cognitive metadata if the output supports it
        if hasattr(output, 'metadata') or hasattr(output, '__dict__'):
            cognitive_metadata = {
                'cognitive_confidence': cognitive_analysis.get('confidence', 0.5),
                'cognitive_patterns_matched': len(context.get('cognitive_patterns', [])),
                'attention_value': context.get('attention_allocation', 0.5),
                'memory_associations': len(context.get('memory_associations', [])),
                'cognitive_processing_applied': True
            }
            
            # Add metadata to output (implementation would vary based on RequestOutput structure)
            try:
                if hasattr(enhanced_output, '__dict__'):
                    enhanced_output.__dict__['cognitive_metadata'] = cognitive_metadata
            except Exception:
                # Fallback if metadata addition fails
                pass
        
        return enhanced_output
    
    def _update_performance_stats(self, cognitive_overhead: float):
        """Update performance statistics."""
        # Update running average of cognitive overhead
        current_avg = self._performance_stats['average_cognitive_overhead']
        enhanced_count = self._performance_stats['cognitive_enhanced_requests']
        
        if enhanced_count > 0:
            self._performance_stats['average_cognitive_overhead'] = (
                (current_avg * (enhanced_count - 1) + cognitive_overhead) / enhanced_count
            )
    
    def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get cognitive enhancement statistics."""
        stats = self._performance_stats.copy()
        
        if self.cognitive_engine:
            stats.update(self.cognitive_engine.get_statistics())
        
        return stats
    
    # Delegate other methods to base engine
    def __getattr__(self, name):
        """Delegate unknown methods to the base engine."""
        return getattr(self.base_engine, name)
    
    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class OpenCogAphroditeEngineBuilder:
    """Builder class for creating OpenCog-enhanced Aphrodite engines."""
    
    @staticmethod
    def create_enhanced_engine(
        base_engine: AphroditeEngine,
        cognitive_config: Optional[CognitiveConfig] = None,
        enable_all_features: bool = True
    ) -> OpenCogAphroditeEngine:
        """
        Create an OpenCog-enhanced Aphrodite engine.
        
        Args:
            base_engine: Base Aphrodite engine
            cognitive_config: Optional cognitive configuration
            enable_all_features: Whether to enable all cognitive features
            
        Returns:
            Enhanced engine with cognitive capabilities
        """
        opencog_config = OpenCogAphroditeConfig(
            enable_opencog=True,
            cognitive_config=cognitive_config,
            cognitive_preprocessing=enable_all_features,
            cognitive_postprocessing=enable_all_features,
            cognitive_orchestration=enable_all_features,
            enable_pattern_learning=enable_all_features,
            enable_memory_consolidation=enable_all_features
        )
        
        return OpenCogAphroditeEngine(base_engine, opencog_config)
    
    @staticmethod
    def create_lightweight_enhanced_engine(
        base_engine: AphroditeEngine
    ) -> OpenCogAphroditeEngine:
        """
        Create a lightweight OpenCog-enhanced engine with minimal overhead.
        
        Args:
            base_engine: Base Aphrodite engine
            
        Returns:
            Lightweight enhanced engine
        """
        cognitive_config = CognitiveConfig(
            atomspace_max_size=10000,  # Smaller atomspace
            reasoning_threads=2,  # Fewer reasoning threads
            memory_capacity=10000,  # Smaller memory capacity
            cognitive_cycles_per_second=10,  # Lower cycle rate
            enable_memory_consolidation=False,  # Disable for performance
        )
        
        opencog_config = OpenCogAphroditeConfig(
            enable_opencog=True,
            cognitive_config=cognitive_config,
            cognitive_preprocessing=True,
            cognitive_postprocessing=False,  # Disable for performance
            cognitive_orchestration=True,
            enable_pattern_learning=True,
            enable_memory_consolidation=False  # Disable for performance
        )
        
        return OpenCogAphroditeEngine(base_engine, opencog_config)