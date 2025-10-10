"""
Probabilistic and logical reasoning engines for OpenCog integration.

Provides advanced reasoning capabilities for large-scale inference through
probabilistic logic programming and formal reasoning systems.
"""

import asyncio
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)

try:
    from .atomspace import AtomSpaceManager, Atom, AtomType, TruthValue, Link, Node
except ImportError:
    from atomspace import AtomSpaceManager, Atom, AtomType, TruthValue, Link, Node


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    atom: Atom
    truth_value: TruthValue
    confidence: float
    reasoning_path: List[Atom]
    computation_time: float


class ProbabilisticReasoner:
    """
    Probabilistic reasoning engine using Probabilistic Logic Networks (PLN).
    
    Implements forward and backward chaining with uncertainty quantification
    for large-scale cognitive inference.
    """
    
    def __init__(self, atomspace: AtomSpaceManager, 
                 threshold: float = 0.7, max_steps: int = 1000):
        self.atomspace = atomspace
        self.threshold = threshold
        self.max_steps = max_steps
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._cache: Dict[str, ReasoningResult] = {}
        self._inference_rules = self._initialize_inference_rules()
    
    def _initialize_inference_rules(self) -> List[Callable]:
        """Initialize standard PLN inference rules."""
        return [
            self._deduction_rule,
            self._induction_rule,
            self._abduction_rule,
            self._similarity_rule,
            self._inheritance_rule,
            self._implication_rule,
        ]
    
    async def infer_async(self, query_atom: Atom) -> ReasoningResult:
        """Perform asynchronous probabilistic inference."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.infer, query_atom
        )
    
    def infer(self, query_atom: Atom) -> ReasoningResult:
        """
        Perform probabilistic inference on a query atom.
        
        Args:
            query_atom: The atom to perform inference on
            
        Returns:
            ReasoningResult with inferred truth value and reasoning path
        """
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query_atom.uuid}_{query_atom.truth_value.strength}_{query_atom.truth_value.confidence}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Perform forward chaining
        reasoning_path = []
        current_truth_value = query_atom.truth_value
        
        for step in range(self.max_steps):
            # Find applicable inference rules
            applicable_rules = self._find_applicable_rules(query_atom)
            
            if not applicable_rules:
                break
            
            # Apply the best rule
            best_rule, premises = self._select_best_rule(applicable_rules, query_atom)
            new_truth_value = best_rule(premises)
            
            # Update reasoning path
            reasoning_path.extend(premises)
            
            # Check convergence
            if self._has_converged(current_truth_value, new_truth_value):
                break
            
            current_truth_value = new_truth_value
        
        # Calculate final confidence
        final_confidence = self._calculate_inference_confidence(
            reasoning_path, current_truth_value
        )
        
        computation_time = time.time() - start_time
        
        result = ReasoningResult(
            atom=query_atom,
            truth_value=current_truth_value,
            confidence=final_confidence,
            reasoning_path=reasoning_path,
            computation_time=computation_time
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def _find_applicable_rules(self, atom: Atom) -> List[Tuple[Callable, List[Atom]]]:
        """Find inference rules applicable to the given atom."""
        applicable_rules = []
        
        # Get atoms that could be premises for inference
        incoming_atoms = list(self.atomspace.get_incoming_set(atom))
        related_atoms = self._find_related_atoms(atom)
        
        for rule in self._inference_rules:
            premises = self._find_rule_premises(rule, atom, incoming_atoms + related_atoms)
            if premises:
                applicable_rules.append((rule, premises))
        
        return applicable_rules
    
    def _find_related_atoms(self, atom: Atom, max_depth: int = 2) -> List[Atom]:
        """Find atoms related to the given atom through graph traversal."""
        related = []
        visited = {atom}
        queue = [(atom, 0)]
        
        while queue and len(related) < 100:  # Limit for performance
            current_atom, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Add incoming atoms
            for incoming in current_atom.incoming_set:
                if incoming not in visited:
                    related.append(incoming)
                    visited.add(incoming)
                    queue.append((incoming, depth + 1))
            
            # Add outgoing atoms if current atom is a link
            if isinstance(current_atom, Link):
                for outgoing in current_atom.outgoing:
                    if outgoing not in visited:
                        related.append(outgoing)
                        visited.add(outgoing)
                        queue.append((outgoing, depth + 1))
        
        return related
    
    def _find_rule_premises(self, rule: Callable, target: Atom, 
                           candidates: List[Atom]) -> Optional[List[Atom]]:
        """Find premises that make the rule applicable."""
        # This is a simplified premise finder - in practice would be more sophisticated
        rule_name = rule.__name__
        
        if rule_name == "_deduction_rule":
            return self._find_deduction_premises(target, candidates)
        elif rule_name == "_inheritance_rule":
            return self._find_inheritance_premises(target, candidates)
        elif rule_name == "_similarity_rule":
            return self._find_similarity_premises(target, candidates)
        
        return candidates[:2] if len(candidates) >= 2 else None
    
    def _find_deduction_premises(self, target: Atom, candidates: List[Atom]) -> Optional[List[Atom]]:
        """Find premises for deduction rule: (A->B, B->C) => A->C"""
        for i, atom1 in enumerate(candidates):
            for atom2 in candidates[i+1:]:
                if (isinstance(atom1, Link) and isinstance(atom2, Link) and
                    atom1.atom_type == AtomType.IMPLICATION and
                    atom2.atom_type == AtomType.IMPLICATION):
                    # Check if they can form a deduction chain
                    if (len(atom1.outgoing) >= 2 and len(atom2.outgoing) >= 2 and
                        atom1.outgoing[1] == atom2.outgoing[0]):
                        return [atom1, atom2]
        return None
    
    def _find_inheritance_premises(self, target: Atom, candidates: List[Atom]) -> Optional[List[Atom]]:
        """Find premises for inheritance rule."""
        inheritance_links = [a for a in candidates if 
                           isinstance(a, Link) and a.atom_type == AtomType.INHERITANCE]
        return inheritance_links[:2] if len(inheritance_links) >= 2 else None
    
    def _find_similarity_premises(self, target: Atom, candidates: List[Atom]) -> Optional[List[Atom]]:
        """Find premises for similarity rule."""
        similarity_links = [a for a in candidates if 
                          isinstance(a, Link) and a.atom_type == AtomType.SIMILARITY]
        return similarity_links[:2] if len(similarity_links) >= 2 else None
    
    def _select_best_rule(self, applicable_rules: List[Tuple[Callable, List[Atom]]], 
                         target: Atom) -> Tuple[Callable, List[Atom]]:
        """Select the best applicable rule based on confidence and relevance."""
        if not applicable_rules:
            return self._deduction_rule, []
        
        # Score rules based on premise strength and relevance
        best_score = -1
        best_rule = None
        best_premises = None
        
        for rule, premises in applicable_rules:
            score = self._calculate_rule_score(rule, premises, target)
            if score > best_score:
                best_score = score
                best_rule = rule
                best_premises = premises
        
        return best_rule, best_premises or []
    
    def _calculate_rule_score(self, rule: Callable, premises: List[Atom], 
                            target: Atom) -> float:
        """Calculate a score for how good a rule is for the target."""
        if not premises:
            return 0.0
        
        # Base score on premise truth values
        premise_strength = sum(p.truth_value.strength for p in premises) / len(premises)
        premise_confidence = sum(p.truth_value.confidence for p in premises) / len(premises)
        
        # Bonus for rule type relevance
        rule_bonus = {
            '_deduction_rule': 0.8,
            '_inheritance_rule': 0.7,
            '_similarity_rule': 0.6,
            '_induction_rule': 0.5,
            '_abduction_rule': 0.5,
            '_implication_rule': 0.7,
        }.get(rule.__name__, 0.5)
        
        return (premise_strength * 0.4 + premise_confidence * 0.4 + rule_bonus * 0.2)
    
    def _has_converged(self, old_tv: TruthValue, new_tv: TruthValue, 
                      epsilon: float = 0.01) -> bool:
        """Check if truth value has converged."""
        return (abs(old_tv.strength - new_tv.strength) < epsilon and
                abs(old_tv.confidence - new_tv.confidence) < epsilon)
    
    def _calculate_inference_confidence(self, reasoning_path: List[Atom], 
                                      final_tv: TruthValue) -> float:
        """Calculate confidence in the inference based on reasoning path."""
        if not reasoning_path:
            return final_tv.confidence
        
        # Confidence decreases with path length but increases with premise strength
        path_penalty = 1.0 / (1.0 + len(reasoning_path) * 0.1)
        premise_strength = sum(a.truth_value.confidence for a in reasoning_path) / len(reasoning_path)
        
        return min(1.0, final_tv.confidence * path_penalty * premise_strength)
    
    # PLN Inference Rules
    def _deduction_rule(self, premises: List[Atom]) -> TruthValue:
        """Deduction rule: (A->B, B->C) => A->C"""
        if len(premises) < 2:
            return TruthValue(0.5, 0.1)
        
        tv1, tv2 = premises[0].truth_value, premises[1].truth_value
        
        # PLN deduction formula
        strength = tv1.strength * tv2.strength
        confidence = tv1.confidence * tv2.confidence * tv1.strength
        
        return TruthValue(strength, confidence)
    
    def _induction_rule(self, premises: List[Atom]) -> TruthValue:
        """Induction rule for generalization."""
        if not premises:
            return TruthValue(0.5, 0.1)
        
        # Simple induction based on premise strength
        avg_strength = sum(p.truth_value.strength for p in premises) / len(premises)
        min_confidence = min(p.truth_value.confidence for p in premises) * 0.8
        
        return TruthValue(avg_strength, min_confidence)
    
    def _abduction_rule(self, premises: List[Atom]) -> TruthValue:
        """Abduction rule for hypothesis formation."""
        if not premises:
            return TruthValue(0.5, 0.1)
        
        # Abduction typically has lower confidence
        avg_strength = sum(p.truth_value.strength for p in premises) / len(premises)
        confidence = min(p.truth_value.confidence for p in premises) * 0.6
        
        return TruthValue(avg_strength, confidence)
    
    def _similarity_rule(self, premises: List[Atom]) -> TruthValue:
        """Similarity-based inference."""
        if len(premises) < 2:
            return TruthValue(0.5, 0.1)
        
        tv1, tv2 = premises[0].truth_value, premises[1].truth_value
        
        # Similarity strength is geometric mean
        strength = math.sqrt(tv1.strength * tv2.strength)
        confidence = min(tv1.confidence, tv2.confidence) * 0.9
        
        return TruthValue(strength, confidence)
    
    def _inheritance_rule(self, premises: List[Atom]) -> TruthValue:
        """Inheritance-based inference."""
        if not premises:
            return TruthValue(0.5, 0.1)
        
        # Inheritance propagates truth values with some attenuation
        max_strength = max(p.truth_value.strength for p in premises) * 0.9
        avg_confidence = sum(p.truth_value.confidence for p in premises) / len(premises)
        
        return TruthValue(max_strength, avg_confidence)
    
    def _implication_rule(self, premises: List[Atom]) -> TruthValue:
        """Implication-based inference."""
        if len(premises) < 2:
            return TruthValue(0.5, 0.1)
        
        # Modus ponens style inference
        antecedent_tv = premises[0].truth_value
        implication_tv = premises[1].truth_value
        
        strength = min(antecedent_tv.strength, implication_tv.strength)
        confidence = antecedent_tv.confidence * implication_tv.confidence
        
        return TruthValue(strength, confidence)


class LogicEngine:
    """
    Formal logic engine for deterministic reasoning.
    
    Implements classical and non-classical logical inference for
    symbolic reasoning tasks.
    """
    
    def __init__(self, atomspace: AtomSpaceManager):
        self.atomspace = atomspace
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._theorem_cache: Dict[str, bool] = {}
    
    async def reason_async(self, query_atom: Atom) -> Dict[str, Any]:
        """Perform asynchronous logical reasoning."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.reason, query_atom
        )
    
    def reason(self, query_atom: Atom) -> Dict[str, Any]:
        """
        Perform logical reasoning on a query atom.
        
        Args:
            query_atom: The atom to reason about
            
        Returns:
            Dictionary with logical reasoning results
        """
        # Find logical premises
        premises = self._find_logical_premises(query_atom)
        
        # Apply logical rules
        logical_conclusions = []
        for premise in premises:
            if isinstance(premise, Link) and premise.atom_type == AtomType.IMPLICATION:
                conclusion = self._apply_modus_ponens(premise, query_atom)
                if conclusion:
                    logical_conclusions.append(conclusion)
        
        # Check for contradictions
        contradictions = self._check_contradictions(logical_conclusions)
        
        # Calculate logical confidence
        confidence = self._calculate_logical_confidence(
            premises, logical_conclusions, contradictions
        )
        
        return {
            'premises': premises,
            'conclusions': logical_conclusions,
            'contradictions': contradictions,
            'confidence': confidence,
            'is_consistent': len(contradictions) == 0
        }
    
    def _find_logical_premises(self, atom: Atom) -> List[Atom]:
        """Find logical premises related to the atom."""
        premises = []
        
        # Look for implication links where atom appears
        for candidate in self.atomspace.get_atoms_by_type(AtomType.IMPLICATION):
            if isinstance(candidate, Link):
                if atom in candidate.outgoing:
                    premises.append(candidate)
        
        # Look for inheritance relationships
        for candidate in self.atomspace.get_atoms_by_type(AtomType.INHERITANCE):
            if isinstance(candidate, Link):
                if atom in candidate.outgoing:
                    premises.append(candidate)
        
        return premises
    
    def _apply_modus_ponens(self, implication: Link, query_atom: Atom) -> Optional[Atom]:
        """Apply modus ponens rule: (P -> Q, P) => Q"""
        if (len(implication.outgoing) >= 2 and 
            implication.outgoing[0] == query_atom):
            # If we have P and P->Q, conclude Q
            return implication.outgoing[1]
        return None
    
    def _check_contradictions(self, conclusions: List[Atom]) -> List[Tuple[Atom, Atom]]:
        """Check for logical contradictions in conclusions."""
        contradictions = []
        
        for i, atom1 in enumerate(conclusions):
            for atom2 in conclusions[i+1:]:
                if self._are_contradictory(atom1, atom2):
                    contradictions.append((atom1, atom2))
        
        return contradictions
    
    def _are_contradictory(self, atom1: Atom, atom2: Atom) -> bool:
        """Check if two atoms are contradictory."""
        # Simple contradiction check - could be made more sophisticated
        if atom1.name.startswith("NOT ") and atom1.name[4:] == atom2.name:
            return True
        if atom2.name.startswith("NOT ") and atom2.name[4:] == atom1.name:
            return True
        return False
    
    def _calculate_logical_confidence(self, premises: List[Atom], 
                                    conclusions: List[Atom],
                                    contradictions: List[Tuple[Atom, Atom]]) -> float:
        """Calculate confidence in logical reasoning."""
        if not premises:
            return 0.0
        
        # Base confidence on premise strength
        premise_confidence = sum(p.truth_value.confidence for p in premises) / len(premises)
        
        # Reduce confidence for contradictions
        contradiction_penalty = len(contradictions) * 0.2
        
        # Increase confidence for valid conclusions
        conclusion_bonus = min(0.3, len(conclusions) * 0.1)
        
        return max(0.0, premise_confidence + conclusion_bonus - contradiction_penalty)