"""
AtomSpace implementation for cognitive knowledge representation.

Provides the foundational data structure for OpenCog's knowledge representation,
enabling large-scale inference through hypergraph-based semantic networks.
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class AtomType(Enum):
    """Types of atoms in the atomspace."""
    NODE = "Node"
    LINK = "Link"
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode"
    INHERITANCE = "InheritanceLink"
    SIMILARITY = "SimilarityLink"
    EVALUATION = "EvaluationLink"
    IMPLICATION = "ImplicationLink"


@dataclass
class TruthValue:
    """Truth value representation with strength and confidence."""
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


class Atom:
    """Base class for all atoms in the atomspace."""
    
    def __init__(self, name: str, atom_type: AtomType, 
                 truth_value: Optional[TruthValue] = None):
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.atom_type = atom_type
        self.truth_value = truth_value or TruthValue(0.5, 0.5)
        self.incoming_set: Set['Atom'] = set()
        self.attention_value = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def __hash__(self):
        return hash(self.uuid)
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.uuid == other.uuid
    
    def __repr__(self):
        return f"{self.atom_type.value}({self.name}, tv={self.truth_value})"


class Node(Atom):
    """Node atoms represent concepts or entities."""
    
    def __init__(self, name: str, node_type: AtomType = AtomType.CONCEPT,
                 truth_value: Optional[TruthValue] = None):
        super().__init__(name, node_type, truth_value)


class Link(Atom):
    """Link atoms represent relationships between other atoms."""
    
    def __init__(self, name: str, outgoing: List[Atom],
                 link_type: AtomType = AtomType.LINK,
                 truth_value: Optional[TruthValue] = None):
        super().__init__(name, link_type, truth_value)
        self.outgoing = outgoing
        
        # Update incoming sets of target atoms
        for atom in outgoing:
            atom.incoming_set.add(self)
    
    def arity(self) -> int:
        """Return the number of outgoing atoms."""
        return len(self.outgoing)


class AtomSpaceManager:
    """Thread-safe manager for the atomspace hypergraph."""
    
    def __init__(self, max_size: int = 1000000):
        self._atoms: Dict[str, Atom] = {}
        self._name_index: Dict[str, Set[Atom]] = {}
        self._type_index: Dict[AtomType, Set[Atom]] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the atomspace."""
        with self._lock:
            if len(self._atoms) >= self.max_size:
                self._cleanup_low_attention_atoms()
            
            # Check if atom already exists
            existing = self.get_atom_by_name_and_type(atom.name, atom.atom_type)
            if existing:
                # Merge truth values
                existing.truth_value = self._merge_truth_values(
                    existing.truth_value, atom.truth_value)
                return existing
            
            self._atoms[atom.uuid] = atom
            
            # Update indices
            if atom.name not in self._name_index:
                self._name_index[atom.name] = set()
            self._name_index[atom.name].add(atom)
            
            if atom.atom_type not in self._type_index:
                self._type_index[atom.atom_type] = set()
            self._type_index[atom.atom_type].add(atom)
            
            return atom
    
    def get_atom_by_name_and_type(self, name: str, 
                                  atom_type: AtomType) -> Optional[Atom]:
        """Get atom by name and type."""
        with self._lock:
            if name in self._name_index:
                for atom in self._name_index[name]:
                    if atom.atom_type == atom_type:
                        return atom
            return None
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type."""
        with self._lock:
            return list(self._type_index.get(atom_type, set()))
    
    def create_node(self, name: str, node_type: AtomType = AtomType.CONCEPT,
                   truth_value: Optional[TruthValue] = None) -> Node:
        """Create and add a node to the atomspace."""
        node = Node(name, node_type, truth_value)
        return self.add_atom(node)
    
    def create_link(self, name: str, outgoing: List[Atom],
                   link_type: AtomType = AtomType.LINK,
                   truth_value: Optional[TruthValue] = None) -> Link:
        """Create and add a link to the atomspace."""
        link = Link(name, outgoing, link_type, truth_value)
        return self.add_atom(link)
    
    def get_incoming_set(self, atom: Atom) -> Set[Atom]:
        """Get all atoms that link to this atom."""
        return atom.incoming_set.copy()
    
    def _merge_truth_values(self, tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """Merge two truth values using confidence-weighted average."""
        total_conf = tv1.confidence + tv2.confidence
        if total_conf == 0:
            return TruthValue(0.5, 0.0)
        
        merged_strength = (tv1.strength * tv1.confidence + 
                          tv2.strength * tv2.confidence) / total_conf
        merged_confidence = min(1.0, total_conf)
        
        return TruthValue(merged_strength, merged_confidence)
    
    def _cleanup_low_attention_atoms(self):
        """Remove atoms with low attention values to free space."""
        with self._lock:
            if len(self._atoms) < self.max_size * 0.8:
                return
                
            # Sort atoms by attention value and remove bottom 10%
            sorted_atoms = sorted(self._atoms.values(), 
                                key=lambda a: a.attention_value)
            to_remove = sorted_atoms[:len(sorted_atoms) // 10]
            
            for atom in to_remove:
                self._remove_atom(atom)
    
    def _remove_atom(self, atom: Atom):
        """Remove an atom from the atomspace."""
        if atom.uuid in self._atoms:
            del self._atoms[atom.uuid]
            
            # Update indices
            if atom.name in self._name_index:
                self._name_index[atom.name].discard(atom)
                if not self._name_index[atom.name]:
                    del self._name_index[atom.name]
            
            if atom.atom_type in self._type_index:
                self._type_index[atom.atom_type].discard(atom)
    
    def size(self) -> int:
        """Return the number of atoms in the atomspace."""
        return len(self._atoms)
    
    def clear(self):
        """Clear all atoms from the atomspace."""
        with self._lock:
            self._atoms.clear()
            self._name_index.clear()
            self._type_index.clear()
    
    def shutdown(self):
        """Shutdown the atomspace manager."""
        self.executor.shutdown(wait=True)