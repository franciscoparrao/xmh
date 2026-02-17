"""
CheckpointManager: Manages saving and loading algorithm states for replay.

Enables counterfactual analysis by allowing algorithm execution to be
resumed from any saved checkpoint with modified parameters or operators.
"""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Checkpoint:
    """A saved algorithm state at a specific point in execution."""
    generation: int
    state: Dict[str, Any]
    metadata: Dict[str, Any]

    def save(self, filepath: str):
        """Save checkpoint to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "Checkpoint":
        """Load checkpoint from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class CheckpointManager:
    """
    Manages checkpoints for an algorithm run.

    Provides functionality for:
    - Creating checkpoints at regular intervals or critical points
    - Loading checkpoints for replay/counterfactual analysis
    - Managing checkpoint storage
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        max_checkpoints: int = 20,
        auto_save: bool = False
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints (None for memory-only)
            max_checkpoints: Maximum checkpoints to keep in memory
            auto_save: Whether to automatically save to disk
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.max_checkpoints = max_checkpoints
        self.auto_save = auto_save

        self.checkpoints: Dict[int, Checkpoint] = {}
        self._checkpoint_order: List[int] = []

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        generation: int,
        algorithm_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Create and store a checkpoint.

        Args:
            generation: Current generation number
            algorithm_state: Complete algorithm state
            metadata: Additional metadata to store

        Returns:
            Created checkpoint
        """
        checkpoint = Checkpoint(
            generation=generation,
            state=self._deep_copy_state(algorithm_state),
            metadata=metadata or {}
        )

        # Store in memory
        self.checkpoints[generation] = checkpoint
        self._checkpoint_order.append(generation)

        # Enforce max checkpoints
        while len(self._checkpoint_order) > self.max_checkpoints:
            oldest = self._checkpoint_order.pop(0)
            del self.checkpoints[oldest]

        # Auto-save if enabled
        if self.auto_save and self.save_dir:
            filepath = self.save_dir / f"checkpoint_gen{generation}.pkl"
            checkpoint.save(str(filepath))

        return checkpoint

    def get_checkpoint(self, generation: int) -> Optional[Checkpoint]:
        """Get checkpoint at exact generation."""
        return self.checkpoints.get(generation)

    def get_nearest_checkpoint(
        self,
        generation: int,
        before: bool = True
    ) -> Optional[Checkpoint]:
        """
        Get checkpoint nearest to specified generation.

        Args:
            generation: Target generation
            before: If True, get checkpoint before or at generation
                   If False, get checkpoint after or at generation

        Returns:
            Nearest checkpoint or None
        """
        if not self.checkpoints:
            return None

        available = sorted(self.checkpoints.keys())

        if before:
            candidates = [g for g in available if g <= generation]
            if candidates:
                return self.checkpoints[max(candidates)]
        else:
            candidates = [g for g in available if g >= generation]
            if candidates:
                return self.checkpoints[min(candidates)]

        return None

    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint generations."""
        return sorted(self.checkpoints.keys())

    def clear(self):
        """Clear all checkpoints from memory."""
        self.checkpoints.clear()
        self._checkpoint_order.clear()

    def save_all(self, prefix: str = "checkpoint"):
        """Save all checkpoints to disk."""
        if not self.save_dir:
            raise ValueError("No save directory specified")

        for gen, checkpoint in self.checkpoints.items():
            filepath = self.save_dir / f"{prefix}_gen{gen}.pkl"
            checkpoint.save(str(filepath))

    def load_all(self, prefix: str = "checkpoint"):
        """Load all checkpoints from disk."""
        if not self.save_dir:
            raise ValueError("No save directory specified")

        for filepath in self.save_dir.glob(f"{prefix}_gen*.pkl"):
            checkpoint = Checkpoint.load(str(filepath))
            self.checkpoints[checkpoint.generation] = checkpoint
            if checkpoint.generation not in self._checkpoint_order:
                self._checkpoint_order.append(checkpoint.generation)

        self._checkpoint_order.sort()

    @staticmethod
    def _deep_copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of algorithm state."""
        copied = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, dict):
                copied[key] = CheckpointManager._deep_copy_state(value)
            elif isinstance(value, list):
                copied[key] = [
                    v.copy() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                copied[key] = value
        return copied
