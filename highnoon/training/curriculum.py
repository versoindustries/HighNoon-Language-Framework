# src/training/curriculum.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Curriculum Learning for Molecular Property Prediction - Phase 3 Enhancement
============================================================================

Implements curriculum learning strategies that progressively increase
task difficulty during training:

1. Size-based curriculum: Start with small molecules, gradually add larger ones
2. Difficulty-based curriculum: Start with "easy" molecules (low prediction error)
3. Diversity-based curriculum: Gradually increase chemical diversity

This approach helps the model learn more effectively by providing a structured
learning progression, similar to how humans learn from simple to complex concepts.

References:
    - Bengio et al. (2009) - "Curriculum Learning"
    - Soviany et al. (2022) - "Curriculum Learning: A Survey"
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Curriculum learning scheduler for molecular datasets.

    Manages the progressive introduction of training examples based on
    complexity metrics (e.g., number of atoms, molecular weight, diversity).
    """

    def __init__(
        self,
        strategy: str = "size_based",
        num_stages: int = 5,
        warmup_epochs: int = 10,
        stage_increment_epochs: int = 20,
        initial_fraction: float = 0.2,
        verbose: bool = True,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            strategy: Curriculum strategy ('size_based', 'difficulty_based', 'mixed')
            num_stages: Number of curriculum stages
            warmup_epochs: Epochs to train on initial subset before expanding
            stage_increment_epochs: Epochs between curriculum expansions
            initial_fraction: Fraction of data to start with (easiest examples)
            verbose: Print curriculum progression logs
        """
        self.strategy = strategy
        self.num_stages = num_stages
        self.warmup_epochs = warmup_epochs
        self.stage_increment_epochs = stage_increment_epochs
        self.initial_fraction = initial_fraction
        self.verbose = verbose

        # Internal state
        self.current_stage = 0
        self.current_epoch = 0
        self.dataset_sorted_indices = None
        self.complexity_scores = None

    def compute_complexity_scores(
        self, molecules: list[dict], strategy: str | None = None
    ) -> np.ndarray:
        """
        Compute complexity scores for molecules based on curriculum strategy.

        Args:
            molecules: List of molecule dictionaries with properties
            strategy: Override default strategy

        Returns:
            complexity_scores: Array of complexity scores (higher = more complex)
        """
        strategy = strategy or self.strategy
        num_molecules = len(molecules)
        scores = np.zeros(num_molecules)

        if strategy == "size_based":
            # Complexity = number of atoms
            for i, mol in enumerate(molecules):
                scores[i] = mol.get("num_atoms", len(mol.get("positions", [])))

        elif strategy == "weight_based":
            # Complexity = molecular weight
            for i, mol in enumerate(molecules):
                scores[i] = mol.get("molecular_weight", mol.get("num_atoms", 0) * 12.0)

        elif strategy == "difficulty_based":
            # Complexity = prediction error (requires pre-trained model or estimated difficulty)
            # For now, use a proxy: number of heavy atoms or specific functional groups
            for i, mol in enumerate(molecules):
                # Simple proxy: molecules with more heavy atoms are "harder"
                num_heavy_atoms = mol.get("num_heavy_atoms", mol.get("num_atoms", 0))
                num_bonds = mol.get("num_bonds", 0)
                scores[i] = num_heavy_atoms + 0.1 * num_bonds

        elif strategy == "mixed":
            # Combine size and estimated difficulty
            for i, mol in enumerate(molecules):
                size_score = mol.get("num_atoms", 0)
                weight_score = mol.get("molecular_weight", 0) / 100.0  # Normalize
                scores[i] = 0.6 * size_score + 0.4 * weight_score

        else:
            raise ValueError(f"Unknown curriculum strategy: {strategy}")

        return scores

    def prepare_curriculum(self, molecules: list[dict]) -> np.ndarray:
        """
        Prepare curriculum by sorting molecules by complexity.

        Args:
            molecules: List of molecule dictionaries

        Returns:
            sorted_indices: Indices of molecules sorted by complexity (easy to hard)
        """
        # Compute complexity scores
        self.complexity_scores = self.compute_complexity_scores(molecules)

        # Sort indices by complexity (ascending: easy first)
        sorted_indices = np.argsort(self.complexity_scores)

        self.dataset_sorted_indices = sorted_indices

        if self.verbose:
            log.info(f"[CURRICULUM] Prepared curriculum for {len(molecules)} molecules")
            log.info(
                f"[CURRICULUM] Complexity range: {self.complexity_scores.min():.2f} to {self.complexity_scores.max():.2f}"
            )
            log.info(f"[CURRICULUM] Strategy: {self.strategy}, Stages: {self.num_stages}")

        return sorted_indices

    def get_current_subset_indices(self, epoch: int) -> np.ndarray:
        """
        Get indices of molecules to include in training at current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            subset_indices: Indices of molecules to include
        """
        if self.dataset_sorted_indices is None:
            raise ValueError("Curriculum not prepared. Call prepare_curriculum() first.")

        self.current_epoch = epoch

        # Determine current stage based on epoch
        if epoch < self.warmup_epochs:
            # Warmup: use initial fraction
            self.current_stage = 0
            fraction = self.initial_fraction
        else:
            # Incremental expansion
            epochs_after_warmup = epoch - self.warmup_epochs
            stage = min(self.num_stages - 1, epochs_after_warmup // self.stage_increment_epochs)
            self.current_stage = stage + 1

            # Linear interpolation between initial_fraction and 1.0
            fraction = self.initial_fraction + (1.0 - self.initial_fraction) * (
                (stage + 1) / self.num_stages
            )

        # Select subset of molecules (easiest fraction)
        num_total = len(self.dataset_sorted_indices)
        num_to_include = int(num_total * fraction)
        num_to_include = max(1, num_to_include)  # At least 1 molecule

        subset_indices = self.dataset_sorted_indices[:num_to_include]

        if self.verbose and epoch % 10 == 0:
            log.info(
                f"[CURRICULUM] Epoch {epoch}: Stage {self.current_stage}/{self.num_stages}, "
                f"Using {len(subset_indices)}/{num_total} molecules ({fraction * 100:.1f}%)"
            )

        return subset_indices

    def get_stage_info(self) -> dict:
        """
        Get information about current curriculum stage.

        Returns:
            stage_info: Dictionary with stage metadata
        """
        return {
            "current_stage": self.current_stage,
            "num_stages": self.num_stages,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "strategy": self.strategy,
        }


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum scheduler that adjusts pacing based on model performance.

    If the model is learning well (low loss), progress faster through curriculum.
    If the model is struggling (high loss or unstable), slow down expansion.
    """

    def __init__(
        self,
        loss_threshold_for_progression: float = 0.1,
        min_epochs_per_stage: int = 10,
        max_epochs_per_stage: int = 50,
        **kwargs,
    ):
        """
        Initialize adaptive curriculum scheduler.

        Args:
            loss_threshold_for_progression: Loss improvement threshold to progress
            min_epochs_per_stage: Minimum epochs before progressing
            max_epochs_per_stage: Maximum epochs to stay in one stage
            **kwargs: Additional arguments for base CurriculumScheduler
        """
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold_for_progression
        self.min_epochs_per_stage = min_epochs_per_stage
        self.max_epochs_per_stage = max_epochs_per_stage

        # Track performance
        self.stage_losses = []
        self.epochs_in_current_stage = 0
        self.best_loss = float("inf")

    def update_with_loss(self, epoch: int, current_loss: float) -> bool:
        """
        Update curriculum based on current training loss.

        Args:
            epoch: Current epoch
            current_loss: Current training or validation loss

        Returns:
            progressed: True if curriculum advanced to next stage
        """
        self.current_epoch = epoch
        self.epochs_in_current_stage += 1

        # Record loss
        self.stage_losses.append(current_loss)

        # Check if we should progress to next stage
        progressed = False

        if self.epochs_in_current_stage >= self.min_epochs_per_stage:
            # Check loss improvement
            recent_loss = np.mean(self.stage_losses[-5:])  # Average last 5 epochs
            loss_improvement = self.best_loss - recent_loss

            # Progress if loss improved sufficiently or max epochs reached
            if (
                loss_improvement > self.loss_threshold
                or self.epochs_in_current_stage >= self.max_epochs_per_stage
            ):
                if self.current_stage < self.num_stages - 1:
                    self.current_stage += 1
                    self.epochs_in_current_stage = 0
                    self.stage_losses = []
                    self.best_loss = recent_loss
                    progressed = True

                    if self.verbose:
                        log.info(
                            f"[ADAPTIVE CURRICULUM] Progressed to stage {self.current_stage}/{self.num_stages} "
                            f"(loss improvement: {loss_improvement:.4f})"
                        )

        # Update best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss

        return progressed


def create_curriculum_batches(
    molecules: list[dict],
    labels: np.ndarray,
    curriculum_indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create batches from curriculum-selected subset of molecules.

    Args:
        molecules: Full list of molecules
        labels: Full array of labels
        curriculum_indices: Indices selected by curriculum scheduler
        batch_size: Batch size
        shuffle: Shuffle selected subset before batching

    Returns:
        batches: List of (molecule_batch, label_batch) tuples
    """
    # Select subset
    subset_molecules = [molecules[i] for i in curriculum_indices]
    subset_labels = labels[curriculum_indices]

    # Shuffle if requested
    if shuffle:
        perm = np.random.permutation(len(subset_molecules))
        subset_molecules = [subset_molecules[i] for i in perm]
        subset_labels = subset_labels[perm]

    # Create batches
    batches = []
    num_batches = (len(subset_molecules) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(subset_molecules))

        batch_molecules = subset_molecules[start_idx:end_idx]
        batch_labels = subset_labels[start_idx:end_idx]

        batches.append((batch_molecules, batch_labels))

    return batches
