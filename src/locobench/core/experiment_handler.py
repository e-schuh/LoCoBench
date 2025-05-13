"""
Experiment Handler Module

This module provides functionality for setting up and configuring local context
benchmark experiments. It handles the creation of index lists for document
concatenation strategies and other experiment setup tasks.
"""

from typing import List, Dict, Any, Tuple
import random
import json
from pathlib import Path


def create_concatenation_indices(
    metadata_path: str,
    concatenation_strategy: str = "random",
    concat_size: int = 2,
    sample_size: int = 100,
    max_total_length: int = None,
    individual_length_range: Tuple[int, int] = None,
    position_specific_ranges: List[Tuple[int, int]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Create a nested list of dataset indices for document concatenation experiments.

    This function creates lists of indices that specify which documents should be
    concatenated together in document combination experiments. The indices are
    based on the dataset_idx field in the provided metadata file.

    Args:
        metadata_path: Path to the metadata JSON file containing document information
        concatenation_strategy: Strategy for selecting documents to concatenate.
            Supports 'random' and 'switch' strategies.
        concat_size: Number of documents to concatenate for each new sample
        sample_size: Number of concatenated samples to create
        max_total_length: Maximum total token length for all concatenated documents combined.
            If None, no length constraint is applied for the total.
        individual_length_range: Optional tuple of (min_length, max_length) defining
            acceptable token length range for each individual document.
            If None, no constraints are applied to individual documents.
        position_specific_ranges: Optional list of (min_length, max_length) tuples defining
            acceptable token length ranges for each position in the concatenation sequence.
            The length of this list must match concat_size if provided.
            If provided, this overrides individual_length_range.

    Returns:
        A tuple containing:
        - A nested list where each inner list contains dataset indices that should
          be concatenated in the order they appear
        - A sorted list of all unique indices used in the concatenation experiment

    Raises:
        ValueError: If an unsupported concatenation strategy is specified,
                  if there are not enough documents for the requested concat_size,
                  or if position_specific_ranges length doesn't match concat_size.
    """
    if concatenation_strategy not in ["random", "switch"]:
        raise ValueError(
            f"Unsupported concatenation strategy: {concatenation_strategy}. "
            f"Supported strategies are: 'random', 'switch'"
        )

    if concat_size < 2:
        raise ValueError(
            f"Concatenation size must be at least 2, but got {concat_size}"
        )

    if sample_size < 1:
        raise ValueError(f"Sample size must be at least 1, but got {sample_size}")

    # Validate position_specific_ranges if provided
    if position_specific_ranges is not None:
        if len(position_specific_ranges) != concat_size:
            raise ValueError(
                f"Length of position_specific_ranges ({len(position_specific_ranges)}) "
                f"must match concat_size ({concat_size})"
            )

    # For the switch strategy, ensure sample_size is even
    if concatenation_strategy == "switch" and sample_size % 2 == 1:
        sample_size += 1
        print(
            f"Adjusted sample_size to {sample_size} to ensure even number for 'switch' strategy"
        )

    # Load metadata
    with open(Path(metadata_path), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Extract dataset indices from metadata
    dataset_indices = []
    doc_lengths = {}  # Store document lengths for filtering
    for doc_id, doc_info in metadata.items():
        idx = doc_info["dataset_idx"]
        dataset_indices.append(idx)
        doc_lengths[idx] = doc_info["token_length"]

    # Check if we have enough documents
    dataset_size = len(dataset_indices)
    if dataset_size < concat_size:
        raise ValueError(
            f"Dataset contains only {dataset_size} documents, but {concat_size} "
            f"are required for concatenation"
        )

    # Prepare position-specific document pools if needed
    document_pools = []

    if position_specific_ranges:
        # Create a pool of documents for each position based on length constraints
        for i, (min_length, max_length) in enumerate(position_specific_ranges):
            position_pool = [
                idx
                for idx in dataset_indices
                if min_length <= doc_lengths[idx] <= max_length
            ]

            if not position_pool:
                raise ValueError(
                    f"No documents found within the length range {(min_length, max_length)} "
                    f"for position {i}"
                )

            if len(position_pool) < sample_size and concatenation_strategy == "random":
                print(
                    f"Warning: Only {len(position_pool)} documents available for position {i} "
                    f"(range {(min_length, max_length)}). This may result in repeated documents."
                )

            document_pools.append(position_pool)
    else:
        # Filter all documents based on common individual length range if specified
        filtered_indices = dataset_indices
        if individual_length_range is not None:
            min_length, max_length = individual_length_range
            filtered_indices = [
                idx
                for idx in dataset_indices
                if min_length <= doc_lengths[idx] <= max_length
            ]

            if len(filtered_indices) < concat_size:
                raise ValueError(
                    f"After filtering by length range {individual_length_range}, "
                    f"only {len(filtered_indices)} documents remain, but {concat_size} "
                    f"are required for concatenation"
                )

        # Use the same pool for all positions when not using position-specific ranges
        for _ in range(concat_size):
            document_pools.append(filtered_indices)

    concatenation_indices = []
    used_indices = set()  # To track all unique indices used

    if concatenation_strategy == "random":
        # Create the specified number of concatenated samples
        samples_created = 0
        max_attempts = sample_size * 10  # Limit attempts to avoid infinite loops
        attempts = 0

        while samples_created < sample_size and attempts < max_attempts:
            attempts += 1

            # Select documents for each position
            selected_indices = []
            for pool in document_pools:
                # If pool is smaller than what we need, we might get duplicates
                # which is fine for random strategy with position constraints
                selected_idx = random.choice(pool)
                selected_indices.append(selected_idx)

            # Check total length constraint if specified
            if max_total_length is not None:
                total_length = sum(doc_lengths[idx] for idx in selected_indices)
                if total_length > max_total_length:
                    continue  # Skip this combination as it exceeds max length

            concatenation_indices.append(selected_indices)
            used_indices.update(selected_indices)
            samples_created += 1

        if samples_created < sample_size:
            print(
                f"Warning: Could only generate {samples_created} unique samples "
                f"(instead of requested {sample_size}) before reaching max attempts."
            )

    elif concatenation_strategy == "switch":
        # For the switch strategy, we create exactly half the number of base samples
        # and then create a pair for each by switching first and last
        base_sample_count = sample_size // 2

        # Use a set to track combinations we've already added
        added_combinations = set()

        samples_created = 0
        max_attempts = base_sample_count * 10  # Limit attempts to avoid infinite loops
        attempts = 0

        while samples_created < base_sample_count and attempts < max_attempts:
            attempts += 1

            # Select documents for each position
            selected_indices = []
            for pool in document_pools:
                selected_idx = random.choice(pool)
                selected_indices.append(selected_idx)

            # Check total length constraint if specified
            if max_total_length is not None:
                total_length = sum(doc_lengths[idx] for idx in selected_indices)
                if total_length > max_total_length:
                    continue  # Skip this combination as it exceeds max length

            # Convert to tuple for hashability to check if we've seen this combination
            selected_tuple = tuple(selected_indices)

            # Create the swapped version
            swapped_indices = selected_indices.copy()
            swapped_indices[0], swapped_indices[-1] = (
                swapped_indices[-1],
                swapped_indices[0],
            )
            swapped_tuple = tuple(swapped_indices)

            # Only add if neither the original nor swapped combination exists
            if (
                selected_tuple not in added_combinations
                and swapped_tuple not in added_combinations
            ):
                # Add the original selection
                concatenation_indices.append(selected_indices)
                added_combinations.add(selected_tuple)
                # Track used indices
                used_indices.update(selected_indices)

                # Add the swapped version
                concatenation_indices.append(swapped_indices)
                added_combinations.add(swapped_tuple)
                # No need to update used_indices again since swapped_indices
                # contains the same elements as selected_indices

                samples_created += 1

        if samples_created < base_sample_count:
            print(
                f"Warning: Could only generate {samples_created * 2} unique samples "
                f"(instead of requested {sample_size}) before reaching max attempts."
            )

    # Convert the set of used indices to a sorted list for consistent output
    used_indices_list = sorted(list(used_indices))

    return concatenation_indices, used_indices_list
