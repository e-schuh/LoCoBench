"""
Embedding Analysis Module

This module provides utilities for analyzing and processing document embeddings.
It includes functions for aggregating late-chunking embeddings by document and position.
"""

import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import torch.nn.functional as F
import os
import json
from datetime import datetime
import numpy as np
from scipy import stats


def parse_latechunk_key(key: str) -> Tuple[str, str, str]:
    """
    Parse a late-chunking embedding key into its components.

    Args:
        key: The composite key in format 'doc_id__posN__seq_doc1_doc2_doc3'

    Returns:
        Tuple of (doc_id, position, sequence_id)
    """
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid late-chunking key format: {key}")

    doc_id = parts[0]
    position = parts[1][3:]  # Extract N from 'posN' as string
    sequence_id = parts[2]

    return doc_id, position, sequence_id


def group_latechunk_embeddings_by_doc_pos(
    latechunk_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Group late-chunking embeddings by document and position.

    Args:
        latechunk_embeddings: Dictionary with keys in format 'doc_id__posN__seq_doc1_doc2_doc3'
                             and values as embedding tensors

    Returns:
        Nested dictionary with first level keys as document IDs, second level keys
        as position indices (as strings), and values as lists of embedding tensors
    """
    # Initialize collections for grouping embeddings
    grouped_embeddings = defaultdict(lambda: defaultdict(list))

    # Parse each key and group embeddings by doc_id and position
    for key, embedding in latechunk_embeddings.items():
        doc_id, position, _ = parse_latechunk_key(key)
        grouped_embeddings[doc_id][position].append(embedding)

    return grouped_embeddings


def group_latechunk_embeddings_by_seqId(
    latechunk_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Group late-chunking embeddings by sequence ID.

    Args:
        latechunk_embeddings: Dictionary with keys in format 'doc_id__posN__seq_doc1_doc2_doc3'
                             and values as embedding tensors
    Returns:
        Nested dictionary with first level keys as sequence IDs, second level keys
        as document IDs, and values as lists of embedding tensors
    """
    # Initialize collections for grouping embeddings
    grouped_embeddings = defaultdict(lambda: defaultdict(list))

    # Parse each key and group embeddings by sequence ID
    for key, embedding in latechunk_embeddings.items():
        doc_id, _, sequence_id = parse_latechunk_key(key)
        grouped_embeddings[sequence_id][doc_id].append(embedding)

    return grouped_embeddings


def compute_mean_embeddings(
    grouped_embeddings: Dict[str, Dict[str, List[torch.Tensor]]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute mean embeddings for each document-position pair from grouped embeddings.

    Args:
        grouped_embeddings: Nested dictionary with first level keys as document IDs,
                           second level keys as position indices, and values as lists
                           of embedding tensors

    Returns:
        Nested dictionary with first level keys as document IDs, second level keys
        as position indices, and values as mean embedding tensors
    """
    result = {}
    for doc_id, positions in grouped_embeddings.items():
        result[doc_id] = {}
        for position, embeddings in positions.items():
            # Stack embeddings and compute mean along first dimension
            stacked_embeddings = torch.stack(embeddings)
            mean_embedding = torch.mean(stacked_embeddings, dim=0)
            result[doc_id][position] = mean_embedding

    return result


def compute_std_embeddings(
    grouped_embeddings: Dict[str, Dict[str, List[torch.Tensor]]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute standard deviation of embeddings for each document-position pair.

    Args:
        grouped_embeddings: Nested dictionary with first level keys as document IDs,
                           second level keys as position indices, and values as lists
                           of embedding tensors

    Returns:
        Nested dictionary with first level keys as document IDs, second level keys
        as position indices, and values as standard deviation tensors
    """
    result = {}
    for doc_id, positions in grouped_embeddings.items():
        result[doc_id] = {}
        for position, embeddings in positions.items():
            # Only compute std if we have more than one sample
            if len(embeddings) > 1:
                # Stack embeddings and compute std along first dimension
                stacked_embeddings = torch.stack(embeddings)
                std_embedding = torch.std(stacked_embeddings, dim=0)
                result[doc_id][position] = std_embedding
            else:
                # If only one sample, std is zero
                result[doc_id][position] = torch.zeros_like(embeddings[0])

    return result


def compute_cosine_similarities(
    grouped_embeddings: Dict[str, Dict[str, List[torch.Tensor]]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute pairwise cosine similarities between all embeddings for each document-position pair.

    Args:
        grouped_embeddings: Nested dictionary with first level keys as document IDs,
                           second level keys as position indices, and values as lists
                           of embedding tensors

    Returns:
        Nested dictionary with first level keys as document IDs, second level keys
        as position indices, and values as similarity matrices where:
        - If only one embedding is present: returns a 1x1 tensor with value 1.0
        - If multiple embeddings: returns an NxN matrix where N is the number of embeddings,
          and each element [i,j] is the cosine similarity between embeddings i and j
    """
    result = {}

    for doc_id, positions in grouped_embeddings.items():
        result[doc_id] = {}

        for position, embeddings in positions.items():
            # If there's only one embedding, return a 1x1 tensor with 1.0 (perfect similarity)
            if len(embeddings) == 1:
                result[doc_id][position] = torch.tensor([[1.0]])
                continue

            # Stack embeddings into a tensor of shape [N, embedding_dim]
            stacked_embeddings = torch.stack(embeddings)

            # Normalize embeddings to unit length for cosine similarity calculation
            normalized_embeddings = F.normalize(stacked_embeddings, p=2, dim=1)

            # Compute pairwise cosine similarity: cos_sim = (A · B) / (||A|| * ||B||)
            # With normalization, this simplifies to the dot product
            # Result is a matrix of shape [N, N]
            similarity_matrix = torch.matmul(
                normalized_embeddings, normalized_embeddings.transpose(0, 1)
            )

            result[doc_id][position] = similarity_matrix

    return result


def eval_switch_exp(
    standalone_embeddings: Dict[str, torch.Tensor],
    latechunk_embeddings: Dict[str, torch.Tensor],
    concat_size: int = None,
    output_dir: Optional[str] = None,
    filename: str = "sim_eval_switch_exp",
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Evaluate the effect of switching documents' positions in sequences by comparing
    standalone embeddings to their late chunking counterparts.

    This function computes cosine similarities between standalone embeddings and
    late chunking embeddings for documents in first and last positions of sequences
    and their switched counterparts.

    Args:
        standalone_embeddings: Dictionary with document IDs as keys and their standalone embeddings as values
        latechunk_embeddings: Dictionary with keys in format 'doc_id__posN__seq_doc1_doc2_doc3'
                             and values as embedding tensors from late chunking approach
        concat_size: Size of the concatenated document sequence (if None, will be inferred from sequence ID)
        output_dir: Directory path to save the results (if None, results will not be saved)
        filename: Base filename to use for saving results (default: "sim_eval_switch_exp")

    Returns:
        A dictionary of results where keys are sequence IDs and values are dictionaries containing
        cosine similarities for first and last documents in original and switched sequences
    """
    # Initialize results dictionary
    results = {}

    # First, identify all unique sequences without the "pos" part
    sequences = set()
    for key in latechunk_embeddings.keys():
        _, _, seq_id = parse_latechunk_key(key)
        sequences.add(seq_id)

    # For each sequence, find its switched counterpart
    processed_seqs = set()

    for seq_id in sequences:
        if seq_id in processed_seqs:
            continue

        # Extract document IDs from the sequence ID
        # Format is "seq_doc1_doc2_doc3..."
        parts = seq_id.split("_")[1:]  # Skip "seq" prefix

        # Infer concat_size from the number of parts if not provided
        sequence_concat_size = len(parts) if concat_size is None else concat_size

        # Create the switched sequence ID (swap first and last document)
        switched_parts = parts.copy()
        switched_parts[0], switched_parts[-1] = switched_parts[-1], switched_parts[0]
        switched_seq_id = "seq_" + "_".join(switched_parts)

        if switched_seq_id not in sequences:
            raise ValueError(
                f"Switched sequence ID {switched_seq_id} not found for sequence ID {seq_id}"
            )

        # Mark both sequences as processed
        processed_seqs.add(seq_id)
        processed_seqs.add(switched_seq_id)

        # Get the first and last document IDs
        first_doc_id = parts[0]
        last_doc_id = parts[-1]

        # Create a result entry for this pair of sequences - using just seq_id as key
        result_key = seq_id
        results[result_key] = {}

        # Get the relevant embeddings
        try:
            # Original sequence - first document at position 0
            first_doc_orig_key = f"{first_doc_id}__pos0__{seq_id}"
            first_doc_orig_embedding = latechunk_embeddings.get(first_doc_orig_key)

            # Original sequence - last document at position (sequence_concat_size-1)
            last_doc_orig_key = f"{last_doc_id}__pos{sequence_concat_size-1}__{seq_id}"
            last_doc_orig_embedding = latechunk_embeddings.get(last_doc_orig_key)

            # Switched sequence - first document (which was last in original) at position 0
            first_doc_switch_key = f"{last_doc_id}__pos0__{switched_seq_id}"
            first_doc_switch_embedding = latechunk_embeddings.get(first_doc_switch_key)

            # Switched sequence - last document (which was first in original) at position (sequence_concat_size-1)
            last_doc_switch_key = (
                f"{first_doc_id}__pos{sequence_concat_size-1}__{switched_seq_id}"
            )
            last_doc_switch_embedding = latechunk_embeddings.get(last_doc_switch_key)

            # Get standalone embeddings for first and last documents
            first_doc_standalone = standalone_embeddings.get(first_doc_id)
            last_doc_standalone = standalone_embeddings.get(last_doc_id)

            # Check if we have all required embeddings
            if None in [
                first_doc_orig_embedding,
                last_doc_orig_embedding,
                first_doc_switch_embedding,
                last_doc_switch_embedding,
                first_doc_standalone,
                last_doc_standalone,
            ]:

                raise ValueError(
                    f"Missing embeddings for sequence pair {seq_id} and {switched_seq_id}"
                )

            # Compute cosine similarities
            # Original sequence first document vs standalone
            first_orig_sim = F.cosine_similarity(
                first_doc_orig_embedding.unsqueeze(0), first_doc_standalone.unsqueeze(0)
            ).item()

            # Original sequence last document vs standalone
            last_orig_sim = F.cosine_similarity(
                last_doc_orig_embedding.unsqueeze(0), last_doc_standalone.unsqueeze(0)
            ).item()

            # Switched sequence first document (original last) vs standalone
            first_switch_sim = F.cosine_similarity(
                first_doc_switch_embedding.unsqueeze(0),
                last_doc_standalone.unsqueeze(0),
            ).item()

            # Switched sequence last document (original first) vs standalone
            last_switch_sim = F.cosine_similarity(
                last_doc_switch_embedding.unsqueeze(0),
                first_doc_standalone.unsqueeze(0),
            ).item()

            # Store results
            results[result_key] = {
                f"{first_doc_id}_orig_pos0": first_orig_sim,
                f"{last_doc_id}_orig_pos{sequence_concat_size-1}": last_orig_sim,
                f"{last_doc_id}_switch_pos0": first_switch_sim,
                f"{first_doc_id}_switch_pos{sequence_concat_size-1}": last_switch_sim,
                "diff_orig": first_orig_sim - last_orig_sim,
                "diff_switch": first_switch_sim - last_switch_sim,
            }

        except Exception as e:

            print(
                f"Error processing sequence pair {seq_id} and {switched_seq_id}: {str(e)}"
            )
            continue

    # Save results to a file if output_dir is provided
    filepath = None
    if output_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add timestamp to filename for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.json"
        filepath = os.path.join(output_dir, full_filename)

        # Convert results to JSON-serializable format (handling floating point values)
        json_results = {}
        for seq_id, metrics in results.items():
            json_results[seq_id] = {k: float(v) for k, v in metrics.items()}

        # Write results to JSON file
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {filepath}")

    return results, filepath


def analyze_switch_exp_results(
    results_or_filepath: Union[Dict[str, Dict[str, float]], str],
    output_dir: Optional[str] = None,
    filename: str = "switch_exp_analysis",
) -> Dict[str, Dict[str, float]]:
    """
    Analyze the results from a switch experiment by collecting and computing statistics
    on the difference values.

    Args:
        results_or_filepath: Either a dictionary containing the results returned by eval_switch_exp
                            or a filepath to a JSON file containing these results
        output_dir: Directory path to save the analysis results (if None, results will not be saved)
        filename: Base filename to use for saving analysis results (default: "switch_exp_analysis")

    Returns:
        A dictionary containing the analysis results (statistics about the differences)
    """
    # Load results if a filepath was provided
    if isinstance(results_or_filepath, str):
        if not os.path.exists(results_or_filepath):
            raise FileNotFoundError(f"Results file not found: {results_or_filepath}")

        with open(results_or_filepath, "r") as f:
            results = json.load(f)
    else:
        results = results_or_filepath

    # Collect all diff_orig and diff_switch values
    diff_orig_values = []
    diff_switch_values = []

    for seq_id, metrics in results.items():
        diff_orig_values.append(metrics["diff_orig"])
        diff_switch_values.append(metrics["diff_switch"])

    # Combine all differences for overall analysis
    all_diffs = diff_orig_values + diff_switch_values

    # Calculate statistics
    analysis_results = {
        "orig": {
            "num_samples": len(diff_orig_values),
            "mean": float(np.mean(diff_orig_values)),
            "std": float(np.std(diff_orig_values, ddof=1)),  # Sample standard deviation
            "min": float(np.min(diff_orig_values)),
            "25th_percentile": float(np.percentile(diff_orig_values, 25)),
            "median": float(np.median(diff_orig_values)),
            "75th_percentile": float(np.percentile(diff_orig_values, 75)),
            "max": float(np.max(diff_orig_values)),
            # Perform one-sample t-test to check if values are significantly greater than zero
            "t_stat": float(
                stats.ttest_1samp(diff_orig_values, 0, alternative="greater")[0]
            ),
            "p_value": float(
                stats.ttest_1samp(diff_orig_values, 0, alternative="greater")[1]
            ),
        },
        "switch": {
            "num_samples": len(diff_switch_values),
            "mean": float(np.mean(diff_switch_values)),
            "std": float(np.std(diff_switch_values, ddof=1)),
            "min": float(np.min(diff_switch_values)),
            "25th_percentile": float(np.percentile(diff_switch_values, 25)),
            "median": float(np.median(diff_switch_values)),
            "75th_percentile": float(np.percentile(diff_switch_values, 75)),
            "max": float(np.max(diff_switch_values)),
            "t_stat": float(
                stats.ttest_1samp(diff_switch_values, 0, alternative="greater")[0]
            ),
            "p_value": float(
                stats.ttest_1samp(diff_switch_values, 0, alternative="greater")[1]
            ),
        },
        "all": {
            "num_samples": len(all_diffs),
            "mean": float(np.mean(all_diffs)),
            "std": float(np.std(all_diffs, ddof=1)),
            "min": float(np.min(all_diffs)),
            "25th_percentile": float(np.percentile(all_diffs, 25)),
            "median": float(np.median(all_diffs)),
            "75th_percentile": float(np.percentile(all_diffs, 75)),
            "max": float(np.max(all_diffs)),
            "t_stat": float(stats.ttest_1samp(all_diffs, 0, alternative="greater")[0]),
            "p_value": float(stats.ttest_1samp(all_diffs, 0, alternative="greater")[1]),
        },
    }

    # Save results to a file if output_dir is provided
    if output_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add timestamp to filename for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.json"
        filepath = os.path.join(output_dir, full_filename)

        # Write results to JSON file
        with open(filepath, "w") as f:
            json.dump(analysis_results, f, indent=2)

        print(f"\nAnalysis results saved to {filepath}")

    return analysis_results
