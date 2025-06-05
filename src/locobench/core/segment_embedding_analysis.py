"""
Utility functions for analyzing segment embeddings in LoCoBench.
This module provides tools for loading and processing embeddings for segment-level analysis.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Import necessary modules for loading embeddings
from locobench.utils.embedding_io import (
    load_standalone_embeddings,
    load_latechunking_embeddings,
)


def parse_latechunk_segments_key(key: str) -> Dict[str, Any]:
    """
    Parse the latechunk keys to extract segment information.

    Args:
        key: A key in the format "segmentID1__pos0__seq_segmentID1_segmentID2_segmentID3"
             Example: "3539177__pos0__seq_3539177_124217_233347"

    Returns:
        Dict containing segment_id, position, and doc_id
    """
    # Split the key by double underscores
    parts = key.split("__")

    # Extract segment ID
    segment_id = parts[0]

    # Extract position (strip 'pos' prefix)
    position = int(parts[1][3:])

    # Extract document ID (strip 'seq_' prefix)
    seq_parts = parts[2].split("_")
    doc_id = "_".join(seq_parts[1:])  # Use the full sequence as document ID

    return {
        "segment_id": segment_id,  # e.g., "3539177"
        "position": position,  # e.g., 0
        "doc_id": doc_id,  # e.g., "3539177_124217_233347"
    }


def load_and_process_embeddings(
    path: str | Path,
    pooling_strategy_segment_standalone: str = "cls",
    pooling_strategy_document: str = "cls",
) -> Dict[str, Any]:
    """
    Load and process embeddings for the analysis.

    Args:
        path: Path to the experiment results
        pooling_strategy_segment_standalone: Either "cls" or "mean"
        pooling_strategy_document: Either "cls" or "mean"

    Returns:
        Dict containing the following keys:
            - segID_to_emb: Dict mapping segment_id to standalone segment embeddings
            - docID_to_emb: Dict mapping doc_id to long document embeddings
            - docID_pos_to_segID: Dict mapping (doc_id, pos) to segment_id
            - docID_to_listof_segIDs: Dict mapping doc_id to list of segment IDs in order
            - num_segments_per_doc: Number of segments per document
            - docID_segID_to_pos: Dict mapping (doc_id, segment_id) to position
            - docID_pos_to_emb: Dict mapping (doc_id, pos) to latechunk segment embedding
    """
    # Load embeddings
    segID_to_emb = load_standalone_embeddings(
        directory=path, pooling_strategy=pooling_strategy_segment_standalone
    )
    latechunk_segments = load_latechunking_embeddings(
        directory=path, embedding_type="segments"
    )
    doc_embeddings = load_latechunking_embeddings(
        directory=path, embedding_type=pooling_strategy_document
    )

    # Convert to float if needed (bfloat16 tensors need to be converted for some operations)
    segID_to_emb = {
        k: v.float() if hasattr(v, "dtype") and v.dtype == torch.bfloat16 else v
        for k, v in segID_to_emb.items()
    }
    latechunk_segments = {
        k: v.float() if hasattr(v, "dtype") and v.dtype == torch.bfloat16 else v
        for k, v in latechunk_segments.items()
    }
    doc_embeddings = {
        k: v.float() if hasattr(v, "dtype") and v.dtype == torch.bfloat16 else v
        for k, v in doc_embeddings.items()
    }

    # Convert all keys to strings for consistency
    segID_to_emb = {str(k): v for k, v in segID_to_emb.items()}
    latechunk_segments = {str(k): v for k, v in latechunk_segments.items()}
    doc_embeddings = {str(k): v for k, v in doc_embeddings.items()}

    # Create mappings needed for our analysis
    docID_pos_to_segID = (
        {}
    )  # Maps (doc_id, pos) to segment_id {('3539177_124217_233347', 0): '3539177', ...}
    docID_segID_to_pos = (
        {}
    )  # Maps (doc_id, segment_id) to position {('3539177_124217_233347', '3539177'): 0, ...}
    docID_to_listof_segIDs = defaultdict(
        list
    )  # Maps doc_id to list of segment_ids in order {'3539177_124217_233347': ['3539177', '124217', '233347'], ...}
    docID_pos_to_emb = (
        {}
    )  # Maps (doc_id, pos) to latechunk segment embedding {('3539177_124217_233347', 0): <tensor>, ...}
    positions = set()  # Set of all positions

    # Process latechunk segments to extract mapping information
    for key in latechunk_segments.keys():
        info = parse_latechunk_segments_key(key)
        doc_id = info["doc_id"]
        position = info["position"]
        segment_id = info["segment_id"]

        # Map (document ID, position) to segment ID
        docID_pos_tuple = (doc_id, position)
        docID_pos_to_segID[docID_pos_tuple] = segment_id

        # Map (document ID, segment ID) to latechunk segment embedding
        docID_pos_to_emb[docID_pos_tuple] = latechunk_segments[key]

        # Build ordered list of segment IDs for each document
        # Ensure segment_ids are stored in correct position order by padding with None if needed
        while len(docID_to_listof_segIDs[doc_id]) <= position:
            docID_to_listof_segIDs[doc_id].append(None)
        docID_to_listof_segIDs[doc_id][position] = segment_id

        # Track all unique positions for calculating total segments later
        positions.add(position)

        # Map (document ID, segment ID) to position for reverse lookup
        docID_segID_tuple = (doc_id, segment_id)
        docID_segID_to_pos[docID_segID_tuple] = position

    # Determine how many segments are in each document (maximum position + 1)
    num_segments_per_doc = max(positions) + 1 if positions else 0

    # Strip "seq_" prefix from document IDs in doc_embeddings to match docID format in other dictionaries
    docID_to_emb = {}
    for doc_id, embedding in doc_embeddings.items():
        cleaned_doc_id = doc_id[4:] if doc_id.startswith("seq_") else doc_id
        docID_to_emb[cleaned_doc_id] = embedding

    # Return all processed data structures needed for analysis in a dictionary
    return {
        "segID_to_emb": segID_to_emb,
        "docID_to_emb": docID_to_emb,
        "docID_pos_to_segID": docID_pos_to_segID,
        "docID_to_listof_segIDs": docID_to_listof_segIDs,
        "num_segments_per_doc": num_segments_per_doc,
        "docID_segID_to_pos": docID_segID_to_pos,
        "docID_pos_to_emb": docID_pos_to_emb,
    }


def load_exp_info(path: str | Path) -> Dict[str, Any]:
    """
    Load experiment information either from embedding_config.json or by parsing the directory name.

    Args:
        path: Path to the experiment results directory

    Returns:
        Dict containing experiment information:
        - model_name: Name of the model used
        - concatenation_strategy: Strategy used for concatenation (permutations, switch, etc.)
        - concat_size: Number of segments concatenated
        - position_specific_ranges: List of position-specific token ranges
        - source_lang: Source language code (if available)
        - target_lang: Target language code (if available)
    """
    path_str = str(path)

    # First try to load from embedding_config.json
    config_path = os.path.join(path_str, "embedding_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract info from config
        return {
            "model_name": config.get("model_name", ""),
            "concatenation_strategy": config.get("concatenation_strategy", ""),
            "concat_size": config.get("concat_size", 0),
            "position_specific_ranges": config.get("position_specific_ranges", []),
            "source_lang": config.get("source_lang", None),
            "target_lang": config.get("target_lang", None),
        }

    # If no config exists, parse information from the directory name
    # Expected format: model-name__dataset__strategy__concat-sizeN_ranges_range1_range2...
    try:
        # Get the directory name without the full path
        dir_name = os.path.basename(path_str.rstrip("/"))

        # Split by double underscores
        parts = dir_name.split("__")

        # Extract model name (could contain underscores)
        model_name = parts[0].replace("_", "/")

        # Extract concatenation strategy (permutations, switch, etc.)
        concatenation_strategy = parts[2] if len(parts) > 2 else ""

        # Extract concat size and ranges
        if len(parts) > 3:
            concat_params = parts[3]

            # Extract concat size
            size_match = re.search(r"concat-size(\d+)", concat_params)
            concat_size = int(size_match.group(1)) if size_match else 0

            # Extract position specific ranges
            ranges = []
            ranges_match = re.findall(r"(\d+)-(\d+)", concat_params)

            if ranges_match:
                ranges = [[int(start), int(end)] for start, end in ranges_match]
        else:
            concat_size = 0
            ranges = []

        return {
            "model_name": model_name,
            "concatenation_strategy": concatenation_strategy,
            "concat_size": concat_size,
            "position_specific_ranges": ranges,
        }

    except Exception as e:
        print(f"Error parsing experiment info from path: {e}")
        return {
            "model_name": "",
            "concatenation_strategy": "",
            "concat_size": 0,
            "position_specific_ranges": [],
        }


class DocumentSegmentSimilarityAnalyzer:
    """
    Class to analyze the similarity between long document embeddings and individual segment embeddings.
    """

    def __init__(self):
        pass

    def calculate_doc_seg_similarity_metrics(
        self,
        embedding_data: Dict[str, Any],
        document_embedding_type: str = "document-level",
    ) -> Dict[str, Any]:
        """
        Analyze the similarity between long document embeddings and individual segment embeddings.

        Args:
            embedding_data: Dictionary containing embedding data with the following keys:
                - segID_to_emb: Dict mapping segment_id to standalone segment embeddings
                - docID_to_emb: Dict mapping doc_id to long document embeddings
                - docID_to_listof_segIDs: Dict mapping doc_id to list of segment IDs in order
                - num_segments_per_doc: Number of segments per document
                - docID_segID_to_pos: Dict mapping (doc_id, segment_id) to position

        Returns:
            Dict with position-based similarity statistics:
                - position_similarities: List of lists containing similarity scores for each position
                - position_means: Mean similarity score for each position
                - position_stds: Standard deviation of similarity scores for each position
                - position_counts: Number of similarity scores for each position
        """
        # Extract required data from the input dictionary
        segID_to_emb = embedding_data[
            "segID_to_emb"
        ]  # Extract standalone segment embeddings
        docID_to_emb = embedding_data[
            "docID_to_emb"
        ]  # Extract document-level embeddings
        docID_pos_to_emb = embedding_data[
            "docID_pos_to_emb"
        ]  # Extract latechunk segment embeddings
        docID_to_listof_segIDs = embedding_data["docID_to_listof_segIDs"]
        num_segments_per_doc = embedding_data["num_segments_per_doc"]
        docID_segID_to_pos = embedding_data["docID_segID_to_pos"]

        # For each position, collect similarities across all documents and permutations
        position_similarities = [[] for _ in range(num_segments_per_doc)]

        if document_embedding_type == "document-level":
            # Process each document
            for doc_id, segments in docID_to_listof_segIDs.items():
                # Skip if document ID not in long document embeddings
                if doc_id not in docID_to_emb:
                    continue

                # Skip if any segments are missing or not in standalone segment embeddings
                if None in segments or any(
                    seg_id not in segID_to_emb for seg_id in segments
                ):
                    continue

                # Get long document embedding
                doc_emb = docID_to_emb[doc_id]

                # Calculate similarity for each segment in the document
                for seg_id in segments:
                    standalone_emb = segID_to_emb[seg_id]
                    # Calculate cosine similarity between document embedding and segment embedding
                    similarity = F.cosine_similarity(doc_emb, standalone_emb, dim=0)
                    # Look up the position of this segment in the document
                    position = docID_segID_to_pos[(doc_id, seg_id)]
                    # Store the similarity score for this position
                    position_similarities[position].append(similarity.item())

        elif document_embedding_type == "latechunk-segment":
            # Process each document
            for doc_id, segments in docID_to_listof_segIDs.items():

                # Skip if any segments are missing or not in standalone segment embeddings
                if None in segments or any(
                    seg_id not in segID_to_emb for seg_id in segments
                ):
                    continue

                # Get latechunk segment embedding for each position
                for seg_id in segments:
                    position = docID_segID_to_pos[(doc_id, seg_id)]
                    tuple_key = (doc_id, position)

                    # Skip if document ID not in latechunk segment embeddings
                    if tuple_key not in docID_pos_to_emb:
                        print("Skipping tuple key:", tuple_key)
                        continue
                    standalone_emb = segID_to_emb[seg_id]
                    latechunk_emb = docID_pos_to_emb[tuple_key]
                    # Calculate cosine similarity between latechunk segment embedding and standalone segment embedding
                    similarity = F.cosine_similarity(
                        latechunk_emb, standalone_emb, dim=0
                    )
                    # Store the similarity score for this position
                    position_similarities[position].append(similarity.item())
        else:
            raise ValueError(
                "document_embedding_type must be either 'document-level' or 'latechunk-segment'"
            )

        # Compute statistics for each position
        # Calculate means, standard deviations, and confidence intervals
        position_means = []
        position_stds = []
        position_ci_lower = []
        position_ci_upper = []
        position_counts = []

        for sims in position_similarities:
            if sims:
                mean = np.mean(sims)
                std = np.std(sims)
                count = len(sims)

                # Calculate 95% confidence interval
                # If we have enough samples, use t-distribution
                if count > 1:
                    ci = stats.t.interval(
                        0.95, count - 1, loc=mean, scale=std / np.sqrt(count)
                    )
                    ci_lower = ci[0]
                    ci_upper = ci[1]
                else:
                    # Not enough samples for CI, just use mean
                    ci_lower = mean
                    ci_upper = mean
            else:
                mean = float("nan")
                std = float("nan")
                count = 0
                ci_lower = float("nan")
                ci_upper = float("nan")

            position_means.append(mean)
            position_stds.append(std)
            position_counts.append(count)
            position_ci_lower.append(ci_lower)
            position_ci_upper.append(ci_upper)

        results = {
            "position_similarities": position_similarities,
            "position_means": position_means,
            "position_stds": position_stds,
            "position_counts": position_counts,
            "position_ci_lower": position_ci_lower,
            "position_ci_upper": position_ci_upper,
        }

        return results

    def run_position_analysis(
        self,
        path: str | Path,
        document_embedding_type: str = "document-level",
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
    ) -> Dict[str, Any]:
        """
        Run complete position-based similarity analysis for a given path.

        Args:
            path: Path to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling

        Returns:
            Dict with position-based similarity results including:
            - position_similarities: List of lists of similarity scores
            - position_means: Mean similarity per position
            - position_stds: Standard deviation of similarity per position
            - position_counts: Number of samples per position
            - path: Original path parameter
            - num_segments: Number of segments per document
            - pooling_strategy_segment_standalone: Pooling strategy used for standalone segment embeddings
            - pooling_strategy_document: Pooling strategy used for document embeddings
            - model_name: Name of the model used in the experiment
            - concatenation_strategy: Strategy used for concatenation
            - concat_size: Number of segments concatenated
            - position_specific_ranges: List of position-specific token ranges
        """
        # Load experiment information
        exp_info = load_exp_info(path)

        # Load and process embeddings
        # print(f"Processing {path}...")
        embedding_data = load_and_process_embeddings(
            path, pooling_strategy_segment_standalone, pooling_strategy_document
        )

        # Run similarity analysis with the embedding data
        results = self.calculate_doc_seg_similarity_metrics(
            embedding_data, document_embedding_type=document_embedding_type
        )

        # Add path and metadata information
        results["path"] = path
        results["num_segments"] = embedding_data["num_segments_per_doc"]
        results["pooling_strategy_segment_standalone"] = (
            pooling_strategy_segment_standalone
        )
        results["pooling_strategy_document"] = pooling_strategy_document

        # Add experiment metadata from config or path parsing
        results["model_name"] = exp_info["model_name"]
        results["concatenation_strategy"] = exp_info["concatenation_strategy"]
        results["concat_size"] = exp_info["concat_size"]
        results["position_specific_ranges"] = exp_info["position_specific_ranges"]

        # Add language information if available
        if "source_lang" in exp_info and exp_info["source_lang"]:
            results["source_lang"] = exp_info["source_lang"]
        if "target_lang" in exp_info and exp_info["target_lang"]:
            results["target_lang"] = exp_info["target_lang"]

        return results


class DirectionalLeakageAnalyzer:
    """
    Class to analyze directional similarity between segments in the same document.
    """

    def __init__(self):
        pass

    def directional_leakage(
        self,
        docID_pos_to_emb: Dict[Tuple[str, int], torch.Tensor],
        docID_pos_to_segID: Dict[Tuple[str, int], str],
        segID_to_emb: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Calculate directional leakage between segments in the same document.

        Args:
            docID_pos_to_emb: Dict mapping (doc_id, position) to contextualized embeddings
            docID_pos_to_segID: Dict mapping (doc_id, position) to segment_id
            segID_to_emb: Dict mapping segment_id to standalone segment embeddings

        Returns:
            Dict containing forward and backward influence measures:
                - forward_influence: List of cosine similarities from earlier to later positions
                - backward_influence: List of cosine similarities from later to earlier positions
                - forward_mean: Mean of forward influence scores
                - backward_mean: Mean of backward influence scores
        """
        # Group by document
        doc_positions = defaultdict(dict)
        for (doc_id, pos), emb in docID_pos_to_emb.items():
            segment_id = docID_pos_to_segID.get((doc_id, pos))
            if segment_id and segment_id in segID_to_emb:
                doc_positions[doc_id][pos] = {
                    "latechunk": emb,
                    "segment_id": segment_id,
                    "standalone": segID_to_emb[segment_id],
                }

        # For each document with multiple positions, calculate directional influence
        forward_influence = []  # Earlier to later
        backward_influence = []  # Later to earlier

        for doc_id, pos_data in doc_positions.items():
            if len(pos_data) < 2:
                continue

            positions = sorted(pos_data.keys())

            # For each pair of positions
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i + 1 :]:  # Only look at later positions
                    # Influence of early concept on contextualized representation of later position
                    forward_leakage = F.cosine_similarity(
                        pos_data[pos1]["standalone"], pos_data[pos2]["latechunk"], dim=0
                    )

                    # Influence of later concept on contextualized representation of earlier position
                    backward_leakage = F.cosine_similarity(
                        pos_data[pos2]["standalone"], pos_data[pos1]["latechunk"], dim=0
                    )

                    forward_influence.append(forward_leakage.item())
                    backward_influence.append(backward_leakage.item())

        return {
            "forward_influence": forward_influence,
            "backward_influence": backward_influence,
            "forward_mean": np.mean(forward_influence) if forward_influence else 0,
            "backward_mean": np.mean(backward_influence) if backward_influence else 0,
        }

    def run_directional_leakage_analysis(
        self, path: str | Path, pooling_strategy_segment_standalone: str = "mean"
    ) -> Dict[str, Any]:
        """
        Run directional leakage analysis for a single experiment.

        Args:
            path: Path to experiment results

        Returns:
            Dict containing directional leakage results and experiment metadata
        """
        # Load experiment info
        exp_info = load_exp_info(path)

        # Load and process embeddings using existing function
        embedding_data = load_and_process_embeddings(
            path=path,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document="cls",  # irrelevant / not used for leakage analysis
        )

        # Calculate directional leakage
        leakage_results = self.directional_leakage(
            docID_pos_to_emb=embedding_data["docID_pos_to_emb"],
            docID_pos_to_segID=embedding_data["docID_pos_to_segID"],
            segID_to_emb=embedding_data["segID_to_emb"],
        )

        # Perform t-test to determine if forward influence is significantly greater than backward
        t_stat, p_value = stats.ttest_ind(
            leakage_results["forward_influence"],
            leakage_results["backward_influence"],
            alternative="greater",  # Test if forward > backward
        )

        # Add t-test results
        leakage_results["t_stat"] = t_stat
        leakage_results["p_value"] = p_value
        leakage_results["significant"] = p_value < 0.05

        # Add experiment metadata from exp_info
        leakage_results["model_name"] = exp_info.get("model_name", "Unknown model name")
        leakage_results["concat_size"] = exp_info.get("concat_size", 0)
        leakage_results["position_specific_ranges"] = exp_info.get(
            "position_specific_ranges", []
        )

        # Add language information if available
        if "source_lang" in exp_info and exp_info["source_lang"]:
            leakage_results["source_lang"] = exp_info["source_lang"]
        if "target_lang" in exp_info and exp_info["target_lang"]:
            leakage_results["target_lang"] = exp_info["target_lang"]

        # Extract path name for display
        path_str = str(path)
        if "results/runs/" in path_str:
            path_name = path_str.split("results/runs/")[1]
            leakage_results["path_name"] = path_name
        else:
            leakage_results["path_name"] = path_str

        return leakage_results
