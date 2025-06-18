#!/usr/bin/env python
"""
Embedding Computation Script for LoCoBench

This script computes embeddings for tokenized documents based on
configuration parameters. It handles both standalone and late-chunking
embedding strategies.

Usage:
    python compute_embeddings.py --config PATH_TO_CONFIG_FILE
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_from_disk

# Add project root to path to ensure imports work
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from locobench.core.document_handler import DocumentHandler
from locobench.core.embedder import StandaloneEmbedder, LateChunkingEmbedder
from locobench.core.experiment_handler import create_concatenation_indices
from locobench.utils.embedding_io import (
    save_standalone_embeddings,
    save_latechunking_embeddings,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def save_output_config(config: Dict[str, Any], output_path: str):
    """
    Save output configuration to a JSON file.

    Args:
        config: Configuration dictionary to save
        output_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def create_run_name(config: Dict[str, Any]) -> str:
    """
    Create a descriptive name for the current embedding run.

    Args:
        config: Configuration dictionary

    Returns:
        A string name for the run directory
    """
    model_name = config["model_name"].replace("/", "_")
    dataset_name = os.path.basename(os.path.dirname(config["tokenized_dataset_path"]))

    # Handle both old and new index generation approaches
    if "indices_path" in config:
        # New approach using pre-generated indices
        concat_size = config.get("concat_size", "unknown")
        source_lang = config.get("source_lang", "unknown")
        target_lang = config.get("target_lang", "none")

        # Extract segment range info
        range_str = ""
        if "source_segment_range" in config:
            min_range, max_range = config["source_segment_range"]
            range_str = f"__range{min_range}-{max_range}"

        # Create run name for parallel indices approach
        if target_lang and target_lang != "none":
            return f"{model_name}__{dataset_name}__parallel__{source_lang}_{target_lang}__concat-size{concat_size}{range_str}"
        else:
            return f"{model_name}__{dataset_name}__parallel__{source_lang}__concat-size{concat_size}{range_str}"
    else:
        # Legacy approach
        concat_strategy = config.get("concatenation_strategy", "unknown")
        concat_size = config.get("concat_size", "unknown")

        # Extract ranges info
        ranges_str = ""
        if "position_specific_ranges" in config:
            ranges = config["position_specific_ranges"]
            ranges_parts = []
            for start, end in ranges:
                ranges_parts.append(f"{start}-{end}")
            ranges_str = f"ranges_{'_'.join(ranges_parts)}"

        return f"{model_name}__{dataset_name}__{concat_strategy}__concat-size{concat_size}_{ranges_str}"


def compute_embeddings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute embeddings according to the configuration.

    Args:
        config: Configuration dictionary with processing parameters

    Returns:
        Updated configuration with output paths
    """
    # Extract configuration parameters
    model_name = config["model_name"]
    tokenized_dataset_path = config["tokenized_dataset_path"]
    metadata_path = config.get(
        "metadata_path"
    )  # May not be needed when using pre-generated indices

    # Get experiment parameters (may not be present when using pre-generated indices)
    concat_params = {}
    if "concatenation_strategy" in config:
        concat_params = {
            "concatenation_strategy": config["concatenation_strategy"],
            "concat_size": config["concat_size"],
            "sample_size": config["sample_size"],
            "max_total_length": config.get("max_total_length"),
            "source_lang": config.get("source_lang", None),
            "target_lang": config.get("target_lang", None),
        }

        # Add position-specific ranges if present
        if "position_specific_ranges" in config:
            concat_params["position_specific_ranges"] = [
                tuple(range_pair) for range_pair in config["position_specific_ranges"]
            ]

    # Determine device
    device = config.get(
        "device",
        (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    print(f"Using device: {device}")

    # Load tokenized dataset
    print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
    tokenized_dataset = load_from_disk(tokenized_dataset_path)

    # Check if indices file is provided
    if "indices_path" in config:
        print(f"Loading indices from file: {config['indices_path']}")
        with open(config["indices_path"], "r") as f:
            indices_data = json.load(f)
        concat_indices = indices_data.get("concat_indices", [])
        standalone_indices = indices_data.get("standalone_indices", [])

        # Extract generation config and merge into main config
        generation_config = indices_data.get("generation_config", {})
        for key, value in generation_config.items():
            if key not in config:  # Don't override existing config values
                config[key] = value

        # Set default concatenation strategy if not specified
        if "concatenation_strategy" not in config:
            config["concatenation_strategy"] = (
                "permutations"  # Default for new parallel indices
            )

        print(
            f"Loaded {len(concat_indices)} concat indices and {len(standalone_indices)} standalone indices"
        )
        print(f"Merged generation config: {generation_config}")
    # Check if reference config is provided for indices (legacy support)
    elif "reference_config_path" in config:
        print(
            f"Loading indices from reference config: {config['reference_config_path']}"
        )
        reference_config = load_config(config["reference_config_path"])
        # Load indices from the reference config
        concat_indices = reference_config.get("concat_indices", [])
        standalone_indices = reference_config.get("standalone_indices", [])
        print(
            f"Loaded {len(concat_indices)} concat indices and {len(standalone_indices)} standalone indices"
        )
    else:
        # Generate concatenation indices as before (legacy approach)
        if not metadata_path or not concat_params:
            raise ValueError(
                "When not using 'indices_path', 'metadata_path' and concatenation parameters "
                "(concatenation_strategy, concat_size, sample_size) must be provided"
            )
        print("Generating new concatenation indices...")
        concat_indices, standalone_indices = create_concatenation_indices(
            metadata_path=metadata_path, **concat_params
        )

    # Initialize document handler with the same tokenizer
    document_handler = DocumentHandler(tokenizer_name=model_name)

    # Get separator for dataset preparation
    separator = config.get("separator", " ")

    # Case for wiki_parallel
    if "mode" in config and config["mode"] == "wiki_parallel":
        print("Preparing datasets for embedding in wiki_parallel mode...")
        source_lang = config.get("source_lang", None)
        target_lang = config.get("target_lang", None)
        if source_lang is None:
            raise ValueError(
                "For wiki_parallel mode, 'source_lang' and optionally 'target_lang' must be specified in the config."
            )
        if target_lang is None:
            print(
                f"Monolingual wiki_parallel mode detected for language:: source: {source_lang}"
            )
        else:
            print(
                f"Bilingual wiki_parallel mode detected for languages:: source: {source_lang} and target: {target_lang}"
            )
        datasets, concat_indices, standalone_indices = (
            document_handler.prepare_datasets_wiki_parallel(
                dataset_dict=tokenized_dataset,
                concat_indices=concat_indices,
                # standalone_indices=standalone_indices,
                separator=separator,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )

    # Case for regular embedding computation
    else:
        print("Preparing datasets for embedding...")
        datasets = document_handler.prepare_datasets(
            dataset=tokenized_dataset,
            concat_indices=concat_indices,
            standalone_indices=standalone_indices,
            separator=separator,
        )

    # Save the indices to configuration
    config["concat_indices"] = concat_indices
    config["standalone_indices"] = standalone_indices

    standalone_dataset = datasets["standalone"]
    concat_dataset = datasets["concatenated"]

    # Create dataloaders
    batch_size_standalone = config.get("batch_size_standalone", 2)
    batch_size_concat = config.get("batch_size_concat", 1)
    standalone_loader = document_handler.get_dataloader(
        standalone_dataset, batch_size=batch_size_standalone, shuffle=False
    )
    concat_loader = document_handler.get_dataloader(
        concat_dataset,
        batch_size=batch_size_concat,
        shuffle=False,
    )

    # Create embedders
    print("Initializing embedders...")
    standalone_embedder = StandaloneEmbedder(
        model_name=model_name,
        device=device,
    )

    latechunk_embedder = LateChunkingEmbedder(model_name=model_name, device=device)

    # Compute standalone embeddings
    print("Computing standalone embeddings...")
    standalone_embeddings = standalone_embedder.embed_dataloader(standalone_loader)

    # Compute late-chunking embeddings
    print("Computing late-chunking embeddings...")
    latechunk_embeddings = latechunk_embedder.embed_dataloader(concat_loader)

    # Create output directory structure
    embeddings_base_dir = config["embeddings_output_dir"]

    # Create a directory name based on parameters
    run_name = create_run_name(config)
    run_dir = os.path.join(embeddings_base_dir, run_name)

    # Create the run directory
    os.makedirs(run_dir, exist_ok=True)

    # Save embeddings
    print(f"Saving embeddings to {run_dir}...")

    # Save standalone embeddings
    mean_standalone_path = save_standalone_embeddings(
        embeddings=standalone_embeddings["mean"],
        output_dir=run_dir,
        pooling_strategy="mean",
    )

    cls_standalone_path = save_standalone_embeddings(
        embeddings=standalone_embeddings["cls"],
        output_dir=run_dir,
        pooling_strategy="cls",
    )

    # Save late-chunking embeddings
    segments_latechunk_path = save_latechunking_embeddings(
        embeddings=latechunk_embeddings["segment_embeddings"],
        output_dir=run_dir,
        embedding_type="segments",
    )

    cls_latechunk_path = save_latechunking_embeddings(
        embeddings=latechunk_embeddings["cls"],
        output_dir=run_dir,
        embedding_type="cls",
    )

    mean_latechunk_path = save_latechunking_embeddings(
        embeddings=latechunk_embeddings["mean"],
        output_dir=run_dir,
        embedding_type="mean",
    )

    # Update config with output paths
    config["embedding_paths"] = {
        "standalone_mean": mean_standalone_path,
        "standalone_cls": cls_standalone_path,
        "latechunking_segments": segments_latechunk_path,
        "latechunking_cls": cls_latechunk_path,
        "latechunking_mean": mean_latechunk_path,
    }
    config["run_dir"] = run_dir

    # Save output config
    output_config_path = os.path.join(run_dir, "embedding_config.json")
    save_output_config(config, output_config_path)

    print(f"Embedding computation complete. Output saved to {run_dir}")
    print(f"Configuration saved to {output_config_path}")

    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute embeddings for LoCoBench")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Compute embeddings
    compute_embeddings(config)


if __name__ == "__main__":
    main()
