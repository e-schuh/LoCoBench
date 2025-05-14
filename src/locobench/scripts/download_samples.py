#!/usr/bin/env python
"""
Dataset Sampling Script for LoCoBench

This script downloads a specified number of random samples from a Hugging Face dataset
using streaming mode. It allows sampling without downloading the entire dataset.

Usage:
    python download_samples.py --dataset_url DATASET_URL --num_samples NUM_SAMPLES --output_path OUTPUT_PATH
    python download_samples.py --config PATH_TO_CONFIG_FILE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path to ensure imports work
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from datasets import Dataset as RegularDataset


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


def download_dataset_samples(
    dataset_url: str,
    num_samples: int,
    output_path: str,
    split: str = "train",
    shuffle: bool = True,
    seed: Optional[int] = 42,
    buffer_size: Optional[int] = 500000,
    output_format: str = "jsonl",
) -> Dict[str, Any]:
    """
    Download random samples from a Hugging Face dataset using streaming.

    Args:
        dataset_url: URL or name of the dataset on Hugging Face
        num_samples: Number of samples to download
        output_path: Path to save the downloaded samples
        split: Dataset split to use (default: "train")
        shuffle: Whether to shuffle the dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        buffer_size: Size of shuffle buffer (default: 500000)
        output_format: Format to save samples in (default: "jsonl")

    Returns:
        Updated configuration with output information
    """
    print(f"Loading dataset {dataset_url} in streaming mode...")

    # Load dataset in streaming mode
    dataset = load_dataset(dataset_url, split=split, streaming=True)

    # Shuffle the dataset if requested
    if shuffle:
        print(f"Shuffling dataset with seed {seed} and buffer size {buffer_size}...")
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare to extract samples
    print(f"Generating {num_samples} samples from dataset...")
    samples = dataset.take(num_samples)

    print(f"Downloading samples and saving to {output_path}...")
    # Save the samples based on the specified format
    if output_format.lower() == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    elif output_format.lower() == "arrow":
        # Convert to regular dataset and save as Arrow format
        sample_list = list(samples)
        regular_dataset = RegularDataset.from_dict(
            {k: [s[k] for s in sample_list] for k in sample_list[0].keys()}
        )
        regular_dataset.save_to_disk(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Create result config
    result_config = {
        "dataset_url": dataset_url,
        "num_samples": num_samples,
        "output_path": output_path,
        "split": split,
        "shuffle": shuffle,
        "seed": seed,
        "output_format": output_format,
    }

    print(f"Successfully saved {num_samples} samples to {output_path}")

    return result_config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download samples from a Hugging Face dataset"
    )

    # Add arguments for either direct parameter passing or config file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to configuration file")
    group.add_argument(
        "--dataset_url", help="URL or name of the dataset on Hugging Face"
    )

    # Add optional arguments
    parser.add_argument("--num_samples", type=int, help="Number of samples to download")
    parser.add_argument("--output_path", help="Path to save the downloaded samples")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the dataset (default: True)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=500000,
        help="Size of shuffle buffer (default: 500000)",
    )
    parser.add_argument(
        "--output_format",
        default="jsonl",
        help="Format to save samples in (default: jsonl)",
    )

    args = parser.parse_args()

    # Process based on whether config file or direct parameters were provided
    if args.config:
        # Load configuration from file
        config = load_config(args.config)

        # Extract required parameters
        dataset_url = config["dataset_url"]
        num_samples = config["num_samples"]
        output_path = config["output_path"]

        # Extract optional parameters with defaults
        split = config.get("split", "train")
        shuffle = config.get("shuffle", True)
        seed = config.get("seed", 42)
        buffer_size = config.get("buffer_size", 500000)
        output_format = config.get("output_format", "jsonl")
    else:
        # Ensure required parameters are provided when not using config
        if not args.num_samples:
            parser.error("--num_samples is required when not using a config file")
        if not args.output_path:
            parser.error("--output_path is required when not using a config file")

        # Use command-line arguments
        dataset_url = args.dataset_url
        num_samples = args.num_samples
        output_path = args.output_path
        split = args.split
        shuffle = args.shuffle
        seed = args.seed
        buffer_size = args.buffer_size
        output_format = args.output_format

    # Download dataset samples
    result_config = download_dataset_samples(
        dataset_url=dataset_url,
        num_samples=num_samples,
        output_path=output_path,
        split=split,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
        output_format=output_format,
    )

    # Save output configuration - check if output_path is a directory
    output_path_obj = Path(output_path)
    if output_path_obj.is_dir():
        # If output_path is a directory, save the config file inside it
        config_filename = f"{output_path_obj.stem}_config.json"
        output_config_path = os.path.join(output_path, config_filename)
    else:
        # If output_path is a file, save the config alongside it
        output_config_path = os.path.join(
            os.path.dirname(output_path), f"{output_path_obj.stem}_config.json"
        )

    save_output_config(result_config, output_config_path)
    print(f"Output configuration saved to {output_config_path}")


if __name__ == "__main__":
    main()
