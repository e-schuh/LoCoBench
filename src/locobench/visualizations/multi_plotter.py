"""
Merged analysis functionality for position similarities and directional leakage.
This module provides a unified approach to plotting multiple analysis results.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from pathlib import Path

# Import the required analyses functions
from ..core.segment_embedding_analysis import (
    DocumentSegmentSimilarityAnalyzer,
    DirectionalLeakageAnalyzer,
    PositionalDirectionalLeakageAnalyzer,
)

from .single_plotters import (
    DirectionalLeakageSinglePlotter,
    PositionSimilaritySinglePlotter,
    PositionalDirectionalLeakageSinglePlotter,
)


class DirectionalLeakageMultiPlotter:
    """
    Class to handle the plotting of directional leakage analysis results in a multi plot.

    NOTE: The averages computed by this plotter may differ slightly from PositionalDirectionalLeakageMultiPlotter
    due to different averaging methodologies:
    - This plotter: Simple mean over all pairwise similarities
    - PositionalDirectionalLeakageMultiPlotter: Averages per-position means (equal weight per position)

    """

    def __init__(self):
        self.analysis_type = "directional_leakage"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot directional leakage for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        directional_leakage_single_plotter = DirectionalLeakageSinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Directional Information Flow",
            pooling_legend_type="segment_standalone",
            subplotter=directional_leakage_single_plotter.plot_directional_leakage_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class DocumentLevel2SegmentStandaloneSimPlotter:
    """
    Class to handle the plotting of position analysis results in a multi plot.
    """

    def __init__(self):
        self.analysis_type = "position"
        self.document_embedding_type = "document-level"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """

        position_similarity_single_plotter = PositionSimilaritySinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            document_embedding_type=self.document_embedding_type,
            title="Similarity between Document-Level Embedding and Standalone Segment Embeddings",
            pooling_legend_type="segment_standalone_and_document",
            subplotter=position_similarity_single_plotter.plot_position_similarities_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class SegmentLatechunk2SegmentStandaloneSimPlotter:
    """
    Class to handle the plotting of position analysis results in a multi plot.
    """

    def __init__(self):
        self.analysis_type = "position"
        self.document_embedding_type = "latechunk-segment"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """

        position_similarity_single_plotter = PositionSimilaritySinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            document_embedding_type=self.document_embedding_type,
            title="Similarity between Contextualized Segment Embeddings and Standalone Segment Embeddings",
            pooling_legend_type="segment_standalone",
            subplotter=position_similarity_single_plotter.plot_position_similarities_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class PositionalDirectionalLeakageMultiPlotter:
    """
    Class to handle the plotting of positional directional leakage analysis results in a multi plot.

    NOTE: The averages computed by this plotter may differ slightly from DirectionalLeakageMultiPlotter
    due to different averaging methodologies:
    - This plotter: Averages per-position means (equal weight per position)
    - DirectionalLeakageMultiPlotter: Simple mean over all pairwise similarities

    """

    def __init__(self):
        self.analysis_type = "positional_directional_leakage"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "mean",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot positional directional leakage for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        positional_directional_leakage_single_plotter = (
            PositionalDirectionalLeakageSinglePlotter()
        )

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Position-wise Directional Information Flow",
            pooling_legend_type="segment_standalone",
            subplotter=positional_directional_leakage_single_plotter.plot_positional_directional_leakage_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


def analyze_and_plot_multiple_results(
    paths: List[str | Path],
    analysis_type: str = "position",  # Either "position" or "directional_leakage"
    document_embedding_type: str = "document-level",
    title: str = "Similarity between Document-Level Embedding and Standalone Segment Embeddings",
    pooling_legend_type: str = "segment_standalone_and_document",  # Either "segment_standalone" or "segment_standalone_and_document"
    subplotter: Callable = None,
    pooling_strategy_segment_standalone: str = "cls",
    pooling_strategy_document: str = "cls",
    matryoshka_dimensions: Optional[List[int]] = None,
    show_segment_lengths: bool = False,
    figure_width: int = 15,
    subplot_height: int = 4,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    return_full_results: bool = False,
    single_model_mode: Optional[bool] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Analyze and plot multiple experiment results in a grid, organized by model name and concat size.
    Supports both position similarity and directional leakage analysis.

    Args:
        paths: List of paths to experiment results
        analysis_type: Type of analysis to perform ("position" or "directional_leakage")
        document_embedding_type: Type of document embedding to use
        title: Title for the plot
        pooling_legend_type: Type of pooling legend to show
        subplotter: Function to use for plotting individual subplots
        pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
        pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
                                      (only used for position analysis)
        matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
        show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
        figure_width: Width of the complete figure in inches
        subplot_height: Height of each subplot in inches
        save_plot: Whether to save the plot
        save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
        show_plot: Whether to display the plot
        return_full_results: Whether to return the full analysis results
        single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

    Returns:
        If return_full_results is True, returns a dictionary with model names as keys and
        lists of result dictionaries as values
    """
    if not subplotter:
        raise ValueError("A subplotter function must be provided.")
    # Setup analysis functions based on type
    if analysis_type == "position":
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        run_analysis = lambda path: doc_seg_analyzer.run_position_analysis(
            path=path,
            document_embedding_type=document_embedding_type,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=matryoshka_dimensions,
        )
        plot_in_subplot = subplotter
    elif analysis_type == "directional_leakage":
        directional_leakage_analyzer = DirectionalLeakageAnalyzer()
        run_analysis = (
            lambda path: directional_leakage_analyzer.run_directional_leakage_analysis(
                path, pooling_strategy_segment_standalone
            )
        )
        plot_in_subplot = subplotter
    elif analysis_type == "positional_directional_leakage":
        positional_directional_leakage_analyzer = PositionalDirectionalLeakageAnalyzer()
        run_analysis = lambda path: positional_directional_leakage_analyzer.run_positional_directional_leakage_analysis(
            path, pooling_strategy_segment_standalone, matryoshka_dimensions
        )
        plot_in_subplot = subplotter
    else:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type}. Use 'position' or 'directional_leakage'."
        )

    # Run analysis for each path
    all_results = []
    for path in paths:
        # print(f"Processing {path}...")
        results = run_analysis(path)
        all_results.append(results)

    # Group results by model name
    model_groups = defaultdict(list)
    for result in all_results:
        model_name = result["model_name"]
        # Abbreviate model names
        if "Alibaba-NLP/gte-multilingual-base" in model_name:
            model_name = "mGTE"
        elif "jinaai/jina-embeddings-v3" in model_name:
            model_name = "jina-v3"
        # Store result with abbreviated model name
        result["abbreviated_model_name"] = model_name
        model_groups[model_name].append(result)

    # Sort results within each group by concat size
    for model_name, results in model_groups.items():
        results.sort(key=lambda x: x["concat_size"])

    # Dictionary to store ranges information for the legend
    ranges_info = {}
    range_id = 1

    # Assign range IDs to each unique range configuration
    for results in model_groups.values():
        for result in results:
            ranges_tuple = tuple(map(tuple, result["position_specific_ranges"]))
            if ranges_tuple and ranges_tuple not in ranges_info:
                ranges_info[ranges_tuple] = f"SL{range_id}"
                result["range_id"] = f"SL{range_id}"
                range_id += 1
            elif ranges_tuple:
                result["range_id"] = ranges_info[ranges_tuple]
            else:
                result["range_id"] = "N/A"

    # Get all unique concat sizes across all models
    all_concat_sizes = set()
    for results in model_groups.values():
        for result in results:
            all_concat_sizes.add(result["concat_size"])
    all_concat_sizes = sorted(all_concat_sizes)
    concat_size_to_row = {size: idx for idx, size in enumerate(all_concat_sizes)}
    num_rows = len(all_concat_sizes)

    # Determine max number of range variations per model per concat size
    max_variations_per_model_size = {}
    for model_name, results in model_groups.items():
        concat_size_groups = defaultdict(list)
        for result in results:
            concat_size_groups[result["concat_size"]].append(result)
        for concat_size, size_results in concat_size_groups.items():
            key = (model_name, concat_size)
            max_variations_per_model_size[key] = len(size_results)

    # Calculate global y-limits for each model (for position and positional directional leakage analysis)
    model_ylims = {}
    if analysis_type == "position":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all y-values (means, CI lower, CI upper)
                all_values.extend(result["position_means"])
                all_values.extend(result["position_ci_lower"])
                all_values.extend(result["position_ci_upper"])

            if all_values:
                # Add a small margin (5%) to the range
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                margin = y_range * 0.05
                model_ylims[model_name] = (y_min - margin, y_max + margin)
            else:
                model_ylims[model_name] = None
    elif analysis_type == "positional_directional_leakage":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all y-values from position forward and backward means
                all_values.extend(result["position_forward_means"].values())
                all_values.extend(result["position_backward_means"].values())

                # Also collect Matryoshka results if available
                if "matryoshka_results" in result:
                    for dim_results in result["matryoshka_results"].values():
                        all_values.extend(
                            dim_results["position_forward_means"].values()
                        )
                        all_values.extend(
                            dim_results["position_backward_means"].values()
                        )

            if all_values:
                # Add a small margin (5%) to the range and respect cosine similarity bounds [-1, 1]
                y_min = max(min(all_values) - 0.02, -1.0)
                y_max = min(max(all_values) + 0.02, 1.0)
                model_ylims[model_name] = (y_min, y_max)
            else:
                model_ylims[model_name] = None

    # Calculate global x-limits for each model (for directional leakage analysis only)
    model_xlims = {}
    if analysis_type == "directional_leakage":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all x-values (forward and backward influence values)
                all_values.extend(result["forward_influence"])
                all_values.extend(result["backward_influence"])

            if all_values:
                # Add a small margin and respect cosine similarity bounds [-1, 1]
                x_min = max(min(all_values) - 0.01, -1.0)
                x_max = min(max(all_values) + 0.01, 1.0)
                model_xlims[model_name] = (x_min, x_max)
            else:
                model_xlims[model_name] = None

    # Calculate total columns needed (sum of max variations per model + spacing)
    total_cols = 0
    model_col_start_indices = {}
    model_col_counts = {}  # Store the width of each model's columns
    spacing_offset = 0
    for model_name in model_groups.keys():
        model_col_start_indices[model_name] = total_cols + spacing_offset
        max_cols_for_model = max(
            max_variations_per_model_size.get((model_name, size), 0)
            for size in all_concat_sizes
        )
        model_col_counts[model_name] = max_cols_for_model
        total_cols += max_cols_for_model
        # Add spacing column after each model except the last one
        if model_name != list(model_groups.keys())[-1]:
            spacing_offset += 1

    # Detect if we have only one model for layout optimization
    if single_model_mode is None:
        # Auto-detect based on data
        single_model_mode = len(model_groups.keys()) == 1
    # If explicitly set by user, use that value

    # Create figure with GridSpec for flexible subplot layout
    if single_model_mode:
        # For single model, make subplots square and use full width
        # Calculate optimal dimensions based on number of columns
        max_cols_for_single_model = max(
            max_variations_per_model_size.get((list(model_groups.keys())[0], size), 0)
            for size in all_concat_sizes
        )
        # Make figure wider and shorter to accommodate square subplots
        fig = plt.figure(
            figsize=(
                max_cols_for_single_model
                * 4.5,  # Much wider - about 4.5 inches per column
                num_rows * 3.5,  # Shorter height - about 3.5 inches per row
            )
        )
    else:
        fig = plt.figure(
            figsize=(
                figure_width * len(model_groups.keys()) * 1.0,
                subplot_height * num_rows,
            )
        )

    # Add spacing between models using width_ratios
    width_ratios = []
    for model_name in model_groups.keys():
        max_cols_for_model = max(
            max_variations_per_model_size.get((model_name, size), 0)
            for size in all_concat_sizes
        )
        width_ratios.extend([1] * max_cols_for_model)
        # Add an extra column with moderate width for spacing after each model except the last one
        if model_name != list(model_groups.keys())[-1]:
            width_ratios.append(0.15)  # Spacing between models

    # Adjust total_cols to account for spacing columns
    spacing_cols = len(model_groups) - 1
    total_cols_with_spacing = total_cols + spacing_cols

    # Create GridSpec with custom width ratios and spacing optimized for single/multi model
    if single_model_mode:
        # For single model, optimize spacing for square subplots
        # Can use less spacing since we're hiding redundant axis labels
        gs = GridSpec(
            num_rows,
            total_cols_with_spacing,
            figure=fig,
            width_ratios=width_ratios,
            wspace=0.12,  # Reduced since y-axis labels are hidden on most subplots
            hspace=0.30,  # Reduced since x-axis labels are hidden on most subplots
        )
    else:
        # For multiple models, use optimized spacing since axis labels are now hidden
        gs = GridSpec(
            num_rows,
            total_cols_with_spacing,
            figure=fig,
            width_ratios=width_ratios,
            wspace=0.35,  # Reduced from 0.45 since y-axis labels are hidden on most subplots
            hspace=0.30,  # Reduced from 0.35 since x-axis labels are hidden on most subplots
        )

    # --- Center model names above their columns ---
    # Calculate the center x-position for each model's columns using axes positions
    # Only show model names if we have multiple models
    if not single_model_mode:
        for model_idx, (model_name, results) in enumerate(model_groups.items()):
            col_start = model_col_start_indices[model_name]
            model_width = model_col_counts[model_name]
            # Find the leftmost and rightmost axes for this model in the top row
            top_row = 0
            left_ax = fig.add_subplot(gs[top_row, col_start])
            right_ax = fig.add_subplot(gs[top_row, col_start + model_width - 1])
            left_bbox = left_ax.get_position()
            right_bbox = right_ax.get_position()
            left_ax.remove()
            right_ax.remove()
            # Center x is the midpoint between left and right axes
            center_x = (left_bbox.x0 + right_bbox.x1) / 2
            fig.text(
                center_x,
                0.93,  # Higher up for more space below model names
                model_name,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )

    # Add concat_size label only once per row (on the left)
    concat_size_label_drawn = set()
    for model_idx, (model_name, results) in enumerate(model_groups.items()):
        concat_size_groups = defaultdict(list)
        for result in results:
            concat_size_groups[result["concat_size"]].append(result)

        col_start = model_col_start_indices[model_name]
        model_width = model_col_counts[model_name]

        for concat_size, size_results in concat_size_groups.items():
            row_idx = concat_size_to_row[concat_size]

            for local_col_idx, result in enumerate(size_results):
                col_idx = col_start + local_col_idx

                if local_col_idx >= max_variations_per_model_size.get(
                    (model_name, concat_size), 0
                ):
                    print(
                        f"Warning: Too many results for {model_name} concat size {concat_size}. Skipping extra plots."
                    )
                    break

                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Pass ylim for position analysis or xlim for directional leakage analysis
                if analysis_type == "position":
                    ylim = model_ylims.get(model_name)
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        ylim=ylim,
                        show_segment_lengths=show_segment_lengths,
                    )
                elif analysis_type == "directional_leakage":
                    xlim = model_xlims.get(model_name)
                    plot_in_subplot(
                        ax, result, show_title=True, compact=True, xlim=xlim
                    )
                elif analysis_type == "positional_directional_leakage":
                    ylim = model_ylims.get(model_name)
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        ylim=ylim,
                        show_segment_lengths=show_segment_lengths,
                    )
                else:
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        show_segment_lengths=show_segment_lengths,
                    )

                # Control axis labels visibility
                # Y-axis label: only show on leftmost subplot of each row
                if not (model_idx == 0 and local_col_idx == 0):
                    ax.set_ylabel("")

                # X-axis label: only show on bottom subplot of each column
                if row_idx != num_rows - 1:  # Not the bottom row
                    ax.set_xlabel("")

                # Only draw concat_size label once per row, on the leftmost subplot
                if (
                    row_idx not in concat_size_label_drawn
                    and model_idx == 0
                    and local_col_idx == 0
                ):
                    # Adjust label position based on single model mode
                    x_pos = -0.25 if single_model_mode else -0.50
                    ax.text(
                        x_pos,  # Closer to subplot for single model mode
                        0.5,
                        f"# of Segments: {concat_size}",
                        verticalalignment="center",
                        horizontalalignment="right",
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight="bold",
                        rotation=90,
                    )
                    concat_size_label_drawn.add(row_idx)

            # Fill in any empty spots in the grid with blank subplots
            max_cols_for_this_model_size = max_variations_per_model_size.get(
                (model_name, concat_size), 0
            )
            for local_col_idx in range(len(size_results), max_cols_for_this_model_size):
                col_idx = col_start + local_col_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

            # Add spacing column after each model except the last one
            if model_name != list(model_groups.keys())[-1]:
                spacer_col_idx = col_start + max_cols_for_this_model_size
                ax = fig.add_subplot(gs[row_idx, spacer_col_idx])
                ax.axis("off")

    # Adjust layout - leave more space at top for main title and bottom for legends
    if single_model_mode:
        # For single model, adjust for wider/shorter layout
        plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.93])  # More margins on all sides
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # More space at top and bottom

    # Add overall title
    if single_model_mode:
        # For single model, position title higher and include model name in title
        single_model_name = list(model_groups.keys())[0]
        y_title = 0.96  # Adjusted for new layout
        y_subtitle = y_title - 0.035

        if analysis_type == "position":
            # Include model name in the title for single model
            title_with_model = f"{title} - {single_model_name}"
            fig.suptitle(
                title_with_model,
                fontsize=22,
                fontweight="bold",
                y=y_title,
            )
        elif analysis_type in ("directional_leakage", "positional_directional_leakage"):
            # Include model name in the title for single model
            title_with_model = f"{title} - {single_model_name}"
            fig.suptitle(
                title_with_model,
                fontsize=22,
                fontweight="bold",
                y=y_title,
            )

            # Add legend explaining directional leakage for single model
            subtitle_text = (
                "Forward Similarity (F): Earlier segments' standalone embeddings ↔ Later segments' contextualized embeddings\n"
                "Backward Similarity (B): Later segments' standalone embeddings ↔ Earlier segments' contextualized embeddings"
            )
            fig.text(
                0.5,
                y_subtitle,  # Position between title and plots
                subtitle_text,
                ha="center",
                va="top",
                fontsize=16,  # Slightly smaller for single model
            )
    else:
        # For multiple models, use original positioning
        y_title = 1.03
        y_subtitle = y_title - 0.03

        if analysis_type == "position":
            # Split into two lines for different font sizes for position analysis
            fig.suptitle(
                title,
                fontsize=24,
                fontweight="bold",
                y=y_title,
            )
        elif analysis_type in ("directional_leakage", "positional_directional_leakage"):
            # Directional leakage title
            fig.suptitle(
                title,
                fontsize=24,
                fontweight="bold",
                y=y_title,
            )

        # Add legend explaining directional leakage
        if analysis_type in ("directional_leakage", "positional_directional_leakage"):
            subtitle_text = (
                "Forward Similarity (F): Earlier segments' standalone embeddings ↔ Later segments' contextualized embeddings\n"
                "Backward Similarity (B): Later segments' standalone embeddings ↔ Earlier segments' contextualized embeddings"
            )
            fig.text(
                0.5,
                y_subtitle,  # Position between title and plots
                subtitle_text,
                ha="center",
                va="top",
                fontsize=18,
            )

    # Add custom legends based on analysis type
    if single_model_mode:
        legend_y = 0.01  # Lower position for single model due to new layout
    else:
        legend_y = 0.05
    if analysis_type == "position":
        # Position analysis legend - check if Matryoshka dimensions are present
        handles = []
        labels = []

        # Always include full embedding
        mean_line = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Full Embedding",
        )
        handles.append(mean_line)

        # Add Matryoshka dimensions if present (check first result for Matryoshka data)
        has_matryoshka = any("matryoshka_results" in result for result in all_results)
        if has_matryoshka and matryoshka_dimensions:
            colors = [
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]
            for i, dim in enumerate(matryoshka_dimensions):
                color = colors[i % len(colors)]
                mat_line = mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="--" if i >= 4 else "-",
                    linewidth=2,
                    label=f"Matryoshka D{dim}",
                )
                handles.append(mat_line)

        # Add confidence interval patch
        ci_patch = mpatches.Patch(
            color="blue", alpha=0.2, label="95% Confidence Interval"
        )
        handles.append(ci_patch)

        # Determine number of columns based on number of handles
        ncol = min(len(handles), 6)  # Maximum 6 columns to avoid overcrowding

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            fontsize=9,
            frameon=False,
            columnspacing=2.0,
            handletextpad=1.2,
        )
    elif analysis_type == "positional_directional_leakage":
        # Position analysis legend - check if Matryoshka dimensions are present
        handles = []
        labels = []

        forward_line = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Forward Similarity",
        )
        backward_line = mlines.Line2D(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Backward Similarity",
        )
        handles.append(forward_line)
        handles.append(backward_line)

        # Add Matryoshka dimensions if present (check first result for Matryoshka data)
        has_matryoshka = any("matryoshka_results" in result for result in all_results)
        if has_matryoshka and matryoshka_dimensions:
            colors = [
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]
            for i, dim in enumerate(matryoshka_dimensions):
                color = colors[i % len(colors)]
                mat_line = mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="--" if i >= 4 else "-",
                    linewidth=2,
                    label=f"Matryoshka D{dim}",
                )
                handles.append(mat_line)

        # Add confidence interval patch
        # ci_patch = mpatches.Patch(
        #     color="blue", alpha=0.2, label="95% Confidence Interval"
        # )
        # handles.append(ci_patch)

        # Determine number of columns based on number of handles
        ncol = min(len(handles), 6)  # Maximum 6 columns to avoid overcrowding

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            fontsize=9,
            frameon=False,
            columnspacing=2.0,
            handletextpad=1.2,
        )
    elif analysis_type == "directional_leakage":
        # Directional leakage legend
        blue_patch = mpatches.Patch(color="blue", alpha=0.6, label="Forward Similarity")
        orange_patch = mpatches.Patch(
            color="orange", alpha=0.6, label="Backward Similarity"
        )
        fig.legend(
            handles=[blue_patch, orange_patch],
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=2,
            fontsize=10,
            frameon=False,
            columnspacing=2.5,
            handletextpad=1.5,
        )

    # Create pooling strategy legend text (only if show_segment_lengths is True)
    if show_segment_lengths:
        # Move the legend box further down to avoid overlap with the similarity legend
        if single_model_mode:
            y_pos = -0.03  # Adjusted for new layout
        else:
            y_pos = -0.07  # Lowered from 0 to -0.08

        pooling_legend_lines = ["Pooling Strategies:"]
        if pooling_legend_type == "segment_standalone":
            pooling_legend_lines.append(
                f"Segment Standalone: {pooling_strategy_segment_standalone.upper()}"
            )
        elif pooling_legend_type == "segment_standalone_and_document":
            pooling_legend_lines.append(
                f"Segment Standalone: {pooling_strategy_segment_standalone.upper()}"
            )
            pooling_legend_lines.append(
                f"Document-Level: {pooling_strategy_document.upper()}"
            )
        pooling_legend_text = "\n".join(pooling_legend_lines)

        # Add ranges and pooling strategy legend in a single box if there are different ranges
        if ranges_info:
            # Build segment lengths legend
            legend_lines = ["Segment Lengths:"]
            for ranges_tuple, range_id in ranges_info.items():
                ranges_str = " | ".join(
                    [f"{start}-{end}" for start, end in ranges_tuple]
                )
                legend_lines.append(f"{range_id}=({ranges_str})")
            # Force correct label for the first line (avoid LaTeX/spacing issues)
            segment_lengths_legend_text = (
                r"$\mathbf{Segment\ Lengths:}$" + "\n" + "\n".join(legend_lines[1:])
            )

            # Pooling legend: only first line bold, force correct label
            pooling_legend_lines = pooling_legend_text.split("\n")
            pooling_legend_lines[0] = "Pooling Strategies:"
            pooling_legend_text = (
                r"$\mathbf{Pooling\ Strategies:}$"
                + "\n"
                + "\n".join(pooling_legend_lines[1:])
            )

            # Combine both legends horizontally with spacing
            combined_legend_text = (
                segment_lengths_legend_text + "\n\n" + pooling_legend_text
            )
            # Use a single box, with left alignment
            fig.text(
                0.5,  # Centered horizontally
                y_pos,
                combined_legend_text,
                ha="center",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.7"),
                usetex=False,
            )
        else:
            # Only show pooling strategies legend centered
            pooling_legend_lines = pooling_legend_text.split("\n")
            pooling_legend_lines[0] = "Pooling Strategies:"
            pooling_legend_text = (
                r"$\mathbf{Pooling\ Strategies:}$"
                + "\n"
                + "\n".join(pooling_legend_lines[1:])
            )
            fig.text(
                0.5,  # Center position for the pooling strategies legend
                y_pos,
                pooling_legend_text,
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.5"),
                usetex=False,
            )

    # For directional leakage, print comparison statistics
    if analysis_type == "directional_leakage" and False:
        print("\nStatistical Comparison of Forward vs Backward Similarity:")
        for result in all_results:
            path_name = result.get("path_name", "Unknown")
            print(f"{path_name}")
            print(f"  Forward Mean: {result['forward_mean']:.4f}")
            print(f"  Backward Mean: {result['backward_mean']:.4f}")
            print(
                f"  Difference (F-B): {result['forward_mean'] - result['backward_mean']:.4f}"
            )

            # Report t-test
            t_stat = result.get("t_stat", 0)
            p_value = result.get("p_value", 1.0)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"  T-test: t={t_stat:.3f}, p={p_value:.6f} ({significance})\n")

    # Save plot if requested
    if save_plot:
        if save_path is None:
            filename = f"multi_{analysis_type}_analysis"
            if analysis_type == "position":
                filename += f"_{pooling_strategy_segment_standalone}_{pooling_strategy_document}"
            filename += ".png"
            save_path = os.path.join(str(paths[0]), filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    if return_full_results:
        return {"model_results": dict(model_groups)}

    return None
