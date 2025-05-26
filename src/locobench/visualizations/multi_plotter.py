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
)

from .single_plotters import (
    DirectionalLeakageSinglePlotter,
    PositionSimilaritySinglePlotter,
)


class DirectionalLeakageMultiPlotter:
    """
    Class to handle the plotting of directional leakage analysis results in a multi plot.
    """

    def __init__(self):
        self.analysis_type = "directional_leakage"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot directional leakage for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
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
        directional_leakage_single_plotter = DirectionalLeakageSinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Directional Information Flow",
            pooling_legend_type="segment_standalone",
            subplotter=directional_leakage_single_plotter.plot_directional_leakage_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
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
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
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
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
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
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
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
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
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
    figure_width: int = 15,
    subplot_height: int = 4,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    return_full_results: bool = False,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Analyze and plot multiple experiment results in a grid, organized by model name and concat size.
    Supports both position similarity and directional leakage analysis.

    Args:
        paths: List of paths to experiment results
        analysis_type: Type of analysis to perform ("position" or "directional_leakage")
        pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
        pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
                                      (only used for position analysis)
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

    # Create figure with GridSpec for flexible subplot layout
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

    # Create GridSpec with custom width ratios and increased wspace and hspace
    gs = GridSpec(
        num_rows,
        total_cols_with_spacing,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.25,  # More space between subplots horizontally
        hspace=0.35,  # More space between rows
    )

    # --- Center model names above their columns ---
    # Calculate the center x-position for each model's columns using axes positions
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
                plot_in_subplot(ax, result, show_title=True, compact=True)

                # Only draw concat_size label once per row, on the leftmost subplot
                if (
                    row_idx not in concat_size_label_drawn
                    and model_idx == 0
                    and local_col_idx == 0
                ):
                    ax.text(
                        -0.30,  # Position for label
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
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # More space at top and bottom

    # Add overall title
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
        # subtitle_text = f"Document-level Pooling Strategy: {pooling_strategy_document}, Standalone Segment Pooling Strategy: {pooling_strategy_segment_standalone}"
        # # Add the second line as a separate text object with fontsize=18
        # fig.text(
        #     0.5,
        #     y_subtitle,
        #     subtitle_text,
        #     ha="center",
        #     va="top",
        #     fontsize=18,
        # )

    elif analysis_type == "directional_leakage":
        # Directional leakage title
        fig.suptitle(
            title,
            fontsize=24,
            fontweight="bold",
            y=y_title,
        )

        # Add legend explaining directional leakage
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
    legend_y = 0.05
    if analysis_type == "position":
        # Position analysis legend
        mean_line = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Mean Similarity",
        )
        ci_patch = mpatches.Patch(
            color="blue", alpha=0.2, label="95% Confidence Interval"
        )
        fig.legend(
            handles=[mean_line, ci_patch],
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=2,
            fontsize=10,
            frameon=False,
            columnspacing=2.5,
            handletextpad=1.5,
        )
    else:
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

    # Create pooling strategy legend text
    # Move the legend box further down to avoid overlap with the similarity legend
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
            ranges_str = " | ".join([f"{start}-{end}" for start, end in ranges_tuple])
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
