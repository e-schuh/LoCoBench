import os

from typing import Dict, List, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import numpy as np

# Import the required analyses functions
from ..core.segment_embedding_analysis import (
    DocumentSegmentSimilarityAnalyzer,
    DirectionalLeakageAnalyzer,
    PositionalDirectionalLeakageAnalyzer,
)


class PositionSimilaritySinglePlotter:
    """
    Class for plotting position-based similarity results in a single plot.
    """

    def __init__(self):
        pass

    def plot_position_similarities(
        self,
        results: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot position-based similarity results from run_position_analysis().

        Args:
            results: Dictionary returned from run_position_analysis()
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the figure (if None, figure is not saved)
            show_plot: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        # Extract data from results
        position_means = results["position_means"]
        position_ci_lower = results["position_ci_lower"]
        position_ci_upper = results["position_ci_upper"]
        positions = list(
            range(1, len(position_means) + 1)
        )  # 1-based positions for x-axis

        # Extract metadata for plot title
        model_name = results.get("model_name", "Unknown model")
        concat_size = results.get("concat_size", 0)
        position_ranges = results.get("position_specific_ranges", [])
        pooling_standalone_segment = results.get(
            "pooling_strategy_segment_standalone", "cls"
        )
        pooling_document = results.get("pooling_strategy_document", "cls")

        # Format position ranges for title
        ranges_str = " | ".join([f"{start}-{end}" for start, end in position_ranges])
        if not ranges_str:
            ranges_str = "not specified"

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot means with a line
        ax.plot(
            positions,
            position_means,
            "o-",
            color="blue",
            linewidth=2,
            label="Mean Similarity",
        )

        # Add 95% confidence interval bars around each point (not along the line)
        for i, (pos, mean, ci_lower, ci_upper) in enumerate(
            zip(positions, position_means, position_ci_lower, position_ci_upper)
        ):
            # Width of the confidence interval bar
            bar_width = 0.4
            # Create rectangle patch for the confidence interval
            rect = plt.Rectangle(
                (pos - bar_width / 2, ci_lower),  # (x, y) of bottom left corner
                bar_width,  # width
                ci_upper - ci_lower,  # height
                color="blue",
                alpha=0.2,
            )
            ax.add_patch(rect)

        # Add a dummy patch for the legend
        dummy_rect = plt.Rectangle(
            (0, 0), 1, 1, color="blue", alpha=0.2, label="95% Confidence Interval"
        )

        # Add labels and title
        ax.set_xlabel("Segment Position")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(
            f"Position-based Similarity Analysis\n{model_name} (standalone: {pooling_standalone_segment}, document: {pooling_document})\n"
            f"concat size: {concat_size}, ranges: {ranges_str}"
        )

        # Set x-axis ticks to be integers
        ax.set_xticks(positions)

        # Format y-axis to show at most 3 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Add grid for better readability
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend including the dummy rectangle for confidence interval
        handles, labels = ax.get_legend_handles_labels()
        handles.append(dummy_rect)
        ax.legend(handles=handles)

        # Add sample counts as annotations above each point
        for i, count in enumerate(results["position_counts"]):
            ax.annotate(
                f"n={count}",
                xy=(positions[i], position_means[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    def analyze_and_plot(
        self,
        path: str | Path,
        document_embedding_type: str = "document-level",
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        figsize: Tuple[int, int] = (10, 6),
        save_plot: bool = False,
        show_plot: bool = True,
        return_full_results: bool = False,
        return_t_stats: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze and plot position-based similarity metrics for a given experiment.

        Args:
            path: Path to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
            figsize: Size of the figure (width, height) in inches
            save_plot: Whether to save the plot in the experiment directory
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            return_t_stats: Whether to perform and return pairwise t-tests between position similarities

        Returns:
            Dict with position-based similarity results (same as run_position_analysis)
            If return_t_stats is True, includes t-test statistics for pairwise position comparisons
        """
        # Run the position analysis
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        results = doc_seg_analyzer.run_position_analysis(
            path=path,
            document_embedding_type=document_embedding_type,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
        )

        # Generate a save path if requested
        save_path = None
        if save_plot:
            # Extract a filename from the path
            path_str = str(path)
            if isinstance(path, str):
                base_dir = path
            else:  # Path object
                base_dir = str(path)

            filename = f"position_analysis_{pooling_strategy_segment_standalone}_{pooling_strategy_document}.png"
            save_path = os.path.join(base_dir, filename)

        # Plot the results
        self.plot_position_similarities(
            results=results, figsize=figsize, save_path=save_path, show_plot=show_plot
        )

        # Perform pairwise t-tests if requested
        if return_t_stats:
            # Get position similarities from results
            position_similarities = results["position_similarities"]

            # Compute t-test results
            t_test_results = self.compute_position_t_tests(position_similarities)

            # Print results to standard output
            print("\nPairwise position t-tests (one-tailed, H1: pos_i > pos_j):")
            print(t_test_results["t_test_df"].to_string(index=False))

            # Add t-test results to the return dictionary
            if return_full_results:
                results["t_test_results"] = t_test_results
                return results
            else:
                return t_test_results

        # Return results if requested
        elif return_full_results:
            return results

        return None

    def plot_position_similarities_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot position-based similarity results in a given subplot.

        Args:
            ax: Matplotlib Axes object to plot on
            results: Dictionary returned from run_position_analysis()
            show_title: Whether to show the title (default: True)
            compact: Whether to use a compact plot style for multi-plot figures (default: True)
            ylim: Optional tuple specifying (min, max) y-axis limits
            show_segment_lengths: Whether to show segment length information in title and legends (default: True)
        """
        # Extract data from results
        position_means = results["position_means"]
        position_ci_lower = results["position_ci_lower"]
        position_ci_upper = results["position_ci_upper"]
        positions = list(
            range(1, len(position_means) + 1)
        )  # 1-based positions for x-axis

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        # Use abbreviated model name if available
        model_name = results.get(
            "abbreviated_model_name", results.get("model_name", "Unknown model")
        )

        # Plot means with a line (full embeddings)
        ax.plot(
            positions,
            position_means,
            "o-",
            color="blue",
            linewidth=2,
            label="Full",
        )

        # Add 95% confidence interval for full embeddings
        for i, (pos, mean, ci_lower, ci_upper) in enumerate(
            zip(positions, position_means, position_ci_lower, position_ci_upper)
        ):
            # Width of the confidence interval bar
            bar_width = 0.4
            # Create rectangle patch for the confidence interval
            rect = plt.Rectangle(
                (pos - bar_width / 2, ci_lower),  # (x, y) of bottom left corner
                bar_width,  # width
                ci_upper - ci_lower,  # height
                color="blue",
                alpha=0.2,
            )
            ax.add_patch(rect)

        # Plot Matryoshka dimensions if available
        if "matryoshka_results" in results and "matryoshka_dimensions" in results:
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
            matryoshka_results = results["matryoshka_results"]
            matryoshka_dimensions = results["matryoshka_dimensions"]

            for i, dim in enumerate(matryoshka_dimensions):
                if dim in matryoshka_results:
                    color = colors[i % len(colors)]
                    dim_means = matryoshka_results[dim]["position_means"]
                    dim_ci_lower = matryoshka_results[dim]["position_ci_lower"]
                    dim_ci_upper = matryoshka_results[dim]["position_ci_upper"]

                    # Plot means for this dimension
                    ax.plot(
                        positions,
                        dim_means,
                        "o-",
                        color=color,
                        linewidth=2,
                        label=f"D{dim}",
                        linestyle=(
                            "--" if i >= 4 else "-"
                        ),  # Use dashed lines for dimensions 5 and beyond
                    )

                    # Add confidence intervals for this dimension
                    for j, (pos, mean, ci_lower, ci_upper) in enumerate(
                        zip(positions, dim_means, dim_ci_lower, dim_ci_upper)
                    ):
                        bar_width = 0.3  # Slightly narrower for Matryoshka dimensions
                        rect = plt.Rectangle(
                            (pos - bar_width / 2, ci_lower),
                            bar_width,
                            ci_upper - ci_lower,
                            color=color,
                            alpha=0.15,  # More transparent for Matryoshka dimensions
                        )
                        ax.add_patch(rect)

        # Add labels (compact for subplots)
        ax.set_xlabel("Position")
        ax.set_ylabel("Cosine Similarity")

        # Add segment length as subtitle instead of text in the plot
        if show_title:
            current_title = ax.get_title()

            # Get language information if available
            source_lang = results.get("source_lang")
            target_lang = results.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Languages: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Languages: [{source_lang}, ..., {source_lang}]"

            if show_segment_lengths:
                title_text += f"; SL:: {range_id}"

            if current_title:
                ax.set_title(f"{current_title}\n{title_text}", fontsize=9)
            else:
                ax.set_title(title_text, fontsize=9)

        # Set x-axis ticks to be integers
        ax.set_xticks(positions)

        # Set y-axis limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # Format y-axis to show at most 3 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Add grid for better readability
        ax.grid(True, linestyle="--", alpha=0.7)

        # Note: Individual subplot legends are not shown - main legend is at the bottom of the figure

    def compute_position_t_tests(
        self,
        position_similarities: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Compute pairwise t-tests between position similarities.

        Args:
            position_similarities: List of lists containing similarity scores for each position

        Returns:
            Dictionary containing t-test results:
            - t_stats: Dictionary mapping comparison key to t-statistic
            - p_values: Dictionary mapping comparison key to p-value
            - significant: Dictionary mapping comparison key to significance boolean
            - t_test_df: Pandas DataFrame with all test results formatted for display
        """
        from scipy import stats
        import pandas as pd

        num_positions = len(position_similarities)

        # Initialize dictionaries to store t-test results
        t_stats = {}
        p_values = {}
        significant = {}

        # Perform pairwise t-tests (pos_i > pos_j)
        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                # Only perform test if both positions have samples
                if position_similarities[i] and position_similarities[j]:
                    # Perform one-tailed t-test (is position i > position j?)
                    t_stat, p_val = stats.ttest_ind(
                        position_similarities[i],
                        position_similarities[j],
                        alternative="greater",
                    )

                    key = f"pos{i+1}>pos{j+1}"
                    t_stats[key] = t_stat
                    p_values[key] = p_val
                    significant[key] = p_val < 0.05

        # Create DataFrame for easier reading
        t_test_df = pd.DataFrame(
            {
                "Comparison": list(t_stats.keys()),
                "t-statistic": list(t_stats.values()),
                "p-value": list(p_values.values()),
                "Significant (p<0.05)": list(significant.values()),
            }
        )

        # Create return dictionary with all results
        t_test_results = {
            "t_stats": t_stats,
            "p_values": p_values,
            "significant": significant,
            "t_test_df": t_test_df,
        }

        return t_test_results


class DirectionalLeakageSinglePlotter:
    """
    Class to handle plotting of directional leakage results in a single plot.
    """

    def __init__(self):
        pass

    def plot_directional_leakage(
        self,
        results: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot directional leakage analysis for a single experiment.

        Args:
            results: Results from run_directional_leakage_analysis
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the plot (if None, plot is not saved)
            show_plot: Whether to display the plot

        Returns:
            The matplotlib figure object
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the directional leakage results
        self.plot_directional_leakage_in_subplot(
            ax, results, show_title=True, compact=False
        )

        # Set specific title for a standalone segment plot
        model_name = results["model_name"]
        path_name = results.get("path_name", "Unknown")

        # Set a more detailed title for single plots
        plt.title(
            f"Directional Information Leakage\n{model_name}\n{path_name}", fontsize=12
        )

        # Set labels
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")

        # Add t-test results as text
        t_stat = results.get("t_stat", 0)
        p_value = results.get("p_value", 1.0)
        significance = (
            "statistically significant"
            if p_value < 0.05
            else "not statistically significant"
        )

        plt.figtext(
            0.5,
            -0.05,
            f"T-test results: t={t_stat:.4f}, p={p_value:.6f}\nThe difference is {significance} at α=0.05",
            ha="center",
            fontsize=10,
        )

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    def analyze_and_plot(
        self,
        path: str | Path,
        figsize: Tuple[int, int] = (10, 6),
        save_plot: bool = False,
        show_plot: bool = True,
        return_full_results: bool = False,
        pooling_strategy_segment_standalone: str = "cls",
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze and plot directional leakage for a single experiment.

        Args:
            path: Path to experiment results
            figsize: Size of the figure (width, height) in inches
            save_plot: Whether to save the plot in the experiment directory
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results

        Returns:
            If return_full_results is True, returns the directional leakage results
        """
        # Run the directional leakage analysis
        directional_leakage_analyzer = DirectionalLeakageAnalyzer()
        results = directional_leakage_analyzer.run_directional_leakage_analysis(
            path,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
        )

        # Generate a save path if requested
        save_path = None
        if save_plot:
            if isinstance(path, str):
                base_dir = path
            else:  # Path object
                base_dir = str(path)

            filename = "directional_leakage_analysis.png"
            save_path = os.path.join(base_dir, filename)

        # Plot the results
        self.plot_directional_leakage(
            results=results, figsize=figsize, save_path=save_path, show_plot=show_plot
        )

        # Return results if requested
        if return_full_results:
            return results

        return None

    def plot_directional_leakage_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot directional leakage results in an existing subplot.

        Args:
            ax: Matplotlib axes to plot on
            results: Results from run_directional_leakage_analysis
            show_title: Whether to show the title
            compact: Whether to use a compact plot style for multi-plot figures
            xlim: Optional tuple specifying (min, max) x-axis limits
        """
        # Create histograms comparing forward and backward influence
        num_bins = 20 if compact else 30
        alpha = 0.6 if compact else 0.7
        num_bins = 30
        alpha = 0.7

        # Plot histograms
        ax.hist(
            results["forward_influence"],
            alpha=alpha,
            label=f"F: {results['forward_mean']:.3f}" if compact else "Forward",
            bins=num_bins,
            color="blue",
        )
        ax.hist(
            results["backward_influence"],
            alpha=alpha,
            label=f"B: {results['backward_mean']:.3f}" if compact else "Backward",
            bins=num_bins,
            color="orange",
        )

        # Add mean lines
        ax.axvline(
            x=results["forward_mean"],
            color="blue",
            linestyle="--",
            label=None if compact else f"Forward Mean: {results['forward_mean']:.4f}",
        )
        ax.axvline(
            x=results["backward_mean"],
            color="orange",
            linestyle="--",
            label=None if compact else f"Backward Mean: {results['backward_mean']:.4f}",
        )

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        if show_title:
            # Extract model name and size info
            model_name = results["model_name"]
            # Abbreviate model names
            if "Alibaba-NLP/gte-multilingual-base" in model_name:
                model_name = "mGTE"
            elif "jinaai/jina-embeddings-v3" in model_name:
                model_name = "jina-v3"
            # Store result with abbreviated model name
            results["abbreviated_model_name"] = model_name

            # Always add segment length as a subtitle
            if compact:
                # For multi-plot displays, show the segment length and language info if available
                # Get language information if available
                source_lang = results.get("source_lang")
                target_lang = results.get("target_lang")

                # Add language information if available
                if target_lang and source_lang:
                    title_text = (
                        f"Languages: [{target_lang}, {source_lang}, ..., {source_lang}]"
                    )
                elif source_lang:
                    title_text = f"Languages: [{source_lang}, ..., {source_lang}]"

                if show_segment_lengths:
                    title_text += f"; SL:: {range_id}"

                ax.set_title(title_text, fontsize=9)
            else:
                # For single plots, show more detailed information
                size_info = str(results["concat_size"])
                # Extract ranges information
                ranges = []
                if results["position_specific_ranges"]:
                    for start, end in results["position_specific_ranges"]:
                        ranges.append(f"{start}-{end}")
                ranges_str = " | ".join(ranges) if ranges else "N/A"

                # Get language information if available
                source_lang = results.get("source_lang")
                target_lang = results.get("target_lang")

                title_text = f"{model_name} (size {size_info})\nSL: {range_id}"

                # Add language information if available
                if target_lang and source_lang:
                    title_text += f"; Lang:: f: {target_lang}, re: {source_lang}"
                elif source_lang:
                    title_text += f"; Lang:: {source_lang}"

                ax.set_title(
                    title_text,
                    fontsize=9 if compact else 12,
                )

        # Always set axis labels, even in compact mode
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")

        # Add legend
        ax.legend(fontsize=8 if compact else 10)

        # Set x-axis limits
        if xlim is not None:
            # Use provided global limits for consistent scaling across model
            ax.set_xlim(xlim)
        elif (
            len(results["forward_influence"]) > 0
            and len(results["backward_influence"]) > 0
        ):
            # Fallback to automatic limits if no global limits provided
            all_values = results["forward_influence"] + results["backward_influence"]
            min_val = max(min(all_values) - 0.01, -1.0)  # Cosine sim range is [-1, 1]
            max_val = min(max(all_values) + 0.01, 1.0)
            ax.set_xlim(min_val, max_val)


class PositionalDirectionalLeakageSinglePlotter:
    """
    Class for plotting positional directional leakage results in a single plot.
    """

    def __init__(self):
        pass

    def plot_positional_directional_leakage_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot positional directional leakage results in a given subplot.

        Args:
            ax: Matplotlib Axes object to plot on
            results: Dictionary returned from run_positional_directional_leakage_analysis()
            show_title: Whether to show the title (default: True)
            compact: Whether to use a compact plot style for multi-plot figures (default: True)
            ylim: Optional tuple specifying (min, max) y-axis limits
            show_segment_lengths: Whether to show segment length information in title and legends (default: True)
        """
        # Extract data from results
        position_forward_means = results["position_forward_means"]
        position_backward_means = results["position_backward_means"]
        all_positions = results["all_positions"]

        # Convert to lists for plotting, ensuring 1-based indexing
        positions = list(range(1, len(all_positions) + 1))
        forward_means = [position_forward_means.get(pos, 0) for pos in all_positions]
        backward_means = [
            position_backward_means.get(pos, 0) for pos in all_positions
        ]  # Plot forward influence (excluding last position which is always 0)
        forward_positions = positions[:-1] if len(positions) > 1 else positions
        forward_values = forward_means[:-1] if len(forward_means) > 1 else forward_means

        if forward_positions:
            ax.plot(
                forward_positions,
                forward_values,
                "o",
                color="blue",
                linewidth=2,
                linestyle="-",
                # label="Forward",
                markersize=5,
            )

        # Plot backward influence (excluding first position which is always 0)
        backward_positions = positions[1:] if len(positions) > 1 else []
        backward_values = backward_means[1:] if len(backward_means) > 1 else []

        if backward_positions:
            ax.plot(
                backward_positions,
                backward_values,
                "o",
                color="orange",
                linewidth=2,
                linestyle="-",
                # label="Backward",
                markersize=5,
            )

        # Plot Matryoshka dimensions if available
        if "matryoshka_results" in results and "matryoshka_dimensions" in results:
            colors = [
                "red",
                "green",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]
            matryoshka_results = results["matryoshka_results"]
            matryoshka_dimensions = results["matryoshka_dimensions"]

            for i, dim in enumerate(matryoshka_dimensions):
                if dim in matryoshka_results:
                    color = colors[i % len(colors)]
                    dim_forward_means = matryoshka_results[dim][
                        "position_forward_means"
                    ]
                    dim_backward_means = matryoshka_results[dim][
                        "position_backward_means"
                    ]

                    # Convert to lists for plotting
                    dim_forward_values = [
                        dim_forward_means.get(pos, 0) for pos in all_positions
                    ]
                    dim_backward_values = [
                        dim_backward_means.get(pos, 0) for pos in all_positions
                    ]

                    # Plot forward influence for this dimension (excluding last position)
                    dim_forward_positions = forward_positions
                    dim_forward_vals = (
                        dim_forward_values[:-1]
                        if len(dim_forward_values) > 1
                        else dim_forward_values
                    )

                    if dim_forward_positions:
                        ax.plot(
                            dim_forward_positions,
                            dim_forward_vals,
                            "o",
                            color=color,
                            linewidth=2,
                            label=f"F D{dim}",
                            linestyle="--",
                            markersize=4,
                        )

                    # Plot backward influence for this dimension (excluding first position)
                    dim_backward_positions = backward_positions
                    dim_backward_vals = (
                        dim_backward_values[1:] if len(dim_backward_values) > 1 else []
                    )

                    if dim_backward_positions:
                        ax.plot(
                            dim_backward_positions,
                            dim_backward_vals,
                            "s",
                            color=color,
                            linewidth=2,
                            label=f"B D{dim}",
                            linestyle=":",
                            markersize=4,
                        )

        # Add average lines (excluding zeros)
        # Calculate average forward influence (excluding last position which is 0)
        if forward_values:
            forward_avg = np.mean(forward_values)
            ax.axhline(
                y=forward_avg, color="blue", linestyle="--", alpha=0.7, linewidth=1
            )

        # Calculate average backward influence (excluding first position which is 0)
        if backward_values:
            backward_avg = np.mean(backward_values)
            ax.axhline(
                y=backward_avg, color="orange", linestyle="--", alpha=0.7, linewidth=1
            )

        # Add labels
        ax.set_xlabel("Position")
        ax.set_ylabel("Cosine Similarity")

        # Set y-axis limits if provided
        if ylim:
            ax.set_ylim(ylim)

        # Add legend (compact for subplots)
        if compact:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.legend(loc="best")

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        # Use abbreviated model name if available
        model_name = results.get(
            "abbreviated_model_name", results.get("model_name", "Unknown model")
        )

        if show_title:
            current_title = ax.get_title()

            # Get language information if available
            source_lang = results.get("source_lang")
            target_lang = results.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Languages: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Languages: [{source_lang}, ..., {source_lang}]"
            else:
                title_text = "Position-wise Directional Influence"

            if show_segment_lengths:
                # Add range information as subtitle
                title_text += f"; SL:: {range_id}"

            if current_title:
                ax.set_title(f"{current_title}\n{title_text}", fontsize=9)
            else:
                ax.set_title(title_text, fontsize=9)

        # Set integer x-axis ticks
        ax.set_xticks(positions)

        # Grid for better readability
        ax.grid(True, alpha=0.3)
