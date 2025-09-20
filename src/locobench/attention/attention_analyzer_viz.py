"""
Visualization utilities for attention analysis results.

This module provides plotting helpers to visualize the JSON results produced by
`attention_analyzer.analyze_from_config`.

Functions are designed to be called from notebooks or CLIs.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _robust_log_norm(
    mat: np.ndarray, low: float = 1.0, high: float = 99.0
) -> colors.LogNorm:
    """Return a LogNorm with robust percentile clipping.

    Assumes non-negative values; zeros allowed. Uses [low, high] percentiles to
    reduce the impact of outliers.
    """
    p_low, p_high = np.nanpercentile(mat, [low, high])
    vmin = max(float(p_low), 1e-12)
    vmax = float(p_high)
    assert vmax > vmin
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _robust_linear_range(
    mat: np.ndarray, low: float = 1.0, high: float = 95.0
) -> tuple[float, float]:
    """Return (vmin, vmax) from robust percentiles for linear scaling."""
    vmin, vmax = np.nanpercentile(mat, [low, high])
    vmin = float(vmin)
    vmax = float(vmax)
    assert vmax > vmin
    return vmin, vmax


def _plot_special_tokens(
    analysis: Dict[str, Any], out_dir: str, title_prefix: str
) -> None:
    first = analysis.get("first_token")
    last = analysis.get("last_token")
    if first is None and last is None:
        return
    L = None
    first_vals = None
    last_vals = None
    if first is not None:
        first_vals = np.array(first.get("per_layer_means", []), dtype=float)
        L = len(first_vals)
    if last is not None:
        last_vals = np.array(last.get("per_layer_means", []), dtype=float)
        L = len(last_vals) if L is None else min(L, len(last_vals))
    if L is None:
        return
    x = np.arange(L)

    plt.figure(figsize=(8, 4))
    if first_vals is not None and first_vals.size > 0:
        plt.plot(x, first_vals[:L], marker="o", label="first token")
    if last_vals is not None and last_vals.size > 0:
        plt.plot(x, last_vals[:L], marker="s", label="last token")
    plt.xlabel("Layer index")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Special tokens: per-layer means")
    plt.legend()
    _savefig(os.path.join(out_dir, f"{title_prefix}__special_tokens__per_layer.png"))


def _plot_baskets_absolute(
    analysis: Dict[str, Any], out_dir: str, title_prefix: str
) -> None:
    data = analysis.get("baskets_absolute")
    assert data is not None
    means = np.array(data["per_layer_bin_means"], dtype=float)  # [L, B]
    counts = np.array(data["counts"], dtype=float)
    assert means.shape == counts.shape
    L, B = means.shape

    # Heatmap (layers x bins)
    plt.figure(figsize=(max(6, B * 0.2), max(4, L * 0.4)))
    plt.imshow(means, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Incoming attention (mean)")
    plt.xlabel("Absolute bin index")
    plt.ylabel("Layer index")
    plt.title(f"{title_prefix} Absolute baskets: layer x bin")
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_absolute__heatmap.png"))

    # Per-layer curves
    plt.figure(figsize=(8, 5))
    x = np.arange(B)
    for l in range(L):
        plt.plot(x, means[l], alpha=0.75, label=f"L{l}")
    plt.xlabel("Absolute bin index")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Absolute baskets: per-layer curves")
    if L <= 16:
        plt.legend(ncol=2, fontsize=8)
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_absolute__curves.png"))

    # Layer-average curve (ignore bins with zero counts in all layers)
    valid = counts.sum(axis=0) > 0
    avg_curve = np.where(
        valid, np.nanmean(np.where(valid, means, np.nan), axis=0), np.nan
    )
    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_curve, color="black")
    plt.xlabel("Absolute bin index")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Absolute baskets: layer-avg curve")
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_absolute__layer_avg.png"))


def _plot_baskets_relative(
    analysis: Dict[str, Any], out_dir: str, title_prefix: str
) -> None:
    data = analysis.get("baskets_relative")
    assert data is not None
    means = np.array(data["per_layer_bin_means"], dtype=float)  # [L, B]
    counts = np.array(data["counts"], dtype=float)
    assert means.shape == counts.shape
    L, B = means.shape
    x = (np.arange(B) + 0.5) * (100.0 / B)

    # Heatmap
    plt.figure(figsize=(max(6, B * 0.2), max(4, L * 0.4)))
    plt.imshow(means, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Incoming attention (mean)")
    plt.xlabel("Relative position (% of sequence)")
    plt.ylabel("Layer index")
    plt.title(f"{title_prefix} Relative baskets: layer x bin")
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_relative__heatmap.png"))

    # Per-layer curves
    plt.figure(figsize=(8, 5))
    for l in range(L):
        plt.plot(x, means[l], alpha=0.75, label=f"L{l}")
    plt.xlabel("Relative position (% of sequence)")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Relative baskets: per-layer curves")
    if L <= 16:
        plt.legend(ncol=2, fontsize=8)
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_relative__curves.png"))

    # Layer-average curve
    valid = counts.sum(axis=0) > 0
    avg_curve = np.where(
        valid, np.nanmean(np.where(valid, means, np.nan), axis=0), np.nan
    )
    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_curve, color="black")
    plt.xlabel("Relative position (% of sequence)")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Relative baskets: layer-avg curve")
    _savefig(os.path.join(out_dir, f"{title_prefix}__baskets_relative__layer_avg.png"))


def _plot_articles(analysis: Dict[str, Any], out_dir: str, title_prefix: str) -> None:
    data = analysis.get("articles")
    assert data is not None
    means = np.array(data["per_layer_article_means"], dtype=float)  # [L, A]
    counts = np.array(data["counts"], dtype=float)
    assert means.shape == counts.shape
    L, A = means.shape

    # Heatmap
    plt.figure(figsize=(max(5, A * 0.8), max(4, L * 0.4)))
    plt.imshow(means, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Incoming attention (mean)")
    plt.xlabel("Article index in concatenation")
    plt.ylabel("Layer index")
    plt.title(f"{title_prefix} Articles: layer x index")
    _savefig(os.path.join(out_dir, f"{title_prefix}__articles__heatmap.png"))

    # Per-layer curves
    x = np.arange(A)
    plt.figure(figsize=(8, 5))
    for l in range(L):
        plt.plot(x, means[l], alpha=0.75, label=f"L{l}")
    plt.xlabel("Article index in concatenation")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Articles: per-layer curves")
    if L <= 16:
        plt.legend(ncol=2, fontsize=8)
    _savefig(os.path.join(out_dir, f"{title_prefix}__articles__curves.png"))

    # Layer-average curve
    valid = counts.sum(axis=0) > 0
    avg_curve = np.where(
        valid, np.nanmean(np.where(valid, means, np.nan), axis=0), np.nan
    )
    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_curve, color="black")
    plt.xlabel("Article index in concatenation")
    plt.ylabel("Incoming attention (mean)")
    plt.title(f"{title_prefix} Articles: layer-avg curve")
    _savefig(os.path.join(out_dir, f"{title_prefix}__articles__layer_avg.png"))


def _plot_maps(analysis: Dict[str, Any], out_dir: str, title_prefix: str) -> None:
    # Absolute map (may be None)
    m_abs = analysis.get("maps_absolute")
    if m_abs is not None and m_abs.get("per_bin_pair_means") is not None:
        mat = np.array(m_abs["per_bin_pair_means"], dtype=float)
        assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
        B = mat.shape[0]
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="viridis",
            aspect="equal",
            norm=_robust_log_norm(mat),
        )
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target absolute bin")
        plt.ylabel("Source absolute bin")
        plt.title(f"{title_prefix} Absolute attention map (bin x bin)")
        _savefig(os.path.join(out_dir, f"{title_prefix}__map_absolute_heatmap.png"))

        # Zoomed linear view (detail in low-to-mid values)
        vmin, vmax = _robust_linear_range(mat)
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target absolute bin")
        plt.ylabel("Source absolute bin")
        plt.title(f"{title_prefix} Absolute attention map (zoomed)")
        _savefig(
            os.path.join(out_dir, f"{title_prefix}__map_absolute_heatmap_zoomed.png")
        )

    # Relative map (may be None)
    m_rel = analysis.get("maps_relative")
    if m_rel is not None and m_rel.get("per_bin_pair_means") is not None:
        mat = np.array(m_rel["per_bin_pair_means"], dtype=float)
        assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
        B = mat.shape[0]
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        extent = [0, 100, 100, 0]  # x: target%, y: source% (top origin correction)
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="viridis",
            aspect="equal",
            extent=extent,
            norm=_robust_log_norm(mat),
        )
        ticks = np.arange(0, 101, 5)
        plt.xticks(ticks, [f"{t}" for t in ticks])
        plt.yticks(ticks, [f"{t}" for t in ticks])
        plt.xlim(0, 100)
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target relative bin (%)")
        plt.ylabel("Source relative bin (%)")
        plt.title(f"{title_prefix} Relative attention map (bin x bin)")
        _savefig(os.path.join(out_dir, f"{title_prefix}__map_relative_heatmap.png"))

        # Zoomed linear view (detail in low-to-mid values)
        vmin, vmax = _robust_linear_range(mat)
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="viridis",
            aspect="equal",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ticks = np.arange(0, 101, 5)
        plt.xticks(ticks, [f"{t}" for t in ticks])
        plt.yticks(ticks, [f"{t}" for t in ticks])
        plt.xlim(0, 100)
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target relative bin (%)")
        plt.ylabel("Source relative bin (%)")
        plt.title(f"{title_prefix} Relative attention map (zoomed)")
        _savefig(
            os.path.join(out_dir, f"{title_prefix}__map_relative_heatmap_zoomed.png")
        )


def plot_results_file(
    results_path: str, plots_out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Load a results JSON and generate plots to disk.

    Returns a small dict with where plots were written.
    """
    with open(results_path, "r") as f:
        payload = json.load(f)
    assert "analysis" in payload and "meta" in payload
    analysis = payload["analysis"]
    analysis_mode = analysis["analysis_mode"]

    if plots_out_dir is None:
        plots_out_dir = os.path.dirname(results_path)
    _ensure_dir(plots_out_dir)

    # Use the JSON file base name without extension to prefix output plots
    title_prefix = os.path.splitext(os.path.basename(results_path))[0]

    if analysis_mode == "baskets":
        _plot_baskets_absolute(analysis, plots_out_dir, title_prefix)
        _plot_baskets_relative(analysis, plots_out_dir, title_prefix)
        _plot_maps(analysis, plots_out_dir, title_prefix)
        _plot_special_tokens(analysis, plots_out_dir, title_prefix)
    elif analysis_mode == "articles":
        _plot_articles(analysis, plots_out_dir, title_prefix)
        _plot_special_tokens(analysis, plots_out_dir, title_prefix)
    else:
        raise AssertionError(f"Unknown analysis_mode: {analysis_mode}")

    return {"plots_dir": plots_out_dir, "mode": analysis_mode, "results": results_path}


def plot_layer_avg_comparison(
    results_paths: Sequence[str],
    labels: Sequence[str],
    plots_out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare layer-average basket curves across multiple results files.

    Inputs:
    - results_paths: list of JSON result files (each with analysis_mode == "baskets").
    - labels: list of legend labels, same length/order as results_paths.
    - plots_out_dir: directory to save the comparison figures; defaults to the
      directory of the first results file.

    Output:
    - dict with the directory used and saved filenames.
    """
    assert len(results_paths) > 0
    assert len(results_paths) == len(labels)

    # Load and validate all payloads
    payloads = []
    for p in results_paths:
        with open(p, "r") as f:
            payloads.append(json.load(f))

    # Strict structure checks
    for payload in payloads:
        assert "analysis" in payload and "meta" in payload
        analysis = payload["analysis"]
        assert analysis.get("analysis_mode") == "baskets"
        assert "baskets_absolute" in analysis and "baskets_relative" in analysis

    if plots_out_dir is None:
        plots_out_dir = os.path.dirname(results_paths[0])
    _ensure_dir(plots_out_dir)

    # Determine naming from meta/analysis and assert consistency across inputs
    base_meta = payloads[0]["meta"]
    base_analysis = payloads[0]["analysis"]
    src_lang = base_meta["source_lang"]
    tgt_lang = base_meta.get("target_lang")
    only_from_first = bool(base_analysis["only_from_first_token"])
    assert isinstance(src_lang, str) and len(src_lang) == 2
    assert (tgt_lang is None) or (isinstance(tgt_lang, str) and len(tgt_lang) == 2)
    for payload in payloads[1:]:
        m = payload["meta"]
        a = payload["analysis"]
        assert m["source_lang"] == src_lang
        assert m.get("target_lang") == tgt_lang
        assert bool(a["only_from_first_token"]) == only_from_first

    prefix = "avg_fromCLS" if only_from_first else "avg_fromAll"
    lang_suffix = src_lang if tgt_lang is None else f"{src_lang}_{tgt_lang}"

    # Collect absolute and relative layer-avg curves, asserting consistent binning for relative only
    abs_curves = []
    rel_curves = []

    rel_B = None

    for payload in payloads:
        analysis = payload["analysis"]

        # Absolute
        abs_means = np.array(
            analysis["baskets_absolute"]["per_layer_bin_means"], dtype=float
        )  # [L, B_abs]
        abs_counts = np.array(analysis["baskets_absolute"]["counts"], dtype=float)
        assert abs_means.shape == abs_counts.shape
        L_abs, B_abs = abs_means.shape
        assert L_abs > 0 and B_abs > 0
        # layer-average across layers; ignore bins with zero counts in all layers
        abs_valid = abs_counts.sum(axis=0) > 0  # [B_abs]
        abs_curve = np.where(
            abs_valid,
            np.nanmean(np.where(abs_valid, abs_means, np.nan), axis=0),
            np.nan,
        )  # [B_abs]
        # Trim trailing NaNs so each series ends at its last available bin
        valid_idx = np.where(~np.isnan(abs_curve))[0]
        assert valid_idx.size > 0
        end = int(valid_idx[-1]) + 1
        abs_curves.append(abs_curve[:end])

        # Relative
        rel_means = np.array(
            analysis["baskets_relative"]["per_layer_bin_means"], dtype=float
        )  # [L, B_rel]
        rel_counts = np.array(analysis["baskets_relative"]["counts"], dtype=float)
        assert rel_means.shape == rel_counts.shape
        L_rel, B_rel = rel_means.shape
        assert L_rel > 0 and B_rel > 0
        rel_valid = rel_counts.sum(axis=0) > 0
        rel_curve = np.where(
            rel_valid,
            np.nanmean(np.where(rel_valid, rel_means, np.nan), axis=0),
            np.nan,
        )  # [B_rel]
        rel_curves.append(rel_curve)
        rel_B = B_rel if rel_B is None else rel_B
        assert rel_B == B_rel  # require same relative binning across inputs

    # Plot absolute comparison
    plt.figure(figsize=(8, 4.5))
    max_abs_len = 0
    for curve, label in zip(abs_curves, labels):
        assert curve.ndim == 1 and curve.size > 0
        x_abs = np.arange(curve.size)
        max_abs_len = max(max_abs_len, curve.size)
        plt.plot(x_abs, curve, marker=None, linewidth=1.8, label=label)
    plt.xlabel("Absolute bin index")
    plt.ylabel("Incoming attention (layer-avg)")
    plt.title("Absolute baskets: layer-avg comparison")
    plt.legend()
    ax = plt.gca()
    if max_abs_len > 0:
        ax.set_xlim(0, max_abs_len - 1)
    # Ticks: small tick at every bin (minor), larger ticks every 5 bins (major), label only majors
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=3)
    abs_out = os.path.join(plots_out_dir, f"{prefix}__absolute__{lang_suffix}.png")
    _savefig(abs_out)

    # Plot relative comparison
    x_rel = (np.arange(rel_B) + 0.5) * (100.0 / rel_B)
    plt.figure(figsize=(8, 4.5))
    for curve, label in zip(rel_curves, labels):
        assert curve.shape == (rel_B,)
        plt.plot(x_rel, curve, marker=None, linewidth=1.8, label=label)
    # Show ticks at 0, 5, 10, ..., 100 (percent of sequence)
    tick_vals = np.arange(0, 101, 5)
    plt.xticks(tick_vals, [f"{t}" for t in tick_vals])
    plt.xlim(0, 100)
    plt.xlabel("Relative position (% of sequence)")
    plt.ylabel("Incoming attention (layer-avg)")
    plt.title("Relative baskets: layer-avg comparison")
    plt.legend()
    rel_out = os.path.join(plots_out_dir, f"{prefix}__relative__{lang_suffix}.png")
    _savefig(rel_out)

    return {
        "plots_dir": plots_out_dir,
        "absolute_plot": abs_out,
        "relative_plot": rel_out,
    }


def plot_attention_maps(
    results_path: str, plots_out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Plot absolute and relative attention maps (baskets) from a results JSON.

    Saves two figures when data is available:
    - attnMap__{experiment_number}__absolute__{lang_suffix}.png
    - attnMap__{experiment_number}__relative__{lang_suffix}.png
    """
    with open(results_path, "r") as f:
        payload = json.load(f)
    assert "analysis" in payload and "meta" in payload
    analysis = payload["analysis"]
    meta = payload["meta"]

    assert analysis.get("analysis_mode") == "baskets"

    # Extract experiment_number from indices_path (…/indices_wiki_parallel_<N>_…)
    idx_path = meta["indices_path"]
    m = re.search(r"wiki_parallel_(\d+)", idx_path)
    assert (
        m is not None
    ), f"Cannot extract experiment_number from indices_path: {idx_path}"
    experiment_number = m.group(1)

    # Language suffix like in layer-avg comparison
    src_lang = meta["source_lang"]
    tgt_lang = meta.get("target_lang")
    assert isinstance(src_lang, str) and len(src_lang) == 2
    assert (tgt_lang is None) or (isinstance(tgt_lang, str) and len(tgt_lang) == 2)
    lang_suffix = src_lang if tgt_lang is None else f"{src_lang}_{tgt_lang}"

    if plots_out_dir is None:
        plots_out_dir = os.path.dirname(results_path)
    _ensure_dir(plots_out_dir)

    out_abs = None
    out_rel = None

    # Absolute map
    m_abs = analysis.get("maps_absolute")
    if m_abs is not None and m_abs.get("per_bin_pair_means") is not None:
        mat = np.array(m_abs["per_bin_pair_means"], dtype=float)
        assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
        B = mat.shape[0]
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="magma",
            aspect="equal",
            norm=_robust_log_norm(mat),
        )
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target absolute bin")
        plt.ylabel("Source absolute bin")
        plt.title("Absolute attention map (bin x bin)")
        out_abs = os.path.join(
            plots_out_dir, f"attnMap__{experiment_number}__absolute__{lang_suffix}.png"
        )
        _savefig(out_abs)

        # Zoomed linear view
        vmin, vmax = _robust_linear_range(mat)
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="magma",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target absolute bin")
        plt.ylabel("Source absolute bin")
        plt.title("Absolute attention map (zoomed)")
        out_abs_zoom = os.path.join(
            plots_out_dir,
            f"attnMap__{experiment_number}__absolute__{lang_suffix}__zoom.png",
        )
        _savefig(out_abs_zoom)

    # Relative map
    m_rel = analysis.get("maps_relative")
    if m_rel is not None and m_rel.get("per_bin_pair_means") is not None:
        mat = np.array(m_rel["per_bin_pair_means"], dtype=float)
        assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
        B = mat.shape[0]
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="magma",
            aspect="equal",
            extent=[0, 100, 100, 0],
            norm=_robust_log_norm(mat),
        )
        ticks = np.arange(0, 101, 5)
        plt.xticks(ticks, [f"{t}" for t in ticks])
        plt.yticks(ticks, [f"{t}" for t in ticks])
        plt.xlim(0, 100)
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target relative bin (%)")
        plt.ylabel("Source relative bin (%)")
        plt.title("Relative attention map (bin x bin)")
        out_rel = os.path.join(
            plots_out_dir, f"attnMap__{experiment_number}__relative__{lang_suffix}.png"
        )
        _savefig(out_rel)

        # Zoomed linear view
        vmin, vmax = _robust_linear_range(mat)
        plt.figure(figsize=(max(5, B * 0.5), max(4, B * 0.5)))
        plt.imshow(
            mat,
            interpolation="nearest",
            cmap="magma",
            aspect="equal",
            extent=[0, 100, 100, 0],
            vmin=vmin,
            vmax=vmax,
        )
        ticks = np.arange(0, 101, 5)
        plt.xticks(ticks, [f"{t}" for t in ticks])
        plt.yticks(ticks, [f"{t}" for t in ticks])
        plt.xlim(0, 100)
        plt.colorbar(label="Avg attention (heads+layers)")
        plt.xlabel("Target relative bin (%)")
        plt.ylabel("Source relative bin (%)")
        plt.title("Relative attention map (zoomed)")
        out_rel_zoom = os.path.join(
            plots_out_dir,
            f"attnMap__{experiment_number}__relative__{lang_suffix}__zoom.png",
        )
        _savefig(out_rel_zoom)

    return {
        "plots_dir": plots_out_dir,
        "absolute_plot": out_abs,
        "relative_plot": out_rel,
        # Optional zoomed outputs (may be undefined if maps missing)
        **(
            {"absolute_plot_zoomed": out_abs_zoom} if "out_abs_zoom" in locals() else {}
        ),
        **(
            {"relative_plot_zoomed": out_rel_zoom} if "out_rel_zoom" in locals() else {}
        ),
    }
