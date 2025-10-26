# LoCoBench: Workbench and Benchmarking Framework for Long-Context Text Embedding Models

This README documents the current workflow to: build a multilingual Wikipedia dataset (comparable corpus), tokenize it, create parallel (aligned) indices, compute (calibrated) embeddings for Experiment 1 (Segment Representation) and Experiment 2 (Information Retention), and run Experiment 3 (Self-Attention Analysis), plus quantitative/qualitative analysis and utilities. 

## Table of Contents

- Overview and Setup
  - Environment and quickstart
  - End-to-end pipeline (high level)
- Data preparation
  - Build the multilingual Wikipedia dataset (wrapper or single steps)
  - Tokenize the dataset
  - Create parallel indices
- Embedding experiments (Exp1 & Exp2)
  - Compute embeddings (monolingual and mixed-language documents)
  - What this produces for the experiments
  - Attention calibration (optional)
- Attention analysis (Exp3)
  - Run the analyzer
  - Visualize attention results
- Results analysis
  - Quantitative (numerical)
  - Qualitative (plots)
- Automation & utilities
  - Generate many configs and run all
  - Adding new models/tokenizers

---

## Overview and Setup

### Environment and quickstart

To set up the Python environment with Poetry:

```bash
git clone <repository>
cd <repository>
poetry install
```

All commands below assume you run them with Poetry:

```bash
poetry run python <script> --config <path-to-config.json>
```

### End-to-end pipeline (high level)

1) Build the multilingual Wikipedia dataset (wrapper script available; single steps also listed)
2) Tokenize the dataset for a specific model
3) Create parallel indices (concat and standalone) with length constraints
4) Compute embeddings using the indices (monolingual or mixed-language documents)

Convenience: generate many embedding configs at once with `create_wiki_parallel_configs.sh`, then run them all with `run_all_configs.sh`.

---

## Data preparation

### Build the multilingual Wikipedia dataset

We rely on the title matching tool from https://github.com/clab/wikipedia-parallel-titles. You can either:

- One-shot wrapper: run all steps with `run_wiki_parallel_steps.sh` (recommended), or
- Execute the single steps manually (retained below for transparency and customization).

The wrapper produces a Hugging Face Dataset on disk with one split per language. The folder name is user-defined; throughout this README we refer to it as `data/wiki_parallel_en_de_hi_it_ko_zh/` as an example. If you use a different name, adjust the paths in later configs accordingly.

#### One-shot wrapper

```bash
./run_wiki_parallel_steps.sh
```

This runs the following in sequence (versions and languages can be changed in the script):
- Download parallel titles for the chosen dump version and languages
- Process titles per language
- Create parallel title matches and convert to page IDs
- Download articles (by ID and by title match)
- Unify the downloaded datasets into a single parallel set

The end result should be a HF dataset on disk with language splits, e.g.:

```
data/wiki_parallel_en_de_hi_it_ko_zh/
  dataset_dict.json
  en/ data-*.arrow ...
  de/ data-*.arrow ...
  hi/ ...
  it/ ...
  ko/ ...
  zh/ ...
```

#### Single steps (for reference/customization)

These are the same steps embedded in the wrapper script:

1. Download titles for selected languages and dump version
2. Process titles per language
3. Create parallel title matches
4. Convert titles to page IDs
5. Download articles via IDs and via title matching
6. Unify the datasets into one parallel set

If you prefer running single steps, see `run_wiki_parallel_steps.sh` for the exact commands and adjust languages or dump version as needed.

### Tokenize the dataset

Use `src/locobench/scripts/tokenize_dataset.py`. For a HF dataset saved on disk, set `dataset_format` to `"arrow"` and point `dataset_path` to the dataset folder.

Command:

```bash
poetry run python src/locobench/scripts/tokenize_dataset.py --config config/tokenization_config_wiki_parallel.json
```

Config (example for Alibaba-NLP/gte-multilingual-base):

```json
{
  "dataset_path": "data/wiki_parallel_en_de_hi_it_ko_zh",
  "dataset_format": "arrow",
  "model_name": "Alibaba-NLP/gte-multilingual-base",
  "id_column": "id",
  "text_column": "text"
}
```

Output:
- A folder at `data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base/` with per-language splits and a `tokenization_config.json` capturing the effective settings and output paths (including `metadata.json` per split).

Notes:
- If `output_path` is omitted, it is derived automatically as `dataset_path/tokenized__<model_name>` where `/` in the model is replaced by `_`.
- For non-"arrow" formats (jsonl, csv, txt), see inline help in `tokenize_dataset.py`.

### Create parallel indices

Use `src/locobench/scripts/create_parallel_indices.py` to pre-compute:
- `concat_indices`: indices of segments to concatenate (permutation-based), and
- `standalone_indices`: the set of used segment indices.

Why this step?
- It ensures the same document indices are used across runs and models.
- You can constrain segment lengths in the source language and enforce similarity to target-language segment lengths.
- The script adjusts `sample_size` up to a multiple of `concat_size!` to balance permutations.

Command:

```bash
poetry run python src/locobench/scripts/create_parallel_indices.py --config config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.config.json
```

Config (example):

```json
{
  "dataset_dir": "data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base",
  "source_lang": "en",
  "target_langs": ["de", "hi", "it", "ko", "zh"],
  "concat_size": 3,
  "sample_size": 1000,
  "max_total_length": 8192,
  "source_segment_range": [1000, 2000],
  "source_target_segment_diffs": [0.7, 0.7, 0.7, 0.7, 0.7],
  "output_path": "config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.json"
}
```

Output:
- A single JSON file (path `output_path`) containing `concat_indices`, `standalone_indices`, the `generation_config`, and summary stats.

Defaults:
- If `output_path` is omitted, a name is auto-derived next to `dataset_dir` using languages and sizes.

---

## Embedding experiments (Exp1 & Exp2)

### Compute embeddings (monolingual and mixed-language documents)

Use `src/locobench/scripts/compute_embeddings.py`. You can run either monolingual (only `source_lang`) or mixed-language documents by including `target_lang` and pointing to the indices file from step 3.

Mixed-language document convention:
- `target_lang` is the language of the first segment in each concatenated document.
- `source_lang` is the language of all remaining segments (2...n) in that document.

Command (example):

```bash
poetry run python src/locobench/scripts/compute_embeddings.py --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_1_de_en.json
```

Config (mixed-language example):

```json
{
  "model_name": "Alibaba-NLP/gte-multilingual-base",
  "tokenized_dataset_path": "data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base",
  "indices_path": "config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.json",
  "mode": "wiki_parallel",
  "source_lang": "de",
  "target_lang": "en",
  "separator": " ",
  "device": "cuda",
  "batch_size_standalone": 32,
  "batch_size_concat": 16,
  "embeddings_output_dir": "results/wiki_parallel"
}
```

Config (monolingual example – simply omit `target_lang`):

```json
{
  "model_name": "Alibaba-NLP/gte-multilingual-base",
  "tokenized_dataset_path": "data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base",
  "indices_path": "config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.json",
  "mode": "wiki_parallel",
  "source_lang": "de",
  "separator": " ",
  "device": "cuda",
  "batch_size_standalone": 32,
  "batch_size_concat": 16,
  "embeddings_output_dir": "results/wiki_parallel"
}
```

Output:
- Embeddings (standalone mean/cls, and late-chunking segments/mean/cls) saved under a descriptive run directory in `results/wiki_parallel/`, plus `embedding_config.json` for reproducibility.
- Run names include model, dataset, mode, languages, and concat-size (and range suffixes when present).

Reproducibility tips:
- When using `indices_path`, concatenation indices are fixed from the file.
- Legacy runs can also reuse indices via `reference_config_path`, but the indices-file workflow above is the recommended path.

### What this produces for the experiments

Following data preparation and embedding computation yields the embeddings needed for:
- Experiment 1: Segment Representation
- Experiment 2: Information Retention

If you evaluate multiple models or language pairs, keep indices fixed and only swap `model_name` and `tokenized_dataset_path` to ensure comparability.

### Attention calibration

Attention calibration re-weights attention scores at inference time, with the goal of distributing representational capacity more evenly across the full sequence. The same config files still work with `run_all_configs.sh` because calibration options are handled by `compute_embeddings.py`.

Keys (top-level apply to both embedders; you can also use per-embedder overrides):
- `apply_attn_calibration` (bool): enable/disable calibration.
- `calib_layers` (string): which layers contribute, e.g., `"last_half"`, `"last"`, `"all"`.
- `calib_source_tokens` (string): token sources, e.g., `"cls"` or `"all"`.
- `calib_basket_size` (int): number of sequences used to build the calibration statistics.

Per-embedder overrides (optional):
- `standalone_apply_attn_calibration`, `standalone_calib_layers`, `standalone_calib_source_tokens`, `standalone_calib_basket_size`
- `latechunk_apply_attn_calibration`, `latechunk_calib_layers`, `latechunk_calib_source_tokens`, `latechunk_calib_basket_size`

Command (example):

```bash
poetry run python src/locobench/scripts/compute_embeddings.py --config config/Alibaba_mGTE/wiki_parallel/clb_LH_CLS_128_embedding_config_wiki_parallel_1_de_en.json
```

Config (mixed-language example; single set of top-level settings applies to both embedders):

```json
{
  "model_name": "Alibaba-NLP/gte-multilingual-base",
  "tokenized_dataset_path": "data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base",
  "indices_path": "config/wiki_parallel/indices_wiki_parallel_1_en_de_hi_it_ko_zh.json",
  "mode": "wiki_parallel",
  "source_lang": "de",
  "target_lang": "en",
  "separator": " ",
  "device": "cuda",
  "batch_size_standalone": 1,
  "batch_size_concat": 1,
  "embeddings_output_dir": "results/attn_clb",
  "apply_attn_calibration": true,
  "calib_layers": "last_half",
  "calib_source_tokens": "cls",
  "calib_basket_size": 128
}
```

How it’s handled:
- `compute_embeddings.py` reads these keys and instantiates both embedders with the effective calibration settings. Per-embedder keys (if provided) override the top-level ones.
- The run directory name is suffixed with a compact tag, e.g., `__LH_CLS_128` when both embedders share the same calibration, or `__saLH_CLS_128__lcA_ATK_64` when they differ.
- The saved `embedding_config.json` in the run dir records `calibration_effective` and `calibration_short_counts` (number of sequences too short for the calibration basket), aiding reproducibility.

Running:
- Place calibrated configs in a folder and execute:

```bash
./run_all_configs.sh <config_folder>
```

or run a single config directly via:

```bash
poetry run python src/locobench/scripts/compute_embeddings.py --config <path-to-config.json>
```

Notes:
- Calibration increases compute and memory; you may want to reduce `batch_size_*` when enabling it.
- For mixed-language documents, the same language convention applies as above (`target_lang` is segment 1; `source_lang` are segments 2...n).

---

## Attention analysis (Exp3)

### Run the analyzer

For Experiment 3 we analyze how attention mass distributes across long concatenated documents. We reuse the same dataset+indices pipeline as above to build a concatenated dataloader and then run the encoder with `output_attentions=True`, aggregating attention into fixed-size token baskets.

Use the pre-wired script:

```bash
./run_attn_analyzer.sh [--sizes "128 64"|--sizes=128,64|-s "128"] <ll> [ll]
```

Where:
- `<ll> [ll]` are one or two ISO-639-1 language codes (e.g., `en` or `en de`).
- `--sizes`/`-s` specifies basket sizes (space- or comma-separated). Default: 128.
- You can also set `BASKET_SIZES` env var (space/comma-separated), e.g., `BASKET_SIZES="128 64" ./run_attn_analyzer.sh en`.

Language convention for mixed-language documents:
- `target_lang` is the language of the first segment in each concatenated document.
- `source_lang` is the language of all remaining segments (2...n).

The script will iterate `i = 1, 2, 3` and for each basket size and language tuple construct the config path:

```
config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_${i}_${LANG_SUFFIX}.json
```

where `LANG_SUFFIX` is `<src>` for monolingual (e.g., `en`) or `<src>_<tgt>` for mixed-language (e.g., `en_de`). Each of these configs is a standard embedding config (see above) that must include at least:
- `model_name`
- `tokenized_dataset_path`
- `indices_path`
- `mode` ("wiki_parallel")
- `source_lang` and optionally `target_lang`
- `separator` (optional)
- `batch_size_concat` (optional)
- `device` (optional)

Example invocations:

```bash
# Monolingual English, default basket size 128
./run_attn_analyzer.sh en

# Mixed-language: first segment in German, remaining in English, basket sizes 256 and 128
./run_attn_analyzer.sh --sizes 256,128 de en

# Monolingual with sizes from environment
BASKET_SIZES="128 64" ./run_attn_analyzer.sh en
```

What it computes (by default):
- Analysis mode: baskets (relative bins default to 20 inside the analyzer)
- Target positions exclude first and last tokens
- Contributions counted only from the first token (CLS/BOS) as source

Outputs:
- JSON files under `results/_attention/<model_tag>__<indices_name>__<source_lang>__<target_lang>/` with filenames like:

```
attn__baskets_bs{basket_size}_rb{rel_bins}__exclude-first__exclude-last__from-first-only.json
```

Direct analyzer usage (optional):
If you need finer control (e.g., relative bins, article-level aggregation, include/exclude tokens, device), you can call the analyzer directly:

```bash
poetry run python -m locobench.attention.attention_analyzer \
  --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_1_de_en.json \
  --analysis_mode baskets \
  --basket_size 128 \
  --batch_size 1 \
  --only_from_first_token
```

This reads the same config format used for embeddings and produces outputs in `results/_attention/`.

### Visualize attention results

Use the helpers in `src/locobench/visualizations/attention_analyzer_viz.py` to visualize the JSON produced by the analyzer. A common workflow is:

```python
import glob
from locobench.visualizations.attention_analyzer_viz import (
  plot_results_file,
  plot_cls_attention_mass_cohorts,
)

# Parameters to select the right result file
LANG1 = "zh"      # language of first segment (mixed-language) or the single language (monolingual)
LANG2 = "hi"      # language of remaining segments (mixed-language); set to None/omit for monolingual
CONCAT_SIZE = 5    # number of segments per concatenated document
BASKET_SIZE = 128  # basket size used during analysis

# Find the matching JSON produced by the analyzer under your results folder
pattern = (
  f"results/_attention/*__indices_wiki_parallel_{CONCAT_SIZE-2}_*__{LANG1}__{LANG2}/"
  f"attn__baskets_bs{BASKET_SIZE}_rb*__exclude-first__exclude-last__from-first-only.json"
)
matches = sorted(glob.glob(pattern))
assert len(matches) > 0, "No matching attention results found; double-check languages, concat size, and basket size."
RESULT_PATH = matches[0]

# Option A: Generate a suite of plots for the results JSON
plot_results_file(RESULT_PATH)

# Option B: Focus on CLS attention mass cohorts (layer groups)
PLOT_OUT_DIR = None  # or set a custom directory
plot_cls_attention_mass_cohorts(
  RESULT_PATH,
  plots_out_dir=PLOT_OUT_DIR,
  title="Attention Mass Distribution of <s> Token",
  show_basket_ranges=True,
  avg_ylim=0.6,
)
```

Notes:
- Adjust LANG1/LANG2 according to your mixed-language convention (LANG1 = first segment; LANG2 = segments 2...n). For monolingual analyses, search for paths ending with `__{LANG1}__None` or simply glob for `__{LANG1}__*` and manually pick the monolingual directory.
- You can also use `plot_layer_avg_comparison([...], labels=[...])` and `plot_attention_maps(RESULT_PATH)` from the same module for additional visual comparisons.

---

## Results analysis

### Quantitative (numerical)

After generating embeddings and/or running attention analyses, you can compute numerical metrics without plotting using helpers in `src/locobench/analysis/numerical_analysis.py`.

Step 1: Gather result paths for a model using the helper `categorize_paths_by_root`.

```python
from locobench.analysis.numerical_analysis import categorize_paths_by_root

language_order = ["en", "zh", "de", "it", "ko", "hi"]

# Example for mGTE
PATHS_MGTE_MONO, PATHS_MGTE_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "Alibaba-NLP_gte-multilingual-base__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)

# Example for Jina
PATHS_JINA_MONO, PATHS_JINA_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "jinaai_jina-embeddings-v3__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)

# Example for Qwen3
PATHS_QWEN3_MONO, PATHS_QWEN3_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "Qwen_Qwen3-Embedding-0.6B__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)
```

Step 2: Define pooling strategies per model.

```python
model_pooling_strats = {
  "Alibaba-NLP/gte-multilingual-base": "cls",
  "jinaai/jina-embeddings-v3": "mean",
  "Qwen/Qwen3-Embedding-0.6B": "cls",
}
```

Step 3: Collect position-analysis results.

- Experiment 1 (Segment Representation):

```python
from locobench.analysis.numerical_analysis import collect_multi_model_position_analysis_results

results_exp1 = collect_multi_model_position_analysis_results(
  paths=PATHS_MGTE_MONO,
  model_pooling_strats=model_pooling_strats,
)
```

- Experiment 2 (Information Retention) uses late-chunking segment embeddings. Important: set `document_embedding_type="latechunk-segment"`.

```python
results_exp2 = collect_multi_model_position_analysis_results(
  paths=PATHS_JINA_MONO,
  model_pooling_strats=model_pooling_strats,
  document_embedding_type="latechunk-segment",
)
```

Step 4: Compute quantitative measures.

```python
from locobench.analysis.numerical_analysis import (
  compute_position_statistical_metrics,
  compute_position_diff_metrics,
)

# Statistical tests and indices (ANOVA, OLS betas, Gini with permutation p-value)
quant_metrics = compute_position_statistical_metrics(results_exp1)

# Relative and absolute differences between positions (incl. 95% CIs)
rel_diff_metrics = compute_position_diff_metrics(results_exp1)
```

Notes:
- The result dictionaries are keyed by ((concat_size, lang_key), model_name), where lang_key is `<src>` for monolingual and `<src>_<tgt>` for mixed-language documents.
- For mixed-language, the convention remains: `target_lang` is the first segment; `source_lang` are segments 2...n.

### Qualitative (plots)

For qualitative inspection and figures, use the multi-plotters in `src/locobench/visualizations/multi_plotter.py`. As with numerical analysis, start by collecting result paths via `categorize_paths_by_root` and pick pooling strategies per model.

```python
from locobench.analysis.numerical_analysis import categorize_paths_by_root
from locobench.visualizations.multi_plotter import (
  DocumentLevel2SegmentStandaloneSimPlotter,
  SegmentLatechunk2SegmentStandaloneSimPlotter,
)

language_order = ["en", "zh", "de", "it", "ko", "hi"]

PATHS_MGTE_MONO, PATHS_MGTE_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "Alibaba-NLP_gte-multilingual-base__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)
PATHS_JINA_MONO, PATHS_JINA_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "jinaai_jina-embeddings-v3__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)
PATHS_QWEN3_MONO, PATHS_QWEN3_MULTI = categorize_paths_by_root(
  "results/wiki_parallel",
  "Qwen_QWen3-Embedding-0.6B__wiki_parallel_en_de_hi_it_ko_zh__parallel__",
  language_order=language_order,
)

# Combine across models if desired
PATHS_MONO = PATHS_JINA_MONO + PATHS_MGTE_MONO  # + PATHS_QWEN3_MONO
PATHS_MULTI = PATHS_JINA_MULTI + PATHS_MGTE_MULTI  # + PATHS_QWEN3_MULTI

model_pooling_strats = {
  "Alibaba-NLP/gte-multilingual-base": "cls",
  "jinaai/jina-embeddings-v3": "mean",
  "Qwen/Qwen3-Embedding-0.6B": "cls",
}
```

Experiment 1 (Segment Representation): document-level to segment-standalone similarity curves.

```python
position_plotter = DocumentLevel2SegmentStandaloneSimPlotter()

# Monolingual plots
position_plotter.plot_multi_models(
  paths=PATHS_MONO,
  model_pooling_strats=model_pooling_strats,
  show_lengths=True,
)

# Mixed-language plots (split columns by source_lang)
position_plotter.plot_multi_models(
  paths=PATHS_MULTI,
  model_pooling_strats=model_pooling_strats,
  show_lengths=True,
  split_plots_by_source_lang=True,
)
```

Experiment 2 (Information Retention): segment late-chunk vs segment standalone similarity curves.

```python
latechunk2standalone_plotter = SegmentLatechunk2SegmentStandaloneSimPlotter()

# Jina monolingual
latechunk2standalone_plotter.plot(
  PATHS_JINA_MONO,
  pooling_strategy_segment_standalone="mean",
  show_lengths=True,
)

# Jina mixed-language (split columns by source_lang)
latechunk2standalone_plotter.plot(
  PATHS_JINA_MULTI,
  pooling_strategy_segment_standalone="mean",
  show_lengths=True,
  split_plots_by_source_lang=True,
)
```

Notes:
- For mixed-language documents, the convention is: `target_lang` is the first segment; `source_lang` are segments 2...n.
- You can optionally pass `matryoshka_dimensions=[32, 128]` to visualize reduced-dimensional embeddings alongside full ones.

---

## Automation & utilities

### Generate many configs and run all

Use `src/locobench/scripts/create_wiki_parallel_configs.sh` to generate a whole set of monolingual and mixed-language configs for a given model and experiment number:

```bash
./src/locobench/scripts/create_wiki_parallel_configs.sh <output_folder> <model> <experiment_number> <source_langs_comma> <target_langs_comma> [batch_size_standalone] [batch_size_concat]
```

Examples:

```bash
./src/locobench/scripts/create_wiki_parallel_configs.sh config/MyModel/wiki_parallel mgte 4 en,de en,de,ko
./src/locobench/scripts/create_wiki_parallel_configs.sh config/MyModel/wiki_parallel jina 4 en,de en,de,ko 16 4
./src/locobench/scripts/create_wiki_parallel_configs.sh config/MyModel/wiki_parallel qwen3 4 en,de en,de,ko
```

Notes:
- The script assumes your tokenized dataset lives at `data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__<model_name>`; adjust inside the script if you used a different dataset folder name.
- It sets `indices_path` to `config/wiki_parallel/indices_wiki_parallel_<EXPERIMENT_NUMBER>_en_de_hi_it_ko_zh.json` by default—ensure your indices file from data preparation matches this path or edit accordingly.
- After generating configs, you can run them all with:

```bash
./run_all_configs.sh <output_folder>
```

### Adding new models/tokenizers

If your tokenizer uses special token fields beyond `input_ids`, `attention_mask`, and `token_type_ids`, update `CustomDataCollator` in `custom_data_collator.py` accordingly so batching remains compatible.