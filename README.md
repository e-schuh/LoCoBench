# Workbench and Benchmarking Framework for Long Context Text Embedding Models

## Reproducing Environments

To recreate the environment exactly as specified:

```bash
git clone <repository>
cd <repository>
poetry install
```


## Creating Embeddings for a Huggingface Dataset
1. Create a config file in the `config` directory for the dataset download. There, specify the dataset url, the number of samples to be generated, the output directory and other parameters. Then, run the following command:
```bash
poetry run python src/locobench/scripts/download_samples.py --config config/download_samples_config_test.json
```

2. Create a config file in the `config` directory for the tokenization process. There, specify the format of the dataset ("arrow", "txt", "jsonl",...), the path to the dataset (if dataset_format is "arrow", then put the path to the folder with the arrow files; if dataset_format is "csv", "jsonl", "txt", then put the path to the file itself), model name, and other parameters. Then, run the following command:
```bash
poetry run python src/locobench/scripts/tokenize_dataset.py --config config/tokenization_config_test.json
```

3. Create a config file in the `config` directory for the embedding generation. There, specify the path to the tokenized dataset (see the output config file of the tokenization process), the path to the metadata file (see the output config file of the tokenization process), parameters for the concatenation process, etc. Then, run the following command:
```bash
poetry run python src/locobench/scripts/compute_embeddings.py --config config/embedding_config_test.json
```

### Reproducing Embedding Experiments

To ensure exact replication of experiments with the same document indices, you can use the `reference_config_path` parameter in your embedding configuration file. This allows you to reuse the exact same concatenation and standalone indices from a previous run:

```json
{
  "model_name": "your-model/name",
  "tokenized_dataset_path": "path/to/tokenized/dataset",
  "metadata_path": "path/to/metadata.json",
  "concatenation_strategy": "switch",
  "concat_size": 3,
  "sample_size": 500,
  // ... other parameters ...
  "reference_config_path": "results/runs/Previous_Run_Name/embedding_config.json"
}
```

When this parameter is specified, the script will load the `concat_indices` and `standalone_indices` from the reference config file rather than generating new ones. This ensures identical document selection across different runs.


##  Adding New Models / Tokenizers

Ensure that in the file `custom_data_collator.py`, the `CustomDataCollator` class is updated to include any special tokens different from "input_ids", "attention_mask", and "token_type_ids".