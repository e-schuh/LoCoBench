# Workbench and Benchmarking Framework for Long Context Text Embedding Models

## Reproducing Environments

To recreate the environment exactly as specified:

```bash
git clone <repository>
cd <repository>
poetry install
```

## Adding New Models / Tokenizers
Ensure that in the file `custom_data_collator.py`, the `CustomDataCollator` class is updated to include any special tokens different from "input_ids", "attention_mask", and "token_type_ids".