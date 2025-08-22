#!/bin/bash

# Wiki Parallel Processing Pipeline
# This script runs all the steps to process Wikipedia parallel data

set -e  # Exit on any error

echo "Starting..."

echo "1; only first token"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_1_en.json --analysis_mode baskets --basket_size 128 --only_from_first_token --batch_size 1

echo "1; all tokens"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_1_en.json --analysis_mode baskets --basket_size 128 --batch_size 1

echo "2; only first token"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_2_en.json --analysis_mode baskets --basket_size 128 --only_from_first_token --batch_size 1

echo "2; all tokens"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_2_en.json --analysis_mode baskets --basket_size 128 --batch_size 1

echo "3; only first token"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_3_en.json --analysis_mode baskets --basket_size 128 --only_from_first_token --batch_size 1

echo "3; all tokens"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_3_en.json --analysis_mode baskets --basket_size 128 --batch_size 1

echo "4; only first token"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_4_en.json --analysis_mode baskets --basket_size 128 --only_from_first_token --batch_size 1

echo "4; all tokens"
poetry run python -m locobench.attention.attention_analyzer --config config/Alibaba_mGTE/wiki_parallel/embedding_config_wiki_parallel_4_en.json --analysis_mode baskets --basket_size 128 --batch_size 1