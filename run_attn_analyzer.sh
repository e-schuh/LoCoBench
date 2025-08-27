#!/bin/bash

# Wiki Parallel Processing Pipeline
# This script runs all the steps to process Wikipedia parallel data

set -e  # Exit on any error

echo "Starting..."

# Validate and build language suffix (supports 1 or 2 ISO-639-1 codes)
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
	echo "Usage: $0 <ll> [ll]" >&2
	exit 1
fi

for arg in "$@"; do
	case "$arg" in
		[a-z][a-z]) ;;
		*) echo "Error: language codes must be two lowercase letters (e.g., en, de, hi). Got: '$arg'" >&2; exit 1 ;;
	esac
done

if [ "$#" -eq 1 ]; then
	LANG_SUFFIX="$1"
else
	LANG_SUFFIX="$1_$2"
fi

CONFIG_DIR="config/Alibaba_mGTE/wiki_parallel"

for i in 4 3 2 1; do
	CONFIG_PATH="${CONFIG_DIR}/embedding_config_wiki_parallel_${i}_${LANG_SUFFIX}.json"

	# echo "${i}; only first token"
	# poetry run python -m locobench.attention.attention_analyzer --config "${CONFIG_PATH}" --analysis_mode baskets --basket_size 128 --only_from_first_token --batch_size 1

	echo "${i}; all tokens"
	poetry run python -m locobench.attention.attention_analyzer --config "${CONFIG_PATH}" --analysis_mode baskets --basket_size 128 --batch_size 1 --exclude_incoming
done