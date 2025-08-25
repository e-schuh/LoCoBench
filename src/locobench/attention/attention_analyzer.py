"""
Attention analysis for long concatenated documents.

This module reuses the existing dataset + indices pipeline to build a
concatenated dataloader, runs the encoder model with output_attentions=True,
and aggregates incoming attention per destination position:

- Basket-level: aggregate per fixed-size token baskets (e.g., 64/128/256)
- Article-level: aggregate per article using doc_boundaries
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModel
from tqdm import tqdm

from locobench.core.document_handler import DocumentHandler


@dataclass
class AnalysisConfig:
    config_path: str
    analysis_mode: str  # "baskets" | "articles"
    basket_size: int = 128
    rel_bins: int = 20
    exclude_first_token: bool = True
    exclude_last_token: bool = True
    max_examples: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    only_from_first_token: bool = False


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_concat_loader_from_config(
    config: Dict[str, Any],
) -> Tuple[torch.utils.data.DataLoader, Dict[str, Any]]:
    model_name = config["model_name"]
    tokenized_dataset_path = config["tokenized_dataset_path"]
    indices_path = config["indices_path"]
    separator = config.get("separator", " ")
    source_lang = config.get("source_lang", "en")
    target_lang = config.get("target_lang")
    batch_size_concat = config.get("batch_size_concat", 1)

    tokenized = load_from_disk(tokenized_dataset_path)
    assert isinstance(tokenized, (Dataset, DatasetDict))

    indices = _load_json(indices_path)
    assert "concat_indices" in indices
    concat_indices: List[List[int]] = indices["concat_indices"]
    assert isinstance(concat_indices, list) and len(concat_indices) > 0

    doc_handler = DocumentHandler(tokenizer_name=model_name)

    if isinstance(tokenized, DatasetDict):
        # wiki_parallel path
        datasets, _, _ = doc_handler.prepare_datasets_wiki_parallel(
            dataset_dict=tokenized,
            concat_indices=concat_indices,
            separator=separator,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        concat_dataset = datasets["concatenated"]
    else:
        # Generic concatenation on a monolingual Dataset
        concat_dataset = doc_handler.create_concatenated_dataset(
            dataset=tokenized,
            concat_indices=concat_indices,
            separator=separator,
        )

    concat_loader = doc_handler.get_dataloader(
        concat_dataset, batch_size=batch_size_concat, shuffle=False
    )
    return concat_loader, {
        "model_name": model_name,
        "separator": separator,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "indices_path": indices_path,
        "dataset_path": tokenized_dataset_path,
    }


class AttentionAggregator:
    def __init__(
        self,
        model_name: str,
        device: Optional[str],
        analysis_mode: str,
        basket_size: int,
        rel_bins: int,
        exclude_first_token: bool,
        exclude_last_token: bool,
        only_from_first_token: bool,
    ) -> None:
        assert analysis_mode in ("baskets", "articles")

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device

        if model_name == "jinaai/jina-embeddings-v3":
            model_name = "jinaai/jina-embeddings-v2-base-en"

        # Prefer lower precision on accelerator to cut memory
        torch_dtype = None
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            torch_dtype = torch.bfloat16
        elif isinstance(self.device, str) and self.device.startswith("mps"):
            torch_dtype = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        self.model_name = model_name

        self.analysis_mode = analysis_mode
        self.basket_size = basket_size
        self.rel_bins = rel_bins
        self.exclude_first_token = exclude_first_token
        self.exclude_last_token = exclude_last_token
        self.only_from_first_token = only_from_first_token

        # Aggregation state
        self.num_layers: Optional[int] = None
        self.num_abs_bins: Optional[int] = None
        self.num_rel_bins: Optional[int] = None
        self.num_articles: Optional[int] = None
        self.seq_len_ref: Optional[int] = None

        # Absolute baskets accumulators [layers, bins]
        self.sum_abs: Optional[torch.Tensor] = None
        self.count_abs: Optional[torch.Tensor] = None
        # Relative baskets accumulators [layers, bins]
        self.sum_rel: Optional[torch.Tensor] = None
        self.count_rel: Optional[torch.Tensor] = None

        # Special token accumulators [layers]
        self.sum_first: Optional[torch.Tensor] = None
        self.count_first: Optional[torch.Tensor] = None
        self.sum_last: Optional[torch.Tensor] = None
        self.count_last: Optional[torch.Tensor] = None

        self.examples_seen = 0

    @torch.no_grad()
    def process_batch(self, batch: Dict[str, Any]) -> None:
        input_ids: torch.Tensor = batch["input_ids"].to(self.device)
        attention_mask: torch.Tensor = batch["attention_mask"].to(self.device)

        assert input_ids.dim() == 2 and attention_mask.shape == input_ids.shape
        bsz, seq_len = input_ids.shape

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        assert isinstance(attentions, (list, tuple)) and len(attentions) > 0

        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        for l in range(num_layers):
            a = attentions[l]
            assert a.shape == (bsz, num_heads, seq_len, seq_len)

        if self.num_layers is None:
            self.num_layers = num_layers
        else:
            assert self.num_layers == num_layers

        # For each example in batch, compute reduced incoming attention vector per layer
        for i in range(bsz):
            mask_i = attention_mask[i]  # [L]
            valid_idx = mask_i.nonzero(as_tuple=False).squeeze(-1)
            assert valid_idx.numel() > 0

            # Compress to only valid tokens (keep order)
            L_eff = valid_idx.numel()
            # No fixed-length requirement; we aggregate with masking (absolute) and percentiles (relative)

            # Build per-layer vector per valid target position
            # Default: incoming[j] = sum over queries of A[query->key=j]
            # If only_from_first_token: use only query at first valid position
            per_layer_vecs: List[torch.Tensor] = []  # each [L_eff]
            for l in range(num_layers):
                a = attentions[l][i]  # [H, L, L]
                if not self.only_from_first_token:
                    # Sum over queries -> keep keys dimension
                    incoming = a.sum(dim=1)  # [H, L]
                    incoming_valid = incoming[:, valid_idx]  # [H, L_eff]
                    vec = incoming_valid.mean(dim=0)  # [L_eff]
                else:
                    # Take attention from first valid query position only
                    q_pos = int(valid_idx[0].item())
                    row = a[:, q_pos, :]  # [H, L]
                    row_valid = row[:, valid_idx]  # [H, L_eff]
                    vec = row_valid.mean(dim=0)  # [L_eff]
                assert vec.shape == (L_eff,)

                # Accumulate first/last token attention per layer (before trimming)
                first_val = float(vec[0].item())
                last_val = float(vec[-1].item())
                if self.sum_first is None:
                    self.sum_first = torch.zeros(num_layers, dtype=torch.float32)
                    self.count_first = torch.zeros(num_layers, dtype=torch.float32)
                    self.sum_last = torch.zeros(num_layers, dtype=torch.float32)
                    self.count_last = torch.zeros(num_layers, dtype=torch.float32)
                self.sum_first[l] += first_val
                self.count_first[l] += 1.0
                self.sum_last[l] += last_val
                self.count_last[l] += 1.0

                per_layer_vecs.append(vec)

            # Optionally exclude first token (e.g., [CLS]/BOS)
            if self.exclude_first_token:
                per_layer_vecs = [v[1:] for v in per_layer_vecs]
                L_eff = per_layer_vecs[0].shape[0]

            # Optionally exclude last token (e.g., trailing SEP/EOS)
            if self.exclude_last_token:
                per_layer_vecs = [v[:-1] for v in per_layer_vecs]
                L_eff = per_layer_vecs[0].shape[0]

            # If exclusions removed all targets, skip main aggregation but keep first/last
            if L_eff == 0:
                self.examples_seen += 1
                continue

            if self.analysis_mode == "baskets":
                self._accumulate_baskets_absolute(per_layer_vecs)
                self._accumulate_baskets_relative(per_layer_vecs)
            else:
                # Article-level using doc_boundaries
                raw_bounds = batch["doc_boundaries"][i]
                # Convert to tensor deterministically; fail fast on unknown types
                if isinstance(raw_bounds, torch.Tensor):
                    doc_bounds = raw_bounds.to(valid_idx.device)
                elif isinstance(raw_bounds, (list, tuple)):
                    doc_bounds = torch.tensor(
                        raw_bounds, dtype=torch.long, device=valid_idx.device
                    )
                else:
                    try:
                        import numpy as _np  # local import to avoid global dependency if unused

                        if isinstance(raw_bounds, _np.ndarray):
                            doc_bounds = torch.from_numpy(raw_bounds).to(
                                valid_idx.device
                            )
                        else:
                            raise AssertionError(
                                f"Unsupported doc_boundaries type: {type(raw_bounds)}"
                            )
                    except Exception:
                        raise AssertionError(
                            f"Unsupported doc_boundaries type: {type(raw_bounds)}"
                        )

                assert (
                    doc_bounds.ndim == 2
                    and doc_bounds.shape[1] == 2
                    and doc_bounds.shape[0] > 0
                )

                # Convert doc_bounds from absolute positions to reduced index space
                # valid_idx maps reduced positions -> absolute indices.
                # We need to map absolute spans to reduced spans by locating their
                # start/end within valid_idx.
                spans: List[Tuple[int, int]] = []
                for s_abs, e_abs in doc_bounds.tolist():
                    # Find positions within valid_idx (assumes contiguous spans)
                    pos_start = (valid_idx == s_abs).nonzero(as_tuple=False)
                    pos_end = (valid_idx == (e_abs - 1)).nonzero(as_tuple=False)
                    assert pos_start.numel() == 1 and pos_end.numel() == 1
                    s_red = int(pos_start.item())
                    e_red = int(pos_end.item()) + 1
                    # If we excluded first token earlier from per_layer_vecs,
                    # shift spans into the trimmed index space and clamp
                    if self.exclude_first_token:
                        s_red = max(0, s_red - 1)
                        e_red = max(0, e_red - 1)
                    # If we excluded last token earlier, clamp end to new length
                    if self.exclude_last_token:
                        e_red = min(e_red, L_eff)
                    spans.append((s_red, e_red))

                if self.num_articles is None:
                    self.num_articles = len(spans)
                else:
                    assert self.num_articles == len(spans)

                self._accumulate_articles(per_layer_vecs, spans)

            self.examples_seen += 1

    def _ensure_abs_buffers(self, L_bins: int) -> None:
        assert self.num_layers is not None
        if self.sum_abs is None:
            self.sum_abs = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
            self.count_abs = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
        elif self.sum_abs.shape[1] < L_bins:
            pad = L_bins - self.sum_abs.shape[1]
            self.sum_abs = torch.cat(
                [self.sum_abs, torch.zeros(self.num_layers, pad, dtype=torch.float32)],
                dim=1,
            )
            self.count_abs = torch.cat(
                [
                    self.count_abs,
                    torch.zeros(self.num_layers, pad, dtype=torch.float32),
                ],
                dim=1,
            )

    def _ensure_rel_buffers(self, L_bins: int) -> None:
        assert self.num_layers is not None
        if self.sum_rel is None:
            self.sum_rel = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
            self.count_rel = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
        else:
            assert self.sum_rel.shape == (self.num_layers, L_bins)
            assert self.count_rel.shape == (self.num_layers, L_bins)

    def _accumulate_baskets_absolute(self, per_layer_vecs: List[torch.Tensor]) -> None:
        # per_layer_vecs: list of [L_eff]
        L_eff = per_layer_vecs[0].shape[0]
        for v in per_layer_vecs:
            assert v.shape == (L_eff,)

        # Determine number of absolute bins for this example
        num_bins = (L_eff + self.basket_size - 1) // self.basket_size
        if self.num_abs_bins is None or num_bins > self.num_abs_bins:
            self.num_abs_bins = num_bins
        self._ensure_abs_buffers(self.num_abs_bins)

        for l, v in enumerate(per_layer_vecs):
            for b in range(num_bins):
                s = b * self.basket_size
                e = min((b + 1) * self.basket_size, L_eff)
                if e <= s:
                    continue
                seg = v[s:e]
                self.sum_abs[l, b] += float(seg.mean())
                self.count_abs[l, b] += 1.0

    def _accumulate_baskets_relative(self, per_layer_vecs: List[torch.Tensor]) -> None:
        # per_layer_vecs: list of [L_eff]
        L_eff = per_layer_vecs[0].shape[0]
        for v in per_layer_vecs:
            assert v.shape == (L_eff,)
        assert L_eff >= self.rel_bins

        if self.num_rel_bins is None:
            self.num_rel_bins = self.rel_bins
        else:
            assert self.num_rel_bins == self.rel_bins
        self._ensure_rel_buffers(self.num_rel_bins)

        edges = torch.linspace(0, L_eff, steps=self.rel_bins + 1)
        edges = torch.floor(edges).to(torch.int64)
        edges[-1] = L_eff
        for k in range(1, edges.numel()):
            if edges[k] <= edges[k - 1]:
                edges[k] = min(edges[k - 1] + 1, L_eff)

        for l, v in enumerate(per_layer_vecs):
            for b in range(self.rel_bins):
                s = int(edges[b].item())
                e = int(edges[b + 1].item())
                if e <= s:
                    continue
                seg = v[s:e]
                self.sum_rel[l, b] += float(seg.mean())
                self.count_rel[l, b] += 1.0

    def _accumulate_articles(
        self, per_layer_vecs: List[torch.Tensor], spans: List[Tuple[int, int]]
    ) -> None:
        num_articles = len(spans)
        if self.num_articles is None:
            self.num_articles = num_articles
        else:
            assert self.num_articles == num_articles

        self._ensure_abs_buffers(num_articles)

        for l, v in enumerate(per_layer_vecs):
            for a_idx, (s, e) in enumerate(spans):
                # Skip empty spans (possible when excluding the first token)
                if not (0 <= s < e <= v.shape[0]):
                    continue
                seg = v[s:e]
                self.sum_abs[l, a_idx] += float(seg.mean())
                self.count_abs[l, a_idx] += 1.0

    def finalize(self) -> Dict[str, Any]:
        assert self.num_layers is not None
        result: Dict[str, Any] = {
            "analysis_mode": self.analysis_mode,
            "exclude_first_token": self.exclude_first_token,
            "exclude_last_token": self.exclude_last_token,
            "only_from_first_token": self.only_from_first_token,
            "num_layers": self.num_layers,
            "examples_seen": self.examples_seen,
        }

        # Attach first/last token aggregates if available
        if self.sum_first is not None and self.sum_last is not None:
            first_means = (self.sum_first / self.count_first.clamp_min(1.0)).tolist()
            last_means = (self.sum_last / self.count_last.clamp_min(1.0)).tolist()
            result.update(
                {
                    "first_token": {
                        "per_layer_means": first_means,
                        "counts": self.count_first.tolist(),
                    },
                    "last_token": {
                        "per_layer_means": last_means,
                        "counts": self.count_last.tolist(),
                    },
                }
            )

        if self.analysis_mode == "articles":
            assert self.sum_abs is not None and self.count_abs is not None
            means_abs = (self.sum_abs / self.count_abs.clamp_min(1.0)).tolist()
            result.update(
                {
                    "articles": {
                        "num_articles": self.sum_abs.shape[1],
                        "per_layer_article_means": means_abs,
                        "counts": self.count_abs.tolist(),
                    }
                }
            )
        else:
            assert self.sum_abs is not None and self.count_abs is not None
            assert self.sum_rel is not None and self.count_rel is not None
            means_abs = (self.sum_abs / self.count_abs.clamp_min(1.0)).tolist()
            means_rel = (self.sum_rel / self.count_rel.clamp_min(1.0)).tolist()
            result.update(
                {
                    "baskets_absolute": {
                        "basket_size": self.basket_size,
                        "num_bins": self.sum_abs.shape[1],
                        "per_layer_bin_means": means_abs,
                        "counts": self.count_abs.tolist(),
                    },
                    "baskets_relative": {
                        "num_bins": self.sum_rel.shape[1],
                        "per_layer_bin_means": means_rel,
                        "counts": self.count_rel.tolist(),
                    },
                }
            )
        return result


def analyze_from_config(args: AnalysisConfig) -> Dict[str, Any]:
    base_config = _load_json(args.config_path)

    # Optional CLI override for concat dataloader batch size
    if args.batch_size is not None:
        assert isinstance(args.batch_size, int) and args.batch_size >= 1
        base_config["batch_size_concat"] = int(args.batch_size)

    concat_loader, meta = build_concat_loader_from_config(base_config)

    device = args.device or base_config.get("device")
    analyzer = AttentionAggregator(
        model_name=meta["model_name"],
        device=device,
        analysis_mode=args.analysis_mode,
        basket_size=args.basket_size,
        rel_bins=args.rel_bins,
        exclude_first_token=args.exclude_first_token,
        exclude_last_token=args.exclude_last_token,
        only_from_first_token=args.only_from_first_token,
    )

    max_examples = args.max_examples
    seen = 0
    for batch in tqdm(concat_loader, desc="Processing batches"):
        analyzer.process_batch(batch)
        seen += batch["input_ids"].shape[0]
        if max_examples is not None and seen >= max_examples:
            break

    result = analyzer.finalize()

    # Compose output path
    out_base = base_config.get("embeddings_output_dir", "results")
    model_tag = base_config["model_name"].replace("/", "_")
    dataset_name = os.path.basename(
        os.path.dirname(base_config["tokenized_dataset_path"])
    )
    mode_tag = (
        f"baskets_bs{args.basket_size}_rb{args.rel_bins}"
        if args.analysis_mode == "baskets"
        else "articles"
    )
    excl_first_tag = "exclude-first" if args.exclude_first_token else "include-first"
    excl_last_tag = (
        "exclude-last" if getattr(args, "exclude_last_token", False) else "include-last"
    )
    src_tag = "from-first-only" if args.only_from_first_token else "from-all"
    source_lang = base_config.get("source_lang", "en")
    target_lang = base_config.get("target_lang", None)
    indices_name = os.path.basename(base_config["indices_path"])
    out_dir = os.path.join(
        out_base,
        "_attention",
        f"{model_tag}__{indices_name}__{source_lang}__{target_lang}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"attn__{mode_tag}__{excl_first_tag}__{excl_last_tag}__{src_tag}.json",
    )

    payload = {
        "meta": meta,
        "analysis": result,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"output_path": out_path, "summary": result}


def _parse_args() -> AnalysisConfig:
    p = argparse.ArgumentParser(description="Attention analyzer for concatenated docs")
    p.add_argument(
        "--config",
        required=True,
        help="Path to config JSON used to build concat dataset",
    )
    p.add_argument(
        "--analysis_mode",
        choices=["baskets", "articles"],
        default="baskets",
        help="Aggregate attention by baskets or by article boundaries",
    )
    p.add_argument(
        "--basket_size",
        type=int,
        default=128,
        help="Basket size in tokens (baskets mode)",
    )
    p.add_argument(
        "--rel_bins",
        type=int,
        default=20,
        help="Number of relative (percentile) bins in baskets mode",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override concat dataloader batch size (default reads from config)",
    )
    p.add_argument(
        "--exclude_first_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the first token (e.g., CLS/BOS) from target positions (default: exclude)",
    )
    p.add_argument(
        "--exclude_last_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the last token (e.g., SEP/EOS) from target positions (default: exclude)",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Process at most this many examples",
    )
    p.add_argument(
        "--device", type=str, default=None, help="Device override: cpu|cuda|mps"
    )
    p.add_argument(
        "--only_from_first_token",
        action="store_true",
        help="If set, compute contributions using only the first token as source (CLS/<s>)",
    )
    a = p.parse_args()
    return AnalysisConfig(
        config_path=a.config,
        analysis_mode=a.analysis_mode,
        basket_size=a.basket_size,
        rel_bins=a.rel_bins,
        exclude_first_token=bool(a.exclude_first_token),
        exclude_last_token=bool(a.exclude_last_token),
        max_examples=a.max_examples,
        batch_size=a.batch_size,
        device=a.device,
        only_from_first_token=bool(a.only_from_first_token),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = _parse_args()

    out = analyze_from_config(args)


if __name__ == "__main__":
    main()
