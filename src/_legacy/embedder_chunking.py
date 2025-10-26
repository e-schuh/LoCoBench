"""
Document Embedder Module

This module provides classes for embedding documents using various strategies.
It supports both standalone embedding (where each document is embedded individually)
and late-chunking embedding (where documents are embedded as part of a larger context).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
from typing import List, Dict, Union, Literal, Optional, Any, Tuple
import tqdm
import importlib.util
from torch import Tensor
import nnsight
from types import SimpleNamespace


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

ATTN_CALIB_SUPPORTED_MODELS = ["Alibaba-NLP/gte-multilingual-base"]


class BaseEmbedder:
    """
    Base class for document embedders.

    Handles common functionality like model initialization and forward pass.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
    ):
        """
        Initialize the BaseEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        if device is None:
            self.device = DEVICE
        else:
            self.device = device

        # Check if xformers is available for memory-efficient attention
        spec = importlib.util.find_spec("xformers")
        if spec is None:
            print("Warning: xformers not found.")
            xformers_available = False
        else:
            xformers_available = True

        if apply_attn_calibration:
            assert (
                model_name in ATTN_CALIB_SUPPORTED_MODELS
            ), f"Attention calibration is only supported for the following models: {ATTN_CALIB_SUPPORTED_MODELS}"
            print("Loading model using eager attn to support calibration")
            hf_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.float16,
            ).to(self.device)
            self.model = nnsight.NNsight(hf_model)

            self.num_layers = int(self.model.config.num_hidden_layers)

            if calib_layers is None:
                raise ValueError(
                    "Must specify calib_layers when using attention calibration"
                )
            elif calib_layers == "last_half":
                self.calib_layer_idx_start = self.num_layers // 2
                self.calib_layer_idx_end = self.num_layers
            elif calib_layers == "last":
                self.calib_layer_idx_start = self.num_layers - 1
                self.calib_layer_idx_end = self.num_layers
            else:
                raise ValueError("Invalid calibration layer selection")

            if calib_source_tokens is None:
                raise ValueError(
                    "Must specify calib_source_tokens when using attention calibration"
                )
            elif calib_source_tokens == "cls":
                self.calib_source_mode = "cls"
                self.calib_source_token_idx = 0
            elif calib_source_tokens == "all":
                self.calib_source_mode = "all"
            else:
                raise ValueError("Invalid calibration source token selection")

            if calib_basket_size is None:
                raise ValueError(
                    "Must specify calib_basket_size when using attention calibration"
                )
            self.calib_basket_size = calib_basket_size

        elif model_name == "Alibaba-NLP/gte-multilingual-base" and xformers_available:
            print("Loading Alibaba-NLP/gte-multilingual-base with xformers support")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                unpad_inputs=True,
                use_memory_efficient_attention=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        elif model_name == "Qwen/Qwen3-Embedding-0.6B" and xformers_available:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map=self.device,
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)

        self.model.eval()
        self.normalize = normalize
        self.model_name = model_name
        self.apply_attn_calibration = apply_attn_calibration

        # Set up Jina v3 task ID if applicable
        if model_name == "jinaai/jina-embeddings-v3":
            self.task_id = self.model._adaptation_map[jina_v3_task]

    def get_output_calibrated(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        # NOTE: Calibrate either CLS-only or all query rows depending on self.calib_source_mode.
        # Pre-validate inputs OUTSIDE the NNsight trace to surface clean assertion messages.
        mask_in = attention_mask.to(self.device, dtype=torch.long)
        B_in, K_in = mask_in.shape
        assert B_in == input_ids.shape[0], "Batch size mismatch between inputs and mask"
        assert (
            self.calib_basket_size is not None and self.calib_basket_size > 0
        ), "calib_basket_size must be positive"
        S = int(self.calib_basket_size)
        valid_len_in = mask_in.sum(dim=1)
        assert torch.all(
            valid_len_in > 0
        ), "Sequences with zero valid tokens are not supported"
        pos_in = torch.arange(K_in, device=mask_in.device).unsqueeze(0)  # (1,K)
        right_padded_in = (pos_in < valid_len_in.unsqueeze(1)).to(mask_in.dtype)
        if not torch.equal(mask_in, right_padded_in):
            bad = (
                (mask_in != right_padded_in)
                .any(dim=1)
                .nonzero(as_tuple=False)
                .squeeze(-1)
                .tolist()
            )
            raise AssertionError(
                f"Attention mask must be right-padded; bad batch indices: {bad}"
            )
        if not torch.all(valid_len_in >= S):
            too_short = (valid_len_in < S).nonzero(as_tuple=False).squeeze(-1).tolist()
            lens = valid_len_in.tolist()
            raise AssertionError(
                f"calib_basket_size={S} exceeds some sequence lengths; offending indices={too_short}, lengths={lens}"
            )

        with self.model.trace() as tracer:
            with tracer.invoke(input_ids=input_ids, attention_mask=attention_mask):
                for i in range(self.calib_layer_idx_start, self.calib_layer_idx_end):
                    attn = self.model.encoder.layer[
                        i
                    ].attention.source.self__attention_0.source.nn_functional_softmax_0.output

                    # Shape assertions (fail fast) — only use ints in messages
                    assert attn.dim() == 4, "Expected attn to be 4D"
                    B, H, Q, K = attn.shape
                    assert (B, K) == (
                        B_in,
                        K_in,
                    ), "attention_mask shape mismatch vs attn"
                    if getattr(self, "calib_source_mode", "cls") == "cls":
                        assert (
                            self.calib_source_token_idx < Q
                        ), "CLS/source index out of range"

                    # Prepare mask (use precomputed tensors to avoid Proxy in messages)
                    valid_len = valid_len_in.to(device=attn.device)
                    S = int(self.calib_basket_size)
                    pos = torch.arange(K, device=attn.device).unsqueeze(0)  # (1,K)
                    right_padded = (pos < valid_len.unsqueeze(1)).to(dtype=torch.long)

                    # For self-attention we assume Q == K.
                    assert (
                        Q == K
                    ), f"Expected self-attention with Q==K, got Q={Q}, K={K}"

                    # Select which query rows to calibrate: R = 1 for 'cls', else R = Q for 'all'
                    if getattr(self, "calib_source_mode", "cls") == "cls":
                        q_idx = torch.tensor(
                            [self.calib_source_token_idx], device=attn.device
                        )
                    else:
                        q_idx = torch.arange(Q, device=attn.device)
                    R = q_idx.numel()

                    # Common masks and helpers
                    mask_f = right_padded.to(dtype=attn.dtype)  # (B,K)
                    mask_bk = mask_f.view(B, 1, 1, K)  # (B,1,1,K)
                    qmask = right_padded[:, :Q].to(dtype=torch.bool)  # (B,Q)
                    max_baskets = (K + S - 1) // S
                    base_basket_ids = (pos // S).clamp_max(max_baskets - 1)  # (1,K)

                    # Process queries in chunks to lower VRAM
                    r_chunk = 2
                    for start in range(0, R, r_chunk):
                        end = min(start + r_chunk, R)
                        q_idx_chunk = q_idx[start:end]
                        r = q_idx_chunk.numel()
                        assert r > 0

                        # Extract selected query rows: (B,H,r,K)
                        sel_rows = attn.index_select(dim=2, index=q_idx_chunk)
                        sel_rows_masked = sel_rows * mask_bk  # (B,H,r,K)

                        # Query validity mask (to avoid normalizing padded query rows)
                        qmask_sel = qmask.index_select(
                            dim=1, index=q_idx_chunk
                        )  # (B,r)
                        qmask_sel_b = qmask_sel.view(B, 1, r, 1)  # (B,1,r,1)

                        # Basket ids expanded for this chunk
                        basket_ids = base_basket_ids.view(1, 1, 1, K).expand(B, H, r, K)

                        # Sum attention per basket: (B,H,r,max_baskets)
                        bucket_sums = torch.zeros(
                            (B, H, r, max_baskets), device=attn.device, dtype=attn.dtype
                        )
                        bucket_sums = bucket_sums.scatter_add(
                            dim=-1, index=basket_ids, src=sel_rows_masked
                        )

                        # Gather per-position denominators: (B,H,r,K)
                        denom = bucket_sums.gather(dim=-1, index=basket_ids)

                        # Number of baskets per item: ceil(valid_len / S) -> (B,)
                        n_baskets = ((valid_len + (S - 1)) // S).to(dtype=attn.dtype)
                        n_baskets = n_baskets.view(B, 1, 1, 1)  # (B,1,1,1)

                        # Apply calibration formula per position (masked positions remain zero)
                        scaled = sel_rows_masked * S  # (B,H,r,K)
                        calibrated = torch.where(
                            denom > 0,
                            (scaled / denom) * (1.0 / n_baskets),
                            torch.zeros_like(scaled),
                        )

                        # Enforce valid distribution over valid tokens (sum to 1), keep pads at 0
                        calibrated = calibrated * mask_bk
                        row_sum = calibrated.sum(dim=-1, keepdim=True)  # (B,H, r, 1)
                        row_sum_squeezed = row_sum.squeeze(-1)  # (B,H,r)
                        must_be_pos = qmask_sel.view(B, 1, r).expand(B, H, r)
                        assert torch.all(
                            row_sum_squeezed[must_be_pos] > 0
                        ), "Calibrated row sums must be > 0 for valid queries"

                        # Normalize valid query rows, keep invalid query rows unchanged (original masked rows)
                        denom_rows = torch.where(
                            qmask_sel_b > 0, row_sum, torch.ones_like(row_sum)
                        )
                        normalized_chunk = torch.where(
                            qmask_sel_b > 0,
                            calibrated / denom_rows,
                            sel_rows_masked,
                        )

                        # In-place contiguous slice assignment on the NNsight proxy (avoid LongTensor advanced indexing)
                        self.model.encoder.layer[
                            i
                        ].attention.source.self__attention_0.source.nn_functional_softmax_0.output[
                            :, :, start:end, :
                        ] = normalized_chunk

                # Request the final model output so it is materialized after the trace.
                # We expect to access last_hidden_state downstream.
                model_output = self.model.output.save()

        # After exiting the trace context, saved proxies are concrete values.
        # Ensure downstream code can access `.last_hidden_state` as an attribute.
        if hasattr(model_output, "last_hidden_state"):
            return model_output
        # Some HF models return dict-like outputs through NNsight; assert the key exists and wrap.
        assert isinstance(
            model_output, (dict, tuple, list)
        ), "Unexpected model output type from NNsight"
        if isinstance(model_output, dict):
            assert (
                "last_hidden_state" in model_output
            ), "Model output missing 'last_hidden_state'"
            return SimpleNamespace(last_hidden_state=model_output["last_hidden_state"])
        # Fallback for tuple/list where index 0 is last_hidden_state (common HF convention)
        assert len(model_output) > 0, "Empty model output sequence"
        return SimpleNamespace(last_hidden_state=model_output[0])

    def get_model_outputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        """
        Get model outputs with appropriate handling for special models.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for the input

        Returns:
            Model outputs
        """
        # Move tensors to the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Apply attention calibration if enabled
        if self.apply_attn_calibration:
            with torch.no_grad():
                outputs = self.get_output_calibrated(
                    input_ids=input_ids, attention_mask=attention_mask
                )

        # If model is jinaai/jina-embeddings-v3, then add task_id
        elif self.model_name == "jinaai/jina-embeddings-v3":
            adapter_mask = torch.full(
                (input_ids.shape[0],),
                self.task_id,
                dtype=torch.int32,
                device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adapter_mask=adapter_mask,
                )
        # For other models, use the standard forward pass
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs

    def mean_pooling(
        self, last_hidden_state: torch.Tensor, input_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to the model output.

        Args:
            last_hidden_state: Hidden state output from the model
            input_attention_mask: Attention mask to consider only valid tokens

        Returns:
            Pooled embeddings
        """
        # Expand attention mask to match the shape of the model output
        attention_expanded = input_attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        )
        # Sum up the hidden states and divide by the actual sequence length
        sum_hidden_states = torch.sum(last_hidden_state * attention_expanded, dim=1)
        seq_lengths = torch.sum(input_attention_mask, dim=1, keepdim=True)
        return sum_hidden_states / seq_lengths  # (shape: [batch_size, hidden_size])

    def get_embeddings_from_hidden_states(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get both CLS and mean pooled embeddings from hidden states.

        Args:
            last_hidden_state: Hidden state output from the model
            attention_mask: Attention mask for the input

        Returns:
            Dictionary containing both types of embeddings ('cls' and 'mean')
        """
        if self.model_name in ("Qwen/Qwen3-Embedding-0.6B"):
            # For Qwen models, use the last token pooling strategy
            cls_embeddings = self.last_token_pool(last_hidden_state, attention_mask)
        else:
            # For other models, use the first token as CLS token
            cls_embeddings = last_hidden_state[:, 0]
            # (shape: [batch_size, hidden_size])

        # Get mean pooled embeddings
        mean_embeddings = self.mean_pooling(last_hidden_state, attention_mask)

        # Normalize if required
        if self.normalize:
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

        return {"cls": cls_embeddings, "mean": mean_embeddings}

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


class StandaloneEmbedder(BaseEmbedder):
    """
    Class for creating embeddings from standalone documents.

    This class processes documents that have been individually tokenized
    and creates embeddings using a specified encoder model and pooling strategy.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
    ):
        """
        Initialize the StandaloneEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            jina_v3_task=jina_v3_task,
            apply_attn_calibration=apply_attn_calibration,
            calib_layers=calib_layers,
            calib_source_tokens=calib_source_tokens,
            calib_basket_size=calib_basket_size,
        )

    def embed_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Embed a batch of documents and return both CLS and mean pooled embeddings.

        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', etc. for a batch of documents

        Returns:
            Dictionary with document IDs and their corresponding embeddings (both cls and mean)
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch["input_ids"], batch["attention_mask"])
        last_hidden_state = outputs.last_hidden_state

        # Get both types of embeddings
        all_embeddings = self.get_embeddings_from_hidden_states(
            last_hidden_state, batch["attention_mask"].to(self.device)
        )

        # Create result dictionary with primary embeddings based on pooling strategy
        # and also include both types of embeddings
        result = {
            "ids": batch["id"],
            "cls_embeddings": all_embeddings["cls"].cpu(),
            "mean_embeddings": all_embeddings["mean"].cpu(),
        }

        return result

    def embed_dataloader(
        self, dataloader: DataLoader
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Embed all documents in a dataloader.

        Args:
            dataloader: PyTorch DataLoader containing batches of documents to embed

        Returns:
            Dictionary with "cls" and "mean" as top-level keys, each mapping document IDs to their
            corresponding embeddings
        """
        all_embeddings = {"cls": {}, "mean": {}}
        all_ids = []
        all_cls_embedding_tensors = []
        all_mean_embedding_tensors = []

        # Process each batch in the dataloader
        for batch in tqdm.tqdm(dataloader, desc="Embedding documents"):
            batch_result = self.embed_batch(batch)

            # Extract IDs and embeddings from batch result
            batch_ids = batch_result["ids"]
            batch_cls_embeddings = batch_result["cls_embeddings"]
            batch_mean_embeddings = batch_result["mean_embeddings"]

            # Add to lists
            all_ids.extend(batch_ids)
            all_cls_embedding_tensors.append(batch_cls_embeddings)
            all_mean_embedding_tensors.append(batch_mean_embeddings)

        # Concatenate all embedding tensors
        if all_cls_embedding_tensors:
            all_cls_embeddings_tensor = torch.cat(all_cls_embedding_tensors, dim=0)
            all_mean_embeddings_tensor = torch.cat(all_mean_embedding_tensors, dim=0)

            # Create mapping from IDs to embeddings
            for i, doc_id in enumerate(all_ids):
                all_embeddings["cls"][doc_id] = all_cls_embeddings_tensor[i]
                all_embeddings["mean"][doc_id] = all_mean_embeddings_tensor[i]

        return all_embeddings


class LateChunkingEmbedder(BaseEmbedder):
    """
    Class for creating embeddings using the late chunking approach.

    This class processes concatenated documents where multiple text segments
    are combined into a single input. It embeds the entire input at once, then
    extracts embeddings for each segment based on token boundaries.

    For each batch item, there is:
    - One CLS embedding (from the first token of the sequence)
    - One MEAN embedding (mean pooled from the entire sequence)
    - Multiple segment embeddings (one per text segment using mean pooling)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
    ):
        """
        Initialize the LateChunkingEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            jina_v3_task=jina_v3_task,
            apply_attn_calibration=apply_attn_calibration,
            calib_layers=calib_layers,
            calib_source_tokens=calib_source_tokens,
            calib_basket_size=calib_basket_size,
        )

    def embed_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Embed a batch of concatenated documents and extract segment embeddings.

        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', 'doc_boundaries', etc.

        Returns:
            Dictionary mapping batch indices to sub-dictionaries containing:
            - One CLS embedding for the entire sequence
            - One MEAN embedding for the entire sequence
            - Multiple segment embeddings (one per document) using mean pooling
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch["input_ids"], batch["attention_mask"])
        last_hidden_state = outputs.last_hidden_state

        # Process each item in the batch
        batch_results = {}
        for i in range(batch["input_ids"].size(0)):
            doc_boundaries = batch["doc_boundaries"][i]
            doc_ids = batch["doc_ids"][i]

            # Create a context identifier for this sequence of documents
            context_id = batch["id"][i]

            # Get the CLS token embedding (first token of the entire sequence)
            cls_embedding = last_hidden_state[i, 0]
            if self.normalize:
                cls_embedding = F.normalize(cls_embedding, p=2, dim=0)

            # Get the MEAN pooled embedding for the entire sequence
            attention_mask_for_item = (
                batch["attention_mask"][i].unsqueeze(0).to(self.device)
            )
            mean_embedding = self.mean_pooling(
                last_hidden_state[i].unsqueeze(0), attention_mask_for_item
            ).squeeze(0)
            if self.normalize:
                mean_embedding = F.normalize(mean_embedding, p=2, dim=0)

            # Create embeddings for each segment using document boundaries
            segment_embeddings = {}

            for j, (doc_id, boundary) in enumerate(zip(doc_ids, doc_boundaries)):
                # Extract start and end indices (start inclusive, end exclusive)
                start_idx, end_idx = boundary[0], boundary[1]

                # Extract segment tokens from the hidden state
                segment_hidden = last_hidden_state[i, start_idx:end_idx]

                # Skip empty segments
                if segment_hidden.shape[0] == 0:
                    continue

                # Apply mean pooling for the segment
                segment_embedding = torch.mean(segment_hidden, dim=0)

                # Normalize embedding if required
                if self.normalize:
                    segment_embedding = F.normalize(segment_embedding, p=2, dim=0)

                # Create a unique key for this document that includes the context
                unique_doc_key = f"{doc_id}__pos{j}__{context_id}"

                # Store embedding with the unique document ID
                segment_embeddings[unique_doc_key] = segment_embedding.cpu()

            batch_results[context_id] = {
                "cls": cls_embedding.cpu(),
                "mean": mean_embedding.cpu(),
                "segment_embeddings": segment_embeddings,
            }

        return batch_results

    def embed_dataloader(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Embed all documents in a dataloader using the late chunking approach.

        Args:
            dataloader: PyTorch DataLoader containing batches of concatenated documents

        Returns:
            Dictionary mapping:
            - Document IDs to their mean pooled segment embeddings
            - Special CLS keys to the CLS embeddings for each sequence
            - Special MEAN keys to the MEAN embeddings for each sequence
        """
        all_embeddings = {"cls": {}, "mean": {}, "segment_embeddings": {}}

        # Process each batch in the dataloader
        for batch in tqdm.tqdm(dataloader, desc="Embedding documents (late chunking)"):
            batch_results = self.embed_batch(batch)

            # Update the overall results dictionary with embeddings from this batch
            for context_id, batch_embeddings in batch_results.items():
                # Store the CLS embedding for this context
                all_embeddings["cls"][context_id] = batch_embeddings["cls"]
                # Store the MEAN embedding for this context
                all_embeddings["mean"][context_id] = batch_embeddings["mean"]
                # Store the segment embeddings for this context
                all_embeddings["segment_embeddings"].update(
                    batch_embeddings["segment_embeddings"]
                )

        return all_embeddings
