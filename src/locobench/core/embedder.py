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


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


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

        if model_name == "Alibaba-NLP/gte-multilingual-base" and xformers_available:
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

        # Set up Jina v3 task ID if applicable
        if model_name == "jinaai/jina-embeddings-v3":
            self.task_id = self.model._adaptation_map[jina_v3_task]

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

        # If model is jinaai/jina-embeddings-v3, then add task_id
        if self.model_name == "jinaai/jina-embeddings-v3":
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
