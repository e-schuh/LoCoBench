import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoModel, AutoConfig
from typing import List, Dict, Union, Literal, Optional, Any, Tuple, Callable, Set
import tqdm
import nnsight
import numpy as np

import warnings
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling


@dataclass
class _CalibDeviceTensors:
    """Small per-(device, dtype) tensors, broadcastable against (B, H, R, K)."""

    valid_mask: torch.Tensor              # (B, 1, 1, K), attn dtype
    recip_total: torch.Tensor             # (B, 1, 1, 1) = 1 / n_total_baskets
    q_mask: torch.Tensor                  # (B, Q) bool
    batch_index: torch.Tensor             # (B,) long
    pool_index: Optional[torch.Tensor]    # (B,) long
    bos_index: Optional[torch.Tensor]     # (B,) long
    eos_index: Optional[torch.Tensor]
    bos_share: Optional[torch.Tensor]     # (B, 1, 1, 1)
    eos_share: Optional[torch.Tensor]
    content_budget: Optional[torch.Tensor]  # (B, 1, 1, 1) = 1 - assigned shares


@dataclass
class CalibPlan:
    """Batch geometry for calibration.

    Device- and dtype-independent, so one plan is reused across every
    calibrated layer of a batch.
    """

    source_mode: str
    basket_size: int
    isolate_bos: bool
    isolate_eos: bool
    bos_weight: Union[float, Literal["equal", "original"]]
    eos_weight: Union[float, Literal["equal", "original"]]
    valid_mask: torch.Tensor              # (B, K) bool, host
    n_total: torch.Tensor                 # (B,) long, host
    bos_pos: Optional[List[int]]
    eos_pos: Optional[List[int]]
    content_start: List[int]
    content_len: List[int]
    max_baskets: int
    pool_pos: Optional[List[int]]
    pool_pos_const: Optional[int]         # set when every item pools the same row
    rescale_content: bool
    _cache: Dict[Tuple, _CalibDeviceTensors] = field(
        default_factory=dict, repr=False
    )


class FairSentenceTransformer(SentenceTransformer):
    """SentenceTransformers-compatible embedder with positional fairness calibration.

    This class extends SentenceTransformer to provide attention calibration that mitigates
    positional bias in transformer-based text embeddings. The calibration works by grouping
    tokens into "baskets" of equal size and redistributing attention weights so that each
    basket receives equal total attention, regardless of position in the sequence.

    The calibration is applied during inference via nnsight hooks that intercept and modify
    attention weights after the softmax in specified layers.

    Args:
        model_name_or_path: HuggingFace model identifier or local path to the model.
        device: Device to run the model on (e.g., "cuda", "cpu", "mps"). If None, auto-detected.
        device_map: Device map for multi-GPU or offloading (e.g., "auto"). Mutually exclusive
            with device for placement logic.
        attention_path_override: Custom callable to resolve the attention softmax tensor
            for a given layer index. Signature: (nnsight_model, layer_idx) -> attention_tensor.
            If None, uses predefined paths from ATTENTION_PATHS.
        pooling_override: Force a specific pooling strategy instead of inferring from the model.
            One of "cls" (first token), "mean" (average of all tokens), or "last" (last token).
        calibrated_tokens_override: Force which token positions receive calibrated attention.
            One of "cls", "all", or "last". Defaults to matching the pooling strategy.
        padding_side_override: Force padding side instead of inferring from architecture.
            One of "left" or "right".
        trust_remote_code: Whether to trust remote code when loading the model.
        **kwargs: Additional arguments passed to SentenceTransformer.
    """

    TESTED_MODELS: Tuple[str, ...] = ("Alibaba-NLP/gte-multilingual-base","BAAI/bge-m3", "ibm-granite/granite-embedding-278m-multilingual", "ibm-granite/granite-embedding-107m-multilingual","microsoft/harrier-oss-v1-0.6b", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B")

    DECODER_ARCH_NAMES: Set[str] = {"qwen", "llama", "gpt", "mistral", "falcon"}

    ATTENTION_PATHS: Dict[str, Union[str, Callable[[Any, int], Any]]] = {
        "Alibaba-NLP/gte-multilingual-base": "encoder.layer[{i}].attention.source.self__attention_0.source.nn_functional_softmax_0.output",
        "BAAI/bge-m3": "encoder.layer[{i}].attention.self.source.nn_functional_softmax_0.output",
        "ibm-granite/granite-embedding-278m-multilingual": "encoder.layer[{i}].attention.self.source.nn_functional_softmax_0.output",
        "ibm-granite/granite-embedding-107m-multilingual": "encoder.layer[{i}].attention.self.source.nn_functional_softmax_0.output",
        "microsoft/harrier-oss-v1-0.6b": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-0.6B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-4B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-8B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "jinaai/jina-embeddings-v3": "roberta.encoder.layers[{i}].mixer.inner_attn.source.torch_softmax_0.output",
        "NovaSearch/stella_en_400M_v5": "encoder.layer[{i}].attention.source.self__attention_0.source.nn_functional_softmax_0.output",
    }

    ############################################################################
    #                               NOTES                                      #
    ############################################################################
    # - For NovaSearch/stella_en_400M_v5, the following changes need to be made in modeling.py:
    #   1. change line 684: from attn_implementation=None to attn_implementation="eager"
    #   2. change line 689: from != to ==
    #   3. add new line 940: unpad_inputs = False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        attention_path_override: Optional[Callable[[Any, int], Any]] = None,
        pooling_override: Optional[Literal["cls", "mean", "last"]] = None,
        calibrated_tokens_override: Optional[Literal["cls", "all", "last"]] = None,
        padding_side_override: Optional[Literal["left", "right"]] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        self._torch_dtype = (
            torch.float16
            if torch.cuda.is_available() or torch.backends.mps.is_available()
            else None
        )
        self._config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self._padding_side_override = padding_side_override
        self.padding_side = self._detect_padding_side()

        original_class_name = self.__class__.__name__
        self.__class__.__name__ = "SentenceTransformer"
        try:
            super().__init__(
                model_name_or_path,
                device=device,
                trust_remote_code=trust_remote_code,
                model_kwargs={
                    "device_map": device_map,
                    "torch_dtype": self._torch_dtype,
                },
                tokenizer_kwargs={"padding_side": self.padding_side},
                **kwargs,
            )
        finally:
            self.__class__.__name__ = original_class_name

        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._attention_path_override = attention_path_override
        self._pooling_override = pooling_override
        self._calibrated_tokens_override = calibrated_tokens_override
        if model_name_or_path not in self.TESTED_MODELS:
            warnings.warn(
                f"Model '{model_name_or_path}' is untested; tested models: {self.TESTED_MODELS}",
                RuntimeWarning,
            )

        self.pooling_strategy = self._infer_pooling_strategy()
        self.calib_source_mode = self._calibration_source_mode()

        self._nnsight_model: Optional[nnsight.NNsight] = None
        self._num_layers: Optional[int] = None

        # Default calibration parameters (computed once).
        total_layers = int(self._config.num_hidden_layers)
        self._default_calib_basket_size: int = 128
        self._default_calib_layers: int = max(1, total_layers // 2)
        self._default_calib_strength: float = 0.5

        print(
            "Summary of TextEmbedder configuration:"
            f"\n  Model: {model_name_or_path}"
            f"\n  Padding side: {self.padding_side}"
            f"\n  Pooling strategy: {self.pooling_strategy}"
            f"\n  Calibrated token(s): {self.calib_source_mode}"
            f"\n  Default calib_basket_size: {self._default_calib_basket_size}"
            f"\n  Default calib_layers: {self._default_calib_layers}"
            f"\n  Default calib_strength: {self._default_calib_strength}"
        )

    def _detect_padding_side(self) -> Literal["left", "right"]:
        override = self._padding_side_override
        if override is not None:
            assert override in ("left", "right")
            return override
        model_type = str(getattr(self._config, "model_type", "")).lower()
        if any(name in model_type for name in self.DECODER_ARCH_NAMES):
            return "left"
        return "right"

    def _infer_pooling_strategy(self) -> Literal["cls", "mean", "last"]:
        """Infer pooling strategy from ST pooling module; fallback to CLS."""
        override = self._pooling_override
        if override is not None:
            if override in ("cls", "mean", "last"):
                return override
            raise AssertionError(f"Unsupported pooling override: {override}")

        pooling_modules = [m for m in self._modules.values() if isinstance(m, Pooling)]
        if pooling_modules:
            pooling = pooling_modules[0]
            if getattr(pooling, "pooling_mode_lasttoken", False):
                return "last"
            if getattr(pooling, "pooling_mode_cls_token", False):
                return "cls"
            if getattr(pooling, "pooling_mode_mean_tokens", False):
                return "mean"
        warnings.warn(
            "Pooling strategy could not be inferred; defaulting to first token (CLS)",
            RuntimeWarning,
        )
        return "cls"

    def _calibration_source_mode(self) -> Literal["cls", "all", "last"]:
        """Determine which token(s) to use for attention calibration."""
        override = self._calibrated_tokens_override
        if override is not None:
            if override in ("cls", "all", "last"):
                return override
            raise AssertionError(f"Unsupported calibrated tokens override: {override}")

        if self.pooling_strategy == "cls":
            return "cls"
        if self.pooling_strategy == "mean":
            return "all"
        if self.pooling_strategy == "last":
            return "last"
        raise AssertionError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def _resolve_attention_softmax(self, layer_idx: int):
        assert self._nnsight_model is not None
        resolver = self._attention_path_override
        if resolver is None:
            resolver = self._default_attention_path(self.model_name_or_path)
            self._attention_path_override = resolver # update here.
        assert (
            resolver is not None
        ), f"No attention path resolver for {self.model_name_or_path}"
        return resolver(self._nnsight_model, layer_idx)

    @staticmethod
    def _compile_attention_path(path_template: str) -> Callable[[Any, int], Any]:
        def resolver(model: Any, layer_idx: int) -> Any:
            path = path_template.format(i=layer_idx)
            target = model
            for segment in path.split("."):
                while "[" in segment:
                    attr, bracket, rest = segment.partition("[")
                    assert bracket == "["
                    index_str, closing, remainder = rest.partition("]")
                    assert closing == "]"
                    if attr:
                        target = getattr(target, attr)
                    idx = int(index_str)
                    target = target[idx]
                    segment = remainder
                if segment:
                    target = getattr(target, segment)
            return target

        return resolver

    def _default_attention_path(
        self, model_name: str
    ) -> Optional[Callable[[Any, int], Any]]:
        entry = self.ATTENTION_PATHS.get(model_name)
        if entry is None:
            return None
        if isinstance(entry, str):
            return self._compile_attention_path(entry)
        return entry

    def _ensure_nnsight_model(self) -> None:
        if self._nnsight_model is not None:
            return
        hf_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="eager",
            device_map=self.device_map,
            torch_dtype=self._torch_dtype,
        )
        if self.device_map is None:
            hf_model = hf_model.to(self.device)
        self._num_layers = int(hf_model.config.num_hidden_layers)
        self._nnsight_model = nnsight.NNsight(hf_model)

    def _validate_padding_mask(
        self, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.padding_side in ("left", "right")
        valid_len = attention_mask.sum(dim=1)
        K = attention_mask.size(1)
        pos = torch.arange(K, device=attention_mask.device).unsqueeze(0)
        if self.padding_side == "right":
            expected = (pos < valid_len.unsqueeze(1)).to(attention_mask.dtype)
            start_idx = torch.zeros_like(valid_len, dtype=torch.long)
        else:
            start_idx = (K - valid_len).clamp_min(0).to(torch.long)
            expected = (pos >= start_idx.unsqueeze(1)).to(attention_mask.dtype)
        assert torch.equal(
            attention_mask, expected
        ), f"Attention mask must be {self.padding_side}-padded"
        return expected.to(dtype=torch.bool), start_idx, valid_len.to(torch.long)

    def _build_calib_plan(
        self,
        attention_mask: torch.Tensor,
        basket_size: int,
        isolate_bos: bool,
        isolate_eos: bool,
        bos_weight: Union[float, Literal["equal", "original"]],
        eos_weight: Union[float, Literal["equal", "original"]],
    ) -> CalibPlan:
        """Build the reusable batch geometry.

        Reads `attention_mask` on the host, so call this once per batch and
        outside the nnsight trace, never on a proxy.
        """
        assert basket_size > 0
        assert attention_mask.dim() == 2
        for w in (bos_weight, eos_weight):
            assert w in ("equal", "original") or (
                isinstance(w, (int, float)) and 0.0 <= w <= 1.0
            ), (
                "weight must be a float in [0, 1], 'equal' or 'original', "
                f"got {w!r}"
            )

        valid_mask, start_idx, valid_len = self._validate_padding_mask(attention_mask)
        assert torch.all(valid_len > 0)

        S = int(basket_size)
        n_isolated = int(isolate_bos) + int(isolate_eos)
        content_len = valid_len - n_isolated
        assert torch.all(
            content_len > 0
        ), "content_len must be > 0 after isolating BOS/EOS"

        end_idx = start_idx + valid_len - 1
        content_start = start_idx + int(isolate_bos)
        n_baskets = (content_len + S - 1) // S
        n_total = n_baskets + n_isolated

        assigned = sum(
            float(w)
            for iso, w in ((isolate_bos, bos_weight), (isolate_eos, eos_weight))
            if iso and isinstance(w, (int, float))
        )
        assert assigned < 1.0, "isolated token weights must leave mass for content"

        source_mode = self.calib_source_mode
        if source_mode == "cls":
            pool = start_idx
        elif source_mode == "last":
            pool = end_idx
        elif source_mode == "all":
            pool = None
        else:
            raise AssertionError(f"Unsupported source_mode: {source_mode}")

        # One host sync per batch, rather than one per calibrated layer.
        pool_list = pool.tolist() if pool is not None else None
        const = (
            pool_list[0]
            if pool_list is not None and len(set(pool_list)) == 1
            else None
        )

        return CalibPlan(
            source_mode=source_mode,
            basket_size=S,
            isolate_bos=isolate_bos,
            isolate_eos=isolate_eos,
            bos_weight=bos_weight,
            eos_weight=eos_weight,
            valid_mask=valid_mask,
            n_total=n_total,
            bos_pos=start_idx.tolist() if isolate_bos else None,
            eos_pos=end_idx.tolist() if isolate_eos else None,
            content_start=content_start.tolist(),
            content_len=content_len.tolist(),
            max_baskets=int(n_baskets.max().item()),
            pool_pos=pool_list,
            pool_pos_const=const,
            rescale_content=(
                (isolate_bos and bos_weight != "equal")
                or (isolate_eos and eos_weight != "equal")
            ),
        )

    @staticmethod
    def _calib_device_tensors(
        plan: CalibPlan, device: Any, dtype: torch.dtype
    ) -> _CalibDeviceTensors:
        """Materialise the plan for one device and dtype, cached.

        Deriving both from `attn` rather than from the plan keeps this correct
        under `device_map` sharding, where layers live on different devices.
        """
        key = (str(device), dtype)
        cached = plan._cache.get(key)
        if cached is not None:
            return cached

        B = plan.valid_mask.size(0)
        valid = plan.valid_mask.to(device)
        n_total_f = plan.n_total.to(device=device, dtype=dtype).view(B, 1, 1, 1)

        def share(w: Union[float, Literal["equal", "original"]]):
            if w == "equal":
                return 1.0 / n_total_f
            if w == "original":
                return None      # depends on attention values, resolved per block
            return torch.full_like(n_total_f, float(w))

        def index(pos: Optional[List[int]]) -> Optional[torch.Tensor]:
            if pos is None:
                return None
            return torch.tensor(pos, device=device, dtype=torch.long)

        bos_share = share(plan.bos_weight) if plan.isolate_bos else None
        eos_share = share(plan.eos_weight) if plan.isolate_eos else None

        # Only precomputable when no share depends on the attention values.
        uses_original = (plan.isolate_bos and plan.bos_weight == "original") or (
            plan.isolate_eos and plan.eos_weight == "original"
        )
        budget = None
        if plan.rescale_content and not uses_original:
            assigned = torch.zeros_like(n_total_f)
            if bos_share is not None:
                assigned = assigned + bos_share
            if eos_share is not None:
                assigned = assigned + eos_share
            budget = 1.0 - assigned

        pool_index = index(plan.pool_pos)
        if pool_index is None:
            q_mask = valid
        else:
            q_mask = torch.zeros_like(valid)
            q_mask[torch.arange(B, device=device), pool_index] = True

        mat = _CalibDeviceTensors(
            valid_mask=valid.view(B, 1, 1, -1).to(dtype),
            recip_total=1.0 / n_total_f,
            q_mask=q_mask,
            batch_index=torch.arange(B, device=device),
            pool_index=pool_index,
            bos_index=index(plan.bos_pos),
            eos_index=index(plan.eos_pos),
            bos_share=bos_share,
            eos_share=eos_share,
            content_budget=budget,
        )
        plan._cache[key] = mat
        return mat

    @staticmethod
    def _calibrate_block(
        block: torch.Tensor,
        plan: CalibPlan,
        mat: _CalibDeviceTensors,
        check_sel: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """block: (B, H, R, K) copy of the rows to calibrate. Returns a new tensor."""
        B, H, R, K = block.shape
        S = plan.basket_size
        P = plan.max_baskets * S

        # Align each item's contiguous content span to column 0. This drops
        # padding and isolated tokens and lines the basket grid up in one move.
        # The loop is needed because content_start varies per item under left
        # padding.
        bi = mat.batch_index

        def share(weight, precomputed, index):
            """(B, 1, 1, 1) for a fixed share, (B, H, R, 1) for "original"."""
            if index is None:
                return None
            if weight == "original":
                # Mass the uncalibrated row already put on this token. Reading
                # it from `block` costs one (B, H, R) gather, no full-size copy.
                return block[bi, :, :, index].unsqueeze(-1)
            return precomputed

        bos_share = share(plan.bos_weight, mat.bos_share, mat.bos_index)
        eos_share = share(plan.eos_weight, mat.eos_share, mat.eos_index)

        aligned = block.new_zeros((B, H, R, P))
        for b in range(B):
            c0, L = plan.content_start[b], plan.content_len[b]
            aligned[b, :, :, :L] = block[b, :, :, c0 : c0 + L]

        # Baskets are contiguous blocks of S, so a reshape replaces
        # scatter_add/gather. No int64 index tensor is built.
        view = aligned.view(B, H, R, plan.max_baskets, S)
        sums = view.sum(dim=-1, keepdim=True)
        # Attention is non-negative, so a zero basket sum implies a zero
        # numerator. Filling the divisor with 1 reproduces where(denom > 0, ...).
        sums.masked_fill_(sums == 0, 1.0)
        view.div_(sums)
        aligned.mul_(mat.recip_total)

        if plan.rescale_content:
            budget = mat.content_budget
            if budget is None:
                assigned = torch.zeros_like(block[:, :1, :, :1])
                for sh in (bos_share, eos_share):
                    if sh is not None:
                        assigned = assigned + sh
                budget = (1.0 - assigned).clamp_min(0.0)
            content_sum = aligned.sum(dim=-1, keepdim=True)
            scale = torch.where(
                content_sum > 0,
                budget / content_sum,
                torch.ones_like(content_sum),
            )
            aligned.mul_(scale)

        out = torch.zeros_like(block)
        for b in range(B):
            c0, L = plan.content_start[b], plan.content_len[b]
            out[b, :, :, c0 : c0 + L] = aligned[b, :, :, :L]
        del aligned

        # squeeze(-1) leaves (B, 1, 1) for a fixed share and (B, H, R) for
        # "original"; both broadcast against the (B, H, R) indexed slice.
        if bos_share is not None:
            out[bi, :, :, mat.bos_index] += bos_share.squeeze(-1)
        if eos_share is not None:
            out[bi, :, :, mat.eos_index] += eos_share.squeeze(-1)

        row_sum = out.sum(dim=-1, keepdim=True)
        if check_sel is not None:
            assert torch.all(
                row_sum.masked_select(check_sel) > 0
            ), "Calibrated rows must sum to > 0"
        row_sum.masked_fill_(row_sum == 0, 1.0)
        out.div_(row_sum)
        return out

    def _calibrate_attention_inplace(
        self,
        attn: torch.Tensor,
        plan: CalibPlan,
        strength: float,
        tile: int = 256,
        zero_padded_keys: bool = True,
        check_rows: bool = True,
    ) -> None:
        """Redistribute attention across baskets, editing `attn` in place.

        `attn` has shape (B, H, Q, K). For "cls" and "last" only one query row
        per batch item changes, so the working set is (B, H, K); for "all" the
        query axis is processed in tiles of `tile` rows.

        zero_padded_keys: zero the padded key columns of every row. Softmax
            already drives those to exactly 0 except on query rows where every
            key is masked, which occur with causal attention plus left padding;
            there softmax renormalises over the padded keys instead. Those rows
            sit at padding positions and their outputs are discarded by every
            pooling strategy, but the flag costs one in-place pass and no
            allocation.
        check_rows: keep the assertion that calibrated rows sum to > 0. Costs
            one host sync per call.
        """
        assert attn.dim() == 4, f"Expected attention 4D, got {tuple(attn.shape)}"
        B, H, Q, K = attn.shape
        assert Q == K, "Self-attention expected (Q == K)"
        assert plan.valid_mask.shape == (B, K), (
            f"plan built for {tuple(plan.valid_mask.shape)}, got batch {(B, K)}"
        )
        assert 0.0 <= strength <= 1.0
        assert tile > 0

        mat = self._calib_device_tensors(plan, attn.device, attn.dtype)
        # Must precede the row copies below: `finish` blends against the masked
        # attention, so `orig` has to be captured after this multiply.
        if zero_padded_keys:
            attn.mul_(mat.valid_mask)

        def finish(out: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
            if strength != 1.0:
                out.mul_(strength).add_(orig, alpha=1.0 - strength)
            return out.mul_(mat.valid_mask)

        if plan.pool_pos is not None:
            # One query row per item changes; every other row is left untouched.
            # `orig` is a copy, not a view, and `_calibrate_block` does not
            # mutate it, so the strength blend still sees the pre-edit values.
            # The write into `attn` is the last step.
            if plan.pool_pos_const is not None:
                p = plan.pool_pos_const
                orig = attn[:, :, p : p + 1, :].clone()
            else:
                orig = attn[mat.batch_index, :, mat.pool_index, :].unsqueeze(2)
            sel = (
                torch.ones_like(orig[:, :1, :, :1], dtype=torch.bool)
                if check_rows
                else None
            )
            out = finish(self._calibrate_block(orig, plan, mat, sel), orig)
            if plan.pool_pos_const is not None:
                attn[:, :, plan.pool_pos_const : plan.pool_pos_const + 1, :] = out
            else:
                attn[mat.batch_index, :, mat.pool_index, :] = out.squeeze(2)
            return

        for q0 in range(0, Q, tile):
            q1 = min(q0 + tile, Q)
            # Safe to write tiles back as we go: each output row depends only
            # on the same input row, never on other query rows.
            orig = attn[:, :, q0:q1, :].clone()
            sel = mat.q_mask[:, q0:q1].view(B, 1, q1 - q0, 1)
            # Only rows the calibration selects; the rest keep their values.
            out = self._calibrate_block(orig, plan, mat, sel if check_rows else None)
            attn[:, :, q0:q1, :] = torch.where(sel, finish(out, orig), orig)

    def _extract_last_hidden_state(self, model_output: Any) -> torch.Tensor:
        if hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state
        if isinstance(model_output, dict):
            assert "last_hidden_state" in model_output, "Missing last_hidden_state"
            return model_output["last_hidden_state"]
        assert isinstance(model_output, (list, tuple)) and len(model_output) > 0
        return model_output[0]

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask.to(hidden.device)
        valid_mask, start_idx, valid_len = self._validate_padding_mask(attention_mask)
        if self.pooling_strategy == "cls":
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), start_idx
            ]
        elif self.pooling_strategy == "mean":
            pooled = self._mean_pool(hidden, attention_mask)
        elif self.pooling_strategy == "last":
            last_idx = start_idx + valid_len - 1
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), last_idx
            ]
        else:
            raise AssertionError(
                f"Unsupported pooling strategy: {self.pooling_strategy}"
            )
        assert pooled.dim() == 2 and pooled.size(0) == hidden.size(0)
        return pooled

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1)
        return summed / lengths

    def _resolve_isolation_defaults(
        self,
        isolate_bos: Optional[bool],
        isolate_eos: Optional[bool],
    ) -> Tuple[bool, bool]:
        if self.pooling_strategy == "mean":
            default_bos, default_eos = False, False
        elif self.pooling_strategy == "cls":
            default_bos, default_eos = True, False
        elif self.pooling_strategy == "last":
            default_bos, default_eos = True, True
        else:
            default_bos, default_eos = True, True
        resolved_bos = default_bos if isolate_bos is None else isolate_bos
        resolved_eos = default_eos if isolate_eos is None else isolate_eos
        return resolved_bos, resolved_eos

    def encode_positionally_fair(
        self,
        sentences: Union[str, List[str]],
        *,
        calib_basket_size: Optional[int] = None,
        calib_layers: Optional[int] = None,
        calib_strength: Optional[float] = None,
        isolate_bos: Optional[bool] = None,
        isolate_eos: Optional[bool] = None,
        bos_weight: Union[float, Literal["equal", "original"]] = "original",
        eos_weight: Union[float, Literal["equal", "original"]] = "equal",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode sentences with positional fairness calibration applied to attention weights.

        Tokens (excluding optionally isolated BOS/EOS) are grouped into consecutive baskets
        of size `calib_basket_size`. Attention weights are redistributed so each basket
        receives equal total attention (1/n_baskets), eliminating positional bias where
        early or late tokens dominate the embedding.

        Sentences shorter than `calib_basket_size` (after accounting for isolated tokens)
        fall back to standard `.encode()` without calibration.

        Args:
            sentences: Single sentence string or list of sentences to encode.
            calib_basket_size: Number of consecutive content tokens per basket. Larger values
                create coarser calibration; smaller values provide finer-grained fairness.
                Defaults to 128 (set during __init__).
            calib_layers: Number of final transformer layers to apply calibration to.
                E.g., calib_layers=3 calibrates the last 3 layers.
                Defaults to 50% of the model's layers (set during __init__).
            calib_strength: Interpolation factor between calibrated (1.0) and original (0.0)
                attention. Value in [0, 1]. Defaults to 0.5 (set during __init__).
            isolate_bos: Whether to treat BOS token as its own basket. If None, inferred
                from pooling strategy (True for cls/last pooling, False for mean).
            isolate_eos: Whether to treat EOS token as its own basket. If None, inferred
                from pooling strategy (True for last pooling, False otherwise).
            bos_weight: Share of the attention row assigned to the BOS token. One of
                "equal" (1/n_baskets, the same as a content basket), "original"
                (whatever mass the uncalibrated attention already gave the token,
                so BOS is left alone and only content positions are rebalanced),
                or a float in [0, 1] giving a fixed share. Content baskets split
                the remainder equally. Ignored when isolate_bos is False.
            eos_weight: Share assigned to the EOS token, same semantics as bos_weight.
                Ignored when isolate_eos is False.
            batch_size: Number of sentences to process simultaneously.
            show_progress_bar: Whether to display a tqdm progress bar during encoding.
            normalize_embeddings: Whether to L2-normalize the output embeddings.
            device: Device to run inference on. If None, uses the model's default device.
            convert_to_numpy: Return embeddings as numpy array (default True).
            convert_to_tensor: Return embeddings as torch.Tensor. Takes precedence over
                convert_to_numpy if both are True.

        Returns:
            Embeddings with shape (n_sentences, embedding_dim) as numpy array or torch.Tensor.
        """
        # Apply defaults computed in __init__ for any unspecified calibration params.
        if calib_basket_size is None:
            calib_basket_size = self._default_calib_basket_size
        if calib_layers is None:
            calib_layers = self._default_calib_layers
        if calib_strength is None:
            calib_strength = self._default_calib_strength

        assert calib_basket_size > 0
        assert calib_layers > 0
        assert 0.0 <= calib_strength <= 1.0

        resolved_bos, resolved_eos = self._resolve_isolation_defaults(
            isolate_bos, isolate_eos
        )

        if isinstance(bos_weight, (int, float)):
            assert 0.0 <= bos_weight <= 1.0, "bos_weight must be in [0, 1]"
        else:
            assert bos_weight in ("equal", "original"), (
                f"bos_weight must be float, 'equal' or 'original', got {bos_weight}"
            )
        if isinstance(eos_weight, (int, float)):
            assert 0.0 <= eos_weight <= 1.0, "eos_weight must be in [0, 1]"
        else:
            assert eos_weight in ("equal", "original"), (
                f"eos_weight must be float, 'equal' or 'original', got {eos_weight}"
            )

        float_share = sum(
            float(w)
            for w, iso in ((bos_weight, resolved_bos), (eos_weight, resolved_eos))
            if iso and isinstance(w, (int, float))
        )
        assert float_share < 1.0, "isolated token weights must leave mass for content"

        self._ensure_nnsight_model()
        assert self._nnsight_model is not None and self._num_layers is not None
        assert calib_layers <= self._num_layers, "calib_layers exceeds model depth"

        if isinstance(sentences, str):
            sentences_list: List[str] = [sentences]
        else:
            sentences_list = list(sentences)
        assert len(sentences_list) > 0

        use_device = (
            torch.device(device) if device is not None else torch.device(self.device)
        )

        n_isolated = int(resolved_bos) + int(resolved_eos)
        preflight_enc = self.tokenizer(
            sentences_list, padding=False, truncation=True, return_tensors=None
        )
        content_lens = [len(ids) - n_isolated for ids in preflight_enc["input_ids"]]
        short_idxs = [i for i, cl in enumerate(content_lens) if cl < calib_basket_size]
        calib_idxs = [i for i, cl in enumerate(content_lens) if cl >= calib_basket_size]

        if short_idxs:
            print(
                f"[Info encode_positionally_fair:] {len(short_idxs)} sample(s) have content "
                f"length < basket size ({calib_basket_size}), using .encode() fallback"
            )

        if not calib_idxs:
            return self.encode(
                sentences_list,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                device=device,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
            )

        calib_sentences = [sentences_list[i] for i in calib_idxs]

        all_embeddings: List[torch.Tensor] = []
        rng = range(0, len(calib_sentences), batch_size)
        iterator = tqdm.tqdm(rng, desc="Encoding (fair)", disable=not show_progress_bar)
        for start in iterator:
            end = min(start + batch_size, len(calib_sentences))
            chunk = calib_sentences[start:end]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids: torch.Tensor = enc["input_ids"]
            attention_mask: torch.Tensor = enc["attention_mask"]
            assert input_ids.dim() == 2 and attention_mask.dim() == 2
            B, L = input_ids.shape
            assert attention_mask.shape == (B, L)

            if self.device_map is None:
                input_ids = input_ids.to(use_device)
                attention_mask = attention_mask.to(use_device)

            # Built once per batch, outside the trace: the padding validation
            # and the basket geometry do not change between layers.
            calib_plan = self._build_calib_plan(
                attention_mask,
                basket_size=calib_basket_size,
                isolate_bos=resolved_bos,
                isolate_eos=resolved_eos,
                bos_weight=bos_weight,
                eos_weight=eos_weight,
            )

            with torch.no_grad():
                with self._nnsight_model.trace() as tracer:
                    with tracer.invoke(
                        input_ids=input_ids, attention_mask=attention_mask
                    ):
                        layer_start = max(0, self._num_layers - calib_layers)
                        for idx in range(layer_start, self._num_layers):
                            attn = self._resolve_attention_softmax(idx)
                            self._calibrate_attention_inplace(
                                attn, calib_plan, strength=calib_strength
                            )
                        model_output = self._nnsight_model.output.save()

            hidden = self._extract_last_hidden_state(model_output)
            pooled = self._pool(hidden, attention_mask)
            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu())
            del model_output, hidden, pooled, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        calib_embeddings = torch.cat(all_embeddings, dim=0)
        assert calib_embeddings.shape[0] == len(calib_idxs)

        if not short_idxs:
            embeddings = calib_embeddings
        else:
            short_sentences = [sentences_list[i] for i in short_idxs]
            short_embeddings = self.encode(
                short_sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                device=device,
                convert_to_numpy=False,
                convert_to_tensor=True,
            )
            assert short_embeddings.shape[0] == len(short_idxs)
            D = calib_embeddings.shape[1]
            assert short_embeddings.shape[1] == D
            embeddings = torch.empty(
                (len(sentences_list), D), dtype=calib_embeddings.dtype
            )
            for out_i, orig_i in enumerate(calib_idxs):
                embeddings[orig_i] = calib_embeddings[out_i]
            for out_i, orig_i in enumerate(short_idxs):
                embeddings[orig_i] = short_embeddings[out_i]

        assert embeddings.shape[0] == len(sentences_list)
        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.detach().numpy()
        return embeddings
