"""The looped transformer inside Megatron's model lifecycle (data parallelism, BF16 wrapping, checkpointing).

Megatron allocates the P + R + C physical layers; the loops over the recurrent block happen in the model's forward.
Tensor, pipeline and context parallelism are not supported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.training.argument_utils import gpt_config_from_args
from megatron.training.models.base import ModelBuilder, compose_hooks
from megatron.training.models.dist_utils import unimodal_build_distributed_models
from megatron.training.models.gpt import GPTModelConfig
from megatron.training.vocab_utils import calculate_padded_vocab_size
from torch import Tensor

from looptf.model import LoopedTransformer, ModelConfig
from looptf.model.losses import token_cross_entropy


@dataclass(kw_only=True)
class LoopedMegatronConfig(GPTModelConfig):
    builder: ClassVar[str] = "looptf.megatron.model.LoopedMegatronBuilder"
    prelude_layers: int = 2
    recurrent_layers: int = 4
    coda_layers: int = 2
    loops: int = 1
    backprop_loops: int = 1
    untie_loops: bool = False
    recompute_loops: bool = False


def build_model_config(args: Any) -> LoopedMegatronConfig:
    config = gpt_config_from_args(args, model_config_cls=LoopedMegatronConfig)
    config.vocab_size = args.vocab_size
    config.should_pad_vocab = True
    for name in ("prelude_layers", "recurrent_layers", "coda_layers", "loops", "backprop_loops", "untie_loops",
                 "recompute_loops"):
        setattr(config, name, getattr(args, name))
    return config


class LoopedMegatronModel(MegatronModule):
    def __init__(self, config: LoopedMegatronConfig, *, padded_vocab_size: int) -> None:
        super().__init__(config.transformer)
        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = self.post_process = True
        self.looped_config = config
        transformer = config.transformer
        self.model = LoopedTransformer(ModelConfig(
            vocab_size=config.vocab_size,
            padded_vocab_size=padded_vocab_size,
            hidden_size=int(transformer.hidden_size),
            intermediate_size=int(transformer.ffn_hidden_size),
            num_attention_heads=int(transformer.num_attention_heads),
            head_dim=int(transformer.hidden_size) // int(transformer.num_attention_heads),
            max_position_embeddings=int(config.seq_length),
            rope_theta=float(config.rotary_base),
            rms_norm_eps=float(transformer.layernorm_epsilon),
            num_prelude_layers=config.prelude_layers,
            num_recurrent_layers=config.recurrent_layers,
            num_coda_layers=config.coda_layers,
            recurrence_steps=config.loops,
            backprop_steps=config.backprop_loops,
            untie_loops=config.untie_loops,
            gradient_checkpointing=config.recompute_loops,
        ))

    def set_input_tensor(self, input_tensor) -> None:
        """Called by Megatron's schedules also without pipeline parallelism."""
        tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if any(tensor is not None for tensor in tensors):
            raise ValueError("pipeline parallelism is not supported")

    def forward(self, input_ids: Tensor, *, labels: Tensor, num_steps: int, backprop_steps: int) -> Tensor:
        """Per-token cross-entropy, [batch, sequence]."""
        output = self.model(input_ids, num_steps=num_steps, backprop_steps=backprop_steps)
        return token_cross_entropy(output.logits, labels)


class LoopedMegatronBuilder(ModelBuilder[LoopedMegatronModel, LoopedMegatronConfig]):
    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> LoopedMegatronModel:
        if pg_collection.pp.size() != 1 or vp_stage is not None or pg_collection.tp.size() != 1:
            raise ValueError("tensor and pipeline parallelism are not supported")
        if pre_process is False or post_process is False:
            raise ValueError("the model runs as a single pipeline stage")
        config = self._model_config
        padded = calculate_padded_vocab_size(config.vocab_size, config.make_vocab_size_divisible_by, 1)
        return LoopedMegatronModel(config, padded_vocab_size=padded)

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
        use_layer_wise_distributed_optimizer: bool = False,
        use_layer_wise_param_layout: bool = True,
    ) -> list[LoopedMegatronModel]:
        config = self._model_config
        models = unimodal_build_distributed_models(
            self.build_model,
            config.transformer,
            pg_collection,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp,
            use_torch_fsdp2,
            wrap_with_ddp,
            data_parallel_random_init,
            mixed_precision_wrapper,
            compose_hooks(config.pre_wrap_hooks),
            model_type,
            use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            use_layer_wise_param_layout=use_layer_wise_param_layout,
        )
        updated = compose_hooks(config.post_wrap_hooks)(models)
        return models if updated is None else updated
